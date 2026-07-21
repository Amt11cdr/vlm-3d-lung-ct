import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from data_common import load_labels, get_split, compute_multilabel_metrics, CACHE_DIR, SEED

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# -------- DATASET --------
class LungDataset2p5D(Dataset):
    def __init__(self, files, file_to_labels):
        self.files = files
        self.file_to_labels = file_to_labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        filename = os.path.basename(file_path).replace(".nii.gz", "")
        cache_path = os.path.join(CACHE_DIR, filename + ".npy")

        if os.path.exists(cache_path):
            slice_stack = torch.from_numpy(np.load(cache_path)).float()
        else:
            data = nib.load(file_path).get_fdata()
            z = data.shape[2] // 2
            z_minus = max(z - 1, 0)
            z_plus = min(z + 1, data.shape[2] - 1)
            slice_stack = np.stack([data[:, :, z_minus], data[:, :, z], data[:, :, z_plus]], axis=0)
            slice_stack = (slice_stack - np.mean(slice_stack)) / (np.std(slice_stack) + 1e-8)
            slice_stack = torch.tensor(slice_stack).float()
            slice_stack = F.interpolate(slice_stack.unsqueeze(0), size=(224, 224),
                                         mode='bilinear', align_corners=False).squeeze(0)

        label = torch.tensor(self.file_to_labels[filename]).float()  # shape (num_labels,)
        return slice_stack, label


def main():
    file_to_labels, label_cols = load_labels()
    num_labels = len(label_cols)
    print(f"{num_labels} pathology labels: {label_cols}")

    train_files, test_files = get_split(file_to_labels)
    print(f"Train files: {len(train_files)} | Test files: {len(test_files)}")

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    train_loader = DataLoader(LungDataset2p5D(train_files, file_to_labels),
                               batch_size=32, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(LungDataset2p5D(test_files, file_to_labels),
                              batch_size=32, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_labels)
    model = model.to(device)

    # -------- CLASS-IMBALANCE HANDLING --------
    # Several pathologies are rare in the training set (e.g. Bronchiectasis, Peribronchial
    # thickening). Plain BCEWithLogitsLoss lets the model get away with (near) always
    # predicting "negative" on those and still score well on accuracy. pos_weight upweights
    # the loss on positive examples per-label, proportional to how rare they are, so the
    # model is actually pushed to learn the minority class instead of ignoring it.
    train_labels_arr = np.array([file_to_labels[os.path.basename(fp).replace(".nii.gz", "")]
                                  for fp in train_files])
    num_pos = train_labels_arr.sum(axis=0)
    num_neg = len(train_labels_arr) - num_pos
    pos_weight = torch.tensor(num_neg / np.clip(num_pos, 1, None), dtype=torch.float32).to(device)
    print("Per-label pos_weight (train set):")
    for name, w, p in zip(label_cols, pos_weight.tolist(), num_pos.tolist()):
        print(f"  {name}: pos_weight={w:.2f} (n_pos={int(p)}/{len(train_labels_arr)})")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    log_file = open("training_log.txt", "w")
    header = (f"{'Epoch':>5s} {'TrainLoss':>10s} {'Val_Acc':>8s} {'Val_AUC':>8s} "
              f"{'Val_Prec':>9s} {'Val_Recall':>10s} {'Val_F1':>8s}")
    print(header)
    log_file.write(header + "\n")

    print("Starting training...")
    num_epochs = 10
    best_macro_auc = -1.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # -------- full validation each epoch (same metrics as evaluate_2p5d.py) --------
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                probs = torch.sigmoid(model(x)).cpu().numpy()
                val_probs.append(probs)
                val_labels.append(y.numpy())
        val_probs = np.concatenate(val_probs, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)

        _, macro = compute_multilabel_metrics(val_labels, val_probs, label_cols)
        macro_auc = macro["auc"]

        # -------- checkpoint on best validation macro-AUC, not just the final epoch --------
        improved = macro_auc > best_macro_auc
        if improved:
            best_macro_auc = macro_auc
            torch.save(model.state_dict(), "model.pth")

        line = (f"{epoch:5d} {train_loss:10.4f} {macro['accuracy']:8.4f} {macro_auc:8.4f} "
                f"{macro['precision']:9.4f} {macro['recall']:10.4f} {macro['f1']:8.4f}"
                f"{'  <- new best AUC, saved model.pth' if improved else ''}")
        print(line)
        log_file.write(line + "\n")
        log_file.flush()

    log_file.close()
    print(f"Training complete. Best Val Macro AUC: {best_macro_auc:.4f} (checkpoint saved to model.pth)")


if __name__ == "__main__":
    main()
