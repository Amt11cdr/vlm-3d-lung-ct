import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score

from data_common import load_labels, get_split, CACHE_DIR, SEED

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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    log_file = open("training_log.txt", "w")
    print("Starting training...")
    num_epochs = 10
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

        # -------- quick validation each epoch --------
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

        aucs = []
        for i in range(num_labels):
            if len(np.unique(val_labels[:, i])) > 1:
                aucs.append(roc_auc_score(val_labels[:, i], val_probs[:, i]))
        macro_auc = float(np.mean(aucs)) if aucs else float('nan')

        line = f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Macro AUC: {macro_auc:.4f} (over {len(aucs)}/{num_labels} labels with both classes present)"
        print(line)
        log_file.write(line + "\n")
        log_file.flush()

    torch.save(model.state_dict(), "model.pth")
    log_file.close()
    print("Training complete. Saved model.pth")


if __name__ == "__main__":
    main()
