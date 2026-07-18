import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

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

    train_files, test_files = get_split(file_to_labels)  # loads the SAME frozen split used in training
    print(f"Evaluating on held-out test set: {len(test_files)} files "
          f"(train set, unused here, has {len(train_files)} files)")

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    test_loader = DataLoader(LungDataset2p5D(test_files, file_to_labels),
                              batch_size=32, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_labels)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print(f"Loaded trained weights from {model_path}")

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            probs = torch.sigmoid(model(x)).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = (all_probs >= 0.5).astype(int)

    lines = [f"Test set: {len(all_labels)} cases (patient-level split, seed={SEED})", ""]
    per_label_aucs = []
    header = f"{'Pathology':35s} {'N+':>6s} {'Acc':>7s} {'AUC':>7s} {'Sens':>7s} {'Spec':>7s}"
    lines.append(header)
    lines.append("-" * len(header))

    for i, name in enumerate(label_cols):
        y_true = all_labels[:, i]
        y_pred = all_preds[:, i]
        y_prob = all_probs[:, i]
        n_pos = int(y_true.sum())
        acc = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_prob)
            per_label_aucs.append(auc)
        except ValueError:
            auc = float('nan')  # only one class present in test set for this label
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else float('nan')
        spec = tn / (tn + fp) if (tn + fp) else float('nan')
        lines.append(f"{name:35s} {n_pos:6d} {acc:7.3f} {auc:7.3f} {sens:7.3f} {spec:7.3f}")

    macro_auc = float(np.mean(per_label_aucs)) if per_label_aucs else float('nan')
    lines.append("")
    lines.append(f"Macro-average AUC (over {len(per_label_aucs)}/{num_labels} labels "
                 f"with both classes present in test set): {macro_auc:.4f}")

    report = "\n".join(lines)
    print(report)
    with open("evaluation_log.txt", "w") as f:
        f.write(report + "\n")
    print("\nSaved to evaluation_log.txt")


if __name__ == "__main__":
    main()
