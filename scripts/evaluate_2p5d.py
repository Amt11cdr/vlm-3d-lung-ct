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

    # Metrics matching the VLM3D Challenge Task 1 (Multi-Abnormality Classification)
    # official evaluation set: AUROC, F1, Precision, Recall, Accuracy. Specificity is
    # kept too since it's clinically important for imbalanced findings even though it's
    # not one of the official ranking metrics. (CRG is the official metric for the
    # separate report-generation task, not classification, so it's not computed here.)
    per_label, macro = compute_multilabel_metrics(all_labels, all_probs, label_cols)

    lines = [f"Test set: {len(all_labels)} cases (patient-level split, seed={SEED})", ""]
    header = f"{'Pathology':35s} {'N+':>6s} {'Acc':>7s} {'AUC':>7s} {'Prec':>7s} {'Recall':>7s} {'Spec':>7s} {'F1':>7s}"
    lines.append(header)
    lines.append("-" * len(header))
    for i, name in enumerate(label_cols):
        lines.append(f"{name:35s} {per_label['n_pos'][i]:6d} {per_label['acc'][i]:7.3f} "
                     f"{per_label['auc'][i]:7.3f} {per_label['precision'][i]:7.3f} "
                     f"{per_label['recall'][i]:7.3f} {per_label['specificity'][i]:7.3f} "
                     f"{per_label['f1'][i]:7.3f}")

    lines.append("")
    lines.append(f"Macro-average AUC (over {macro['auc_n_labels']}/{num_labels} labels "
                 f"with both classes present in test set): {macro['auc']:.4f}")
    lines.append(f"Macro-average F1: {macro['f1']:.4f}")
    lines.append(f"Macro-average Precision: {macro['precision']:.4f}")
    lines.append(f"Macro-average Recall (Sensitivity): {macro['recall']:.4f}")
    lines.append(f"Macro-average Accuracy: {macro['accuracy']:.4f}")

    report = "\n".join(lines)
    print(report)
    with open("evaluation_log.txt", "w") as f:
        f.write(report + "\n")
    print("\nSaved to evaluation_log.txt")


if __name__ == "__main__":
    main()
