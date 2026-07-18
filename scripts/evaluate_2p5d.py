import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# -------- REPRODUCIBILITY --------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------- LABELS --------
labels_df = pd.read_csv("../../hugenv/example_download_script/train_labels.csv")
file_to_label = dict(zip(labels_df['VolumeName'], labels_df['Lung nodule']))

# -------- DATASET --------
class LungDataset2p5D(Dataset):
    def __init__(self, files):
        self.files = files
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        file_path = self.files[idx]
        filename = os.path.basename(file_path).replace(".nii.gz", "")
        data = nib.load(file_path).get_fdata()
        z = data.shape[2] // 2
        z_minus = max(z - 1, 0)
        z_plus = min(z + 1, data.shape[2] - 1)
        slice_stack = np.stack([data[:, :, z_minus], data[:, :, z], data[:, :, z_plus]], axis=0)
        slice_stack = (slice_stack - np.mean(slice_stack)) / (np.std(slice_stack) + 1e-8)
        slice_stack = torch.tensor(slice_stack).float()
        slice_stack = F.interpolate(slice_stack.unsqueeze(0), size=(224, 224),
                                    mode='bilinear', align_corners=False).squeeze(0)
        label = torch.tensor(file_to_label.get(filename, 0)).float()
        return slice_stack, label

# -------- GATHER + SPLIT (deterministic) --------
root_dir = "/scratch/25205761/hugenv/example_download_script/data_volumes/dataset/train"
all_files = []
for root, _, files in os.walk(root_dir):
    for f in files:
        if f.endswith(".nii.gz"):
            all_files.append(os.path.join(root, f))
all_files = sorted(all_files)
rng = random.Random(SEED)
rng.shuffle(all_files)
all_files = all_files[:2000]  # seeded subset for tractable runtime
split = int(0.8 * len(all_files))
train_files = all_files[:split]
test_files = all_files[split:]
print(f"Total: {len(all_files)} | Train: {len(train_files)} | Test: {len(test_files)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(LungDataset2p5D(train_files), batch_size=16, shuffle=True, num_workers=1)
test_loader = DataLoader(LungDataset2p5D(test_files), batch_size=16, shuffle=False, num_workers=1)

# -------- MODEL --------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

# -------- TRAIN --------
print("Training...")
for epoch in range(5):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = model(x).squeeze(-1)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# -------- EVALUATE --------
print("Evaluating on held-out test set...")
model.eval()
all_probs, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        probs = torch.sigmoid(model(x).squeeze(-1)).cpu().numpy()
        all_probs.extend(np.atleast_1d(probs).tolist())
        all_labels.extend(np.atleast_1d(y.numpy()).tolist())

all_labels = np.array(all_labels)
all_preds = (np.array(all_probs) >= 0.5).astype(int)
acc = accuracy_score(all_labels, all_preds)
try:
    auc = roc_auc_score(all_labels, all_probs)
except ValueError:
    auc = float('nan')
tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0,1]).ravel()
sens = tp / (tp + fn) if (tp + fn) else float('nan')
spec = tn / (tn + fp) if (tn + fp) else float('nan')

report = (f"Test set: {len(all_labels)} cases\n"
          f"Accuracy: {acc:.4f}\nAUC: {auc:.4f}\n"
          f"Sensitivity: {sens:.4f}\nSpecificity: {spec:.4f}\n"
          f"Confusion: TN={tn} FP={fp} FN={fn} TP={tp}\nSeed: {SEED}\n")
print(report)
with open("evaluation_log.txt", "w") as f:
    f.write(report)
print("Saved to evaluation_log.txt")
