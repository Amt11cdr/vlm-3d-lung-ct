import os
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch.nn as nn

# -------- LOAD LABELS --------
labels_df = pd.read_csv("../../hugenv/example_download_script/train_labels.csv")

# Map VolumeName → Lung nodule (0/1)
file_to_label = dict(zip(labels_df['VolumeName'], labels_df['Lung nodule']))

# -------- DATASET --------
class LungDataset2p5D(Dataset):
    def __init__(self, root_dir):
        self.files = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.endswith(".nii.gz"):
                    self.files.append(os.path.join(root, f))

        self.files = self.files[:300]  # limit for speed

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        filename = os.path.basename(file_path).replace(".nii.gz", "")

        img = nib.load(file_path)
        data = img.get_fdata()

        # middle slice
        z = data.shape[2] // 2
        z_minus = max(z - 1, 0)
        z_plus = min(z + 1, data.shape[2] - 1)

        # 2.5D stack
        slice_stack = np.stack([
            data[:, :, z_minus],
            data[:, :, z],
            data[:, :, z_plus]
        ], axis=0)

        # normalize
        slice_stack = (slice_stack - np.mean(slice_stack)) / (np.std(slice_stack) + 1e-8)

        slice_stack = torch.tensor(slice_stack).float()

        # resize to 224x224
        slice_stack = F.interpolate(
            slice_stack.unsqueeze(0),
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # label
        label = file_to_label.get(filename, 0)
        label = torch.tensor(label).float()

        return slice_stack, label

# -------- MODEL --------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# -------- TRAIN --------
dataset = LungDataset2p5D("/scratch/25205761/hugenv/example_download_script/data_volumes/dataset/train")
loader = DataLoader(dataset, batch_size=2, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

log_file = open("training_log.txt", "w")

print("Starting training...")

for epoch in range(5):
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        out = model(x).squeeze()
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")
    log_file.write(f"Epoch {epoch}, Loss: {loss.item()}\n")
    log_file.flush()

torch.save(model.state_dict(), "model.pth")

print("Training complete.")
