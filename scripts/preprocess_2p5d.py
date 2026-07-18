import os
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from concurrent.futures import ProcessPoolExecutor, as_completed

from data_common import load_labels, get_split, CACHE_DIR

os.makedirs(CACHE_DIR, exist_ok=True)


def process_one(file_path):
    filename = os.path.basename(file_path).replace(".nii.gz", "")
    out_path = os.path.join(CACHE_DIR, filename + ".npy")
    if os.path.exists(out_path):
        return filename, "skipped (already cached)"
    try:
        data = nib.load(file_path).get_fdata()
        z = data.shape[2] // 2
        z_minus = max(z - 1, 0)
        z_plus = min(z + 1, data.shape[2] - 1)
        slice_stack = np.stack([data[:, :, z_minus], data[:, :, z], data[:, :, z_plus]], axis=0)
        slice_stack = (slice_stack - np.mean(slice_stack)) / (np.std(slice_stack) + 1e-8)
        t = torch.tensor(slice_stack).float()
        t = F.interpolate(t.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        np.save(out_path, t.numpy().astype(np.float32))
        return filename, "ok"
    except Exception as e:
        return filename, f"ERROR: {e}"


if __name__ == "__main__":
    file_to_labels, label_cols = load_labels()
    print(f"Loaded labels for {len(file_to_labels)} volumes, {len(label_cols)} pathology columns:")
    print(label_cols)

    train_files, test_files = get_split(file_to_labels)
    all_files = train_files + test_files
    print(f"Preprocessing {len(all_files)} files ({len(train_files)} train, {len(test_files)} test)")

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    print(f"Using {num_workers} worker processes")

    done = 0
    errors = []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(process_one, fp): fp for fp in all_files}
        for fut in as_completed(futures):
            filename, status = fut.result()
            done += 1
            if status.startswith("ERROR"):
                errors.append((filename, status))
            if done % 100 == 0 or done == len(all_files):
                print(f"[{done}/{len(all_files)}] {filename}: {status}")

    print(f"Done. {len(errors)} errors out of {len(all_files)}.")
    for filename, status in errors:
        print(f"  {filename}: {status}")
