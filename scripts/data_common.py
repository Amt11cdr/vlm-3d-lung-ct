"""
Shared data-handling logic for the VLM#D lung CT pipeline.

Used identically by preprocess_2p5d.py, train_2p5d.py, and evaluate_2p5d.py so the
patient-level train/test split and multi-label loading can never drift out of sync
between scripts again.

File naming convention observed in the dataset: train_<patient>_<session>_<recon>.nii.gz
e.g. train_1_a_1.nii.gz and train_1_a_2.nii.gz are two reconstructions of the SAME scan
for patient "1", session "a" -- they carry identical labels and must never be split
across train/test (that would be leakage). We split at the patient level.
"""
import os
import random
import pandas as pd

SEED = 42

LABELS_CSV = "../../hugenv/example_download_script/train_labels.csv"
ROOT_DIR = "/scratch/25205761/hugenv/example_download_script/data_volumes/dataset/train"
CACHE_DIR = "/scratch/25205761/vlm3d_project/cache_2p5d"
SPLIT_DIR = "/scratch/25205761/vlm3d_project/splits"

MAX_PATIENTS = 1000   # cap for tractable runtime; raise this once the pipeline is validated
TRAIN_FRAC = 0.8


def load_labels():
    """Returns (file_to_labels: dict[str -> list[float]], label_cols: list[str])."""
    df = pd.read_csv(LABELS_CSV)
    label_cols = [c for c in df.columns if c != "VolumeName"]
    file_to_labels = {}
    for _, row in df.iterrows():
        fname = str(row["VolumeName"]).replace(".nii.gz", "")
        file_to_labels[fname] = [float(row[c]) for c in label_cols]
    return file_to_labels, label_cols


def patient_id_from_filename(filename):
    """train_1_a_1 -> '1'. Falls back to the full filename if the pattern doesn't match."""
    parts = filename.split("_")
    if len(parts) >= 2 and parts[0] == "train":
        return parts[1]
    return filename


def gather_labeled_files(file_to_labels):
    files = []
    for root, _, fs in os.walk(ROOT_DIR):
        for f in fs:
            if f.endswith(".nii.gz"):
                fname = f.replace(".nii.gz", "")
                if fname in file_to_labels:
                    files.append(os.path.join(root, f))
    return sorted(files)


def build_patient_split(file_to_labels, max_patients=MAX_PATIENTS, train_frac=TRAIN_FRAC):
    """Groups files by patient, shuffles patients (seeded), splits by patient so all
    reconstructions/sessions of a patient land in the same side of the split."""
    all_files = gather_labeled_files(file_to_labels)

    patient_to_files = {}
    for fp in all_files:
        fname = os.path.basename(fp).replace(".nii.gz", "")
        pid = patient_id_from_filename(fname)
        patient_to_files.setdefault(pid, []).append(fp)

    patient_ids = sorted(patient_to_files.keys())
    rng = random.Random(SEED)
    rng.shuffle(patient_ids)
    patient_ids = patient_ids[:max_patients]

    split_idx = int(train_frac * len(patient_ids))
    train_patients = patient_ids[:split_idx]
    test_patients = patient_ids[split_idx:]

    train_files = sorted(fp for pid in train_patients for fp in patient_to_files[pid])
    test_files = sorted(fp for pid in test_patients for fp in patient_to_files[pid])

    print(f"Patients used: {len(patient_ids)} (train: {len(train_patients)}, test: {len(test_patients)})")
    print(f"Files: train={len(train_files)}, test={len(test_files)}")

    return train_files, test_files


def save_split(train_files, test_files):
    os.makedirs(SPLIT_DIR, exist_ok=True)
    with open(os.path.join(SPLIT_DIR, "train_files.txt"), "w") as f:
        f.write("\n".join(train_files))
    with open(os.path.join(SPLIT_DIR, "test_files.txt"), "w") as f:
        f.write("\n".join(test_files))


def split_exists():
    return (os.path.exists(os.path.join(SPLIT_DIR, "train_files.txt"))
            and os.path.exists(os.path.join(SPLIT_DIR, "test_files.txt")))


def load_split():
    with open(os.path.join(SPLIT_DIR, "train_files.txt")) as f:
        train_files = [l.strip() for l in f if l.strip()]
    with open(os.path.join(SPLIT_DIR, "test_files.txt")) as f:
        test_files = [l.strip() for l in f if l.strip()]
    return train_files, test_files


def get_split(file_to_labels):
    """Loads the frozen split from disk if it exists, otherwise builds and saves one.
    Call this from preprocess_2p5d.py first so train/eval always read the same frozen split."""
    if split_exists():
        return load_split()
    train_files, test_files = build_patient_split(file_to_labels)
    save_split(train_files, test_files)
    return train_files, test_files
