# VLM3D — Multi-Pathology Lung CT Classification

## Motivation

Chest CT scans are read and reported manually by radiologists, one pathology finding
at a time. A model that can flag the presence of multiple thoracic pathologies
directly from a CT volume could help triage studies, catch incidental findings, and
speed up reporting. This project builds and evaluates a deep learning pipeline that
takes a 3D chest CT volume and predicts which of 18 common pathological findings are
present.

## Problem

Each CT study in the dataset is a 3D volume (`.nii.gz`), and each volume is labeled
with 18 binary pathology flags (multi-label, not mutually exclusive):

Medical material, Arterial wall calcification, Cardiomegaly, Pericardial effusion,
Coronary artery wall calcification, Hiatal hernia, Lymphadenopathy, Emphysema,
Atelectasis, Lung nodule, Lung opacity, Pulmonary fibrotic sequela, Pleural effusion,
Mosaic attenuation pattern, Peribronchial thickening, Consolidation, Bronchiectasis,
Interlobular septal thickening.

The task is: given a CT volume, predict the probability of each of these 18 findings.

Two things make this harder than a standard image classification problem:

- **Data scale.** The full dataset is ~2.8TB across ~12,000 volumes, each roughly
  100–250MB compressed. Reading a single slice still requires fully decompressing
  the volume (`.nii.gz` has no partial-read shortcut), so naive data loading is a
  major bottleneck.
- **Patient-level leakage risk.** A single scan session can produce multiple
  reconstructions of the same underlying CT (e.g. `train_1_a_1.nii.gz` and
  `train_1_a_2.nii.gz` — same patient, same session, identical labels). Splitting
  data by file instead of by patient risks putting near-duplicate scans in both the
  train and test sets, silently inflating reported performance.

## Solution / Approach

**Representation.** Rather than processing full 3D volumes, each CT is reduced to a
2.5D representation: the middle axial slice plus its two neighboring slices, stacked
as a 3-channel image, normalized, and resized to 224×224. This lets a standard 2D
CNN backbone (ResNet18, ImageNet-pretrained) be used as the classifier, with its
final layer replaced by an 18-unit output (one logit per pathology) trained with
`BCEWithLogitsLoss` for independent multi-label prediction.

**Patient-level splitting.** All files are grouped by patient ID (parsed from the
`train_<patient>_<session>_<recon>` filename convention) before splitting, so every
reconstruction of a given patient's scan stays entirely within train or entirely
within test. The split is computed once, frozen to disk, and reused identically by
preprocessing, training, and evaluation — see `scripts/data_common.py`, the single
source of truth for label loading and the split, so the three stages can't drift out
of sync.

**Preprocessing / caching.** Because decompressing full volumes on every epoch is
slow, `scripts/preprocess_2p5d.py` runs once (in parallel across CPU workers) to
extract and cache the normalized 3-slice tensor for every volume in the selected
subset. Training and evaluation then read directly from this cache, cutting I/O from
"decompress a multi-hundred-MB volume" to "load a small `.npy` file."

**Scale-down for iteration.** The current pipeline caps the working set to a
configurable number of patients (`MAX_PATIENTS` in `data_common.py`, currently 1000)
so the pipeline can be validated end-to-end before committing compute to the full
~12,000-volume dataset.

## Pipeline

1. `scripts/preprocess_2p5d.py` — builds the patient-level train/test split (frozen
   to `splits/`), and caches the 2.5D tensor for each volume to `cache_2p5d/`.
2. `scripts/train_2p5d.py` — trains a ResNet18 (18-way multi-label output) on the
   cached train set, reporting train loss and validation macro-AUC each epoch.
   Saves `model.pth`.
3. `scripts/evaluate_2p5d.py` — loads `model.pth` and reports per-pathology accuracy,
   AUC, sensitivity, and specificity, plus a macro-averaged AUC across all 18
   labels, on the held-out patient-level test set.

Each stage has a matching SLURM submission script (`submit_preprocess.sh`,
`submit_train.sh`, `submit_eval.sh`) for running on the UCD Sonic HPC cluster.

## Status / Results

Pipeline validated end-to-end on a 1000-patient subset; full results pending
completion of the current preprocessing + training run. This section will be
updated with per-pathology performance once available.

## Model weights

Not tracked in git (`model.pth`, ~43MB). Kept on Sonic at
`/scratch/25205761/vlm3d_project/scripts/model.pth`. Consider Git LFS or a GitHub
Release if versioning the weights becomes necessary.

## Cluster

Developed and trained on UCD's Sonic HPC cluster (`sonic.ucd.ie`).
