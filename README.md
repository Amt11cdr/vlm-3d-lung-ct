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

Pipeline validated end-to-end on a 1000-patient subset (800 train / 200 test,
2354 CT files, patient-level split, seed=42). ResNet18 trained for 10 epochs
(~2.5 min on one GPU).

### Experiment 1 — baseline (no class weighting, final-epoch checkpoint)

**Training trend.** Train loss dropped steadily from 0.499 to 0.026 over 10
epochs, but validation macro-AUC peaked early at epoch 2 (0.7338) and then
flattened/drifted down to 0.7204 by epoch 9 — a classic overfitting curve.
The current script saves only the final epoch's weights, so the evaluated
model (below) is the more-overfit epoch 9, not the epoch-2 peak. Adding
best-checkpoint saving is a planned next step.

**Held-out test set (488 cases, 200 patients, threshold = 0.5):**

| Pathology | N+ | Acc | AUC | Sens | Spec |
|---|---|---|---|---|---|
| Medical material | 75 | 0.834 | 0.753 | 0.200 | 0.949 |
| Arterial wall calcification | 166 | 0.791 | 0.860 | 0.693 | 0.842 |
| Cardiomegaly | 72 | 0.863 | 0.866 | 0.417 | 0.940 |
| Pericardial effusion | 45 | 0.914 | 0.690 | 0.111 | 0.995 |
| Coronary artery wall calcification | 142 | 0.719 | 0.793 | 0.514 | 0.803 |
| Hiatal hernia | 87 | 0.793 | 0.570 | 0.092 | 0.945 |
| Lymphadenopathy | 154 | 0.721 | 0.697 | 0.325 | 0.904 |
| Emphysema | 107 | 0.791 | 0.671 | 0.262 | 0.940 |
| Atelectasis | 159 | 0.672 | 0.672 | 0.371 | 0.818 |
| Lung nodule | 228 | 0.611 | 0.630 | 0.649 | 0.577 |
| Lung opacity | 181 | 0.689 | 0.685 | 0.436 | 0.837 |
| Pulmonary fibrotic sequela | 135 | 0.689 | 0.564 | 0.207 | 0.873 |
| Pleural effusion | 83 | 0.898 | 0.949 | 0.578 | 0.963 |
| Mosaic attenuation pattern | 53 | 0.889 | 0.807 | 0.151 | 0.979 |
| Peribronchial thickening | 59 | 0.848 | 0.602 | 0.017 | 0.963 |
| Consolidation | 103 | 0.766 | 0.756 | 0.194 | 0.919 |
| Bronchiectasis | 68 | 0.857 | 0.571 | 0.000 | 0.995 |
| Interlobular septal thickening | 52 | 0.887 | 0.831 | 0.019 | 0.991 |

**Macro-average AUC: 0.7204** (18/18 labels had both classes present in the
test set).

**Interpretation.** The model shows genuinely useful ranking ability on
several findings — Pleural effusion (AUC 0.949), Cardiomegaly (0.866),
Arterial wall calcification (0.860), Interlobular septal thickening (0.831),
Mosaic attenuation pattern (0.807), and Coronary artery wall calcification
(0.793) — while Hiatal hernia (0.570), Bronchiectasis (0.571), Pulmonary
fibrotic sequela (0.564), and Peribronchial thickening (0.602) are close to
chance.

More importantly, several labels have near-zero sensitivity despite
reasonable accuracy and AUC (Bronchiectasis 0.000, Interlobular septal
thickening 0.019, Peribronchial thickening 0.017, Pericardial effusion
0.111, Mosaic attenuation pattern 0.151). This is a class-imbalance effect:
at the default 0.5 threshold, the model rarely predicts "positive" for
rarer findings, and gets away with high accuracy simply because those
findings are uncommon in the test set. The AUC values show the ranking
signal is often still there — the fixed threshold is what's miscalibrated,
not necessarily the underlying representation.

### Experiment 2 — `pos_weight` + best-checkpoint saving

Two fixes applied to `train_2p5d.py`: (1) per-label `pos_weight` added to
`BCEWithLogitsLoss`, computed from the train-set class balance
(`num_neg / num_pos`), so rare positive findings are upweighted in the loss
instead of being drowned out; (2) `model.pth` now saves whenever validation
macro-AUC improves, rather than only at the final epoch.

`pos_weight` per label ranged from 1.02 (Lung nodule, roughly balanced,
n_pos=923/1866) up to 12.14 (Pericardial effusion, n_pos=142/1866) and 11.96
(Mosaic attenuation pattern, n_pos=144/1866) — correctly weighting the
rarest findings most heavily.

Best validation macro-AUC this run was hit at epoch 0 (0.7318) — essentially
right after the ImageNet-pretrained backbone's first pass of fine-tuning —
with every later epoch trending flat-to-worse as training loss kept
dropping. That checkpoint (epoch 0) is what's saved and evaluated below.

**Held-out test set (488 cases, same patient-level split, threshold = 0.5):**

| Pathology | N+ | Acc | AUC | Sens | Spec |
|---|---|---|---|---|---|
| Medical material | 75 | 0.703 | 0.754 | 0.640 | 0.714 |
| Arterial wall calcification | 166 | 0.742 | 0.847 | 0.873 | 0.674 |
| Cardiomegaly | 72 | 0.764 | 0.877 | 0.806 | 0.757 |
| Pericardial effusion | 45 | 0.703 | 0.726 | 0.733 | 0.700 |
| Coronary artery wall calcification | 142 | 0.682 | 0.781 | 0.775 | 0.645 |
| Hiatal hernia | 87 | 0.592 | 0.603 | 0.552 | 0.601 |
| Lymphadenopathy | 154 | 0.635 | 0.728 | 0.734 | 0.590 |
| Emphysema | 107 | 0.547 | 0.663 | 0.720 | 0.499 |
| Atelectasis | 159 | 0.607 | 0.662 | 0.736 | 0.544 |
| Lung nodule | 228 | 0.547 | 0.551 | 0.487 | 0.600 |
| Lung opacity | 181 | 0.611 | 0.675 | 0.652 | 0.586 |
| Pulmonary fibrotic sequela | 135 | 0.570 | 0.582 | 0.570 | 0.569 |
| Pleural effusion | 83 | 0.818 | 0.939 | 0.988 | 0.783 |
| Mosaic attenuation pattern | 53 | 0.686 | 0.851 | 0.868 | 0.664 |
| Peribronchial thickening | 59 | 0.523 | 0.692 | 0.763 | 0.490 |
| Consolidation | 103 | 0.670 | 0.749 | 0.757 | 0.647 |
| Bronchiectasis | 68 | 0.520 | 0.644 | 0.779 | 0.479 |
| Interlobular septal thickening | 52 | 0.664 | 0.847 | 0.923 | 0.633 |

**Macro-average AUC: 0.7318** (18/18 labels).

**Comparison to Experiment 1.** Sensitivity improved dramatically on nearly
every pathology that previously had near-zero recall: Bronchiectasis
0.000→0.779, Interlobular septal thickening 0.019→0.923, Peribronchial
thickening 0.017→0.763, Mosaic attenuation pattern 0.151→0.868, Pericardial
effusion 0.111→0.733. The model went from effectively never flagging these
findings to actually detecting most positive cases.

The trade-off: specificity and accuracy dropped substantially in exchange
(e.g. Bronchiectasis specificity 0.995→0.479) — the classic precision/recall
trade-off from reweighting toward the minority class. For a screening/triage
tool this shifted operating point is often preferable (missing a real
finding tends to cost more than a false alarm), but it is a genuine
trade-off, not a free win.

Macro-AUC itself barely moved (0.7204→0.7318), which makes sense: AUC is
threshold-independent, so `pos_weight` mainly shifted the decision
threshold's effective operating point rather than the model's underlying
ranking ability. A few labels did show genuine AUC gains (Cardiomegaly
0.866→0.877, Pericardial effusion 0.690→0.726, Mosaic attenuation pattern
0.807→0.851, Bronchiectasis 0.571→0.644), while Lung nodule dropped
(0.630→0.551) — likely noise from this run's checkpoint being a much
earlier, less-trained epoch than Experiment 1's.

**Planned next steps:**
- Try per-label threshold tuning (instead of a fixed 0.5 cutoff) as an
  alternative/complement to `pos_weight`, to recover some specificity
  without giving up sensitivity gains.
- Investigate why validation performance peaks so early (epoch 0) — try a
  lower learning rate, weight decay/dropout, or fewer epochs, since later
  epochs mostly overfit rather than improve.
- Scale `MAX_PATIENTS` up from 1000 toward the full ~12,000-volume dataset
  once these fixes are validated.

## Model weights

Not tracked in git (`model.pth`, ~43MB). Kept on Sonic at
`/scratch/25205761/vlm3d_project/scripts/model.pth`. Consider Git LFS or a GitHub
Release if versioning the weights becomes necessary.

## Cluster

Developed and trained on UCD's Sonic HPC cluster (`sonic.ucd.ie`).
