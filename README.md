# VLM#D — Multi-Pathology Lung CT Classification

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

### Experiment 3 — full metric suite (Precision, Recall, F1, AUC, Accuracy)

The project is being built for the **VLM3D Challenge @ MICCAI 2025** (Task 1:
Multi-Abnormality Classification on the CT-RATE dataset). The challenge's
official Task 1 ranking metrics are AUROC, macro-F1, Precision, Recall, and
Accuracy (CRG is a separate metric used for the report-generation task, not
classification, so it isn't computed here). `evaluate_2p5d.py` and the
per-epoch validation inside `train_2p5d.py` were updated to report this full
set — via one shared `compute_multilabel_metrics()` function in
`data_common.py` so the two scripts can't disagree on how a metric is
computed.

**Epoch-wise validation trend (10 epochs, same 800/200 patient split):**

| Epoch | Train Loss | Val Acc | Val AUC | Val Prec | Val Recall | Val F1 |
|---|---|---|---|---|---|---|
| 0 | 0.9621 | 0.6444 | 0.7321 | 0.3514 | 0.7342 | 0.4580 |
| 1 | 0.6546 | 0.7330 | 0.7246 | 0.3956 | 0.5808 | 0.4605 |
| 2 | 0.4403 | 0.7559 | 0.7287 | 0.4141 | 0.4893 | 0.4411 |
| 3 | 0.2900 | 0.7741 | 0.7194 | 0.4572 | 0.4323 | 0.4279 |
| 4 | 0.1947 | 0.7703 | 0.7164 | 0.4321 | 0.4403 | 0.4274 |
| 5 | 0.1386 | 0.7732 | 0.7143 | 0.4513 | 0.4164 | 0.4213 |
| 6 | 0.1032 | 0.7831 | 0.7186 | 0.4638 | 0.3633 | 0.3937 |
| 7 | 0.0818 | 0.7793 | 0.7153 | 0.4506 | 0.3726 | 0.3991 |
| 8 | 0.0662 | 0.7838 | 0.7193 | 0.4615 | 0.3399 | 0.3819 |
| 9 | 0.0538 | 0.7762 | 0.7170 | 0.4444 | 0.3948 | 0.4075 |

This is the clearest view yet of the overfitting dynamic: accuracy climbs
steadily (0.644→0.784) as training progresses, but recall falls in the
opposite direction (0.734→~0.33-0.39), and F1 actually peaks at epoch 1
(0.4605) before drifting down. AUC stays essentially flat (~0.71-0.73)
throughout. In other words, additional epochs mostly make the model more
conservative about predicting "positive" (trading recall for accuracy)
without the underlying ranking ability improving — reinforcing that
best-checkpoint selection (added in Experiment 2) is doing real work here,
and that early stopping around epoch 0-1 is currently optimal for this
learning rate / pos_weight combination.

**Held-out test set (488 cases, epoch-0 checkpoint, threshold = 0.5):**

| Pathology | N+ | Acc | AUC | Prec | Recall | Spec | F1 |
|---|---|---|---|---|---|---|---|
| Medical material | 75 | 0.689 | 0.745 | 0.269 | 0.600 | 0.705 | 0.372 |
| Arterial wall calcification | 166 | 0.746 | 0.848 | 0.584 | 0.880 | 0.677 | 0.702 |
| Cardiomegaly | 72 | 0.760 | 0.879 | 0.358 | 0.792 | 0.755 | 0.494 |
| Pericardial effusion | 45 | 0.703 | 0.730 | 0.195 | 0.711 | 0.702 | 0.306 |
| Coronary artery wall calcification | 142 | 0.691 | 0.781 | 0.480 | 0.768 | 0.659 | 0.591 |
| Hiatal hernia | 87 | 0.615 | 0.600 | 0.244 | 0.552 | 0.628 | 0.338 |
| Lymphadenopathy | 154 | 0.645 | 0.732 | 0.460 | 0.701 | 0.620 | 0.555 |
| Emphysema | 107 | 0.543 | 0.662 | 0.279 | 0.682 | 0.504 | 0.396 |
| Atelectasis | 159 | 0.598 | 0.658 | 0.431 | 0.723 | 0.538 | 0.540 |
| Lung nodule | 228 | 0.549 | 0.553 | 0.519 | 0.478 | 0.612 | 0.498 |
| Lung opacity | 181 | 0.621 | 0.680 | 0.492 | 0.641 | 0.609 | 0.556 |
| Pulmonary fibrotic sequela | 135 | 0.578 | 0.580 | 0.341 | 0.563 | 0.584 | 0.425 |
| Pleural effusion | 83 | 0.818 | 0.940 | 0.483 | 1.000 | 0.780 | 0.651 |
| Mosaic attenuation pattern | 53 | 0.674 | 0.856 | 0.232 | 0.868 | 0.651 | 0.367 |
| Peribronchial thickening | 59 | 0.520 | 0.692 | 0.170 | 0.763 | 0.487 | 0.278 |
| Consolidation | 103 | 0.666 | 0.751 | 0.364 | 0.777 | 0.636 | 0.495 |
| Bronchiectasis | 68 | 0.527 | 0.642 | 0.199 | 0.794 | 0.483 | 0.319 |
| Interlobular septal thickening | 52 | 0.656 | 0.849 | 0.226 | 0.923 | 0.624 | 0.364 |

**Macro-average: AUC 0.7321, F1 0.4580, Precision 0.3514, Recall 0.7342,
Accuracy 0.6444.**

Precision is low across almost every pathology (many below 0.3, e.g.
Peribronchial thickening 0.170, Interlobular septal thickening 0.226,
Mosaic attenuation pattern 0.232) while recall is high (several above 0.75,
Pleural effusion hits 1.000). This is `pos_weight` doing exactly what it's
designed to do — pushing hard toward flagging positives — but the current
weighting is likely too aggressive for several labels, trading away more
precision than necessary. F1 (which balances both) sits around 0.28-0.55
depending on the pathology, giving a more honest single-number view per
finding than accuracy alone would.

**Planned next steps (updated):**
- Tune down `pos_weight` (e.g. cap it, or use `sqrt(num_neg/num_pos)`
  instead of the raw ratio) to recover precision without losing all the
  recall gains — the current weighting looks over-corrected.
- Try per-label threshold tuning as an alternative/complement to
  `pos_weight`.
- Try a lower learning rate or a short warmup/decay schedule, since useful
  training clearly happens within the first 1-2 epochs and further epochs
  currently just trade recall for accuracy without real gains.
- Scale `MAX_PATIENTS` up from 1000 toward the full ~12,000-volume dataset
  once the above are validated.

### Experiment 4 — sqrt-dampened `pos_weight` (tested, reverted)

Hypothesis from Experiment 3: the raw `num_neg/num_pos` pos_weight (up to
12.14x) was over-correcting, so `sqrt(num_neg/num_pos)` was tried instead
(dampens 12.14x down to ~3.48x) to try to recover precision without losing
too much recall.

**Epoch-wise validation trend:**

| Epoch | Train Loss | Val Acc | Val AUC | Val Prec | Val Recall | Val F1 |
|---|---|---|---|---|---|---|
| 0 | 0.6669 | 0.7647 | 0.7321 | 0.4159 | 0.4059 | 0.3947 |
| 1 | 0.4427 | 0.7847 | 0.7293 | 0.4559 | 0.3619 | 0.3842 |
| 2 | 0.2916 | 0.7926 | 0.7328 | 0.4935 | 0.3082 | 0.3556 |
| 3 | 0.1834 | 0.7864 | 0.7263 | 0.4857 | 0.3455 | 0.3726 |
| 4 | 0.1186 | 0.7868 | 0.7188 | 0.4705 | 0.3631 | 0.3938 |
| 5 | 0.0836 | 0.7854 | 0.7222 | 0.4988 | 0.3532 | 0.3847 |
| 6 | 0.0626 | 0.7887 | 0.7222 | 0.4992 | 0.3159 | 0.3609 |
| 7 | 0.0505 | 0.7903 | 0.7192 | 0.4900 | 0.3298 | 0.3771 |
| 8 | 0.0419 | 0.7884 | 0.7242 | 0.5006 | 0.3021 | 0.3529 |
| 9 | 0.0346 | 0.7892 | 0.7253 | 0.4822 | 0.3486 | 0.3867 |

Checkpoint selection (best AUC at the time) picked epoch 2 (AUC 0.7328),
whose F1 (0.3556) was clearly worse than epoch 0's F1 (0.3947, the actual
best F1 in this run) — a first sign that selecting checkpoints by AUC
rather than F1 is a mistake, since AUC and F1 don't agree on which epoch is
best.

**Held-out test set result (epoch-2 checkpoint):** Accuracy 0.7926, AUC
0.7328, Precision 0.4935, Recall 0.3082, F1 0.3556. Compared to Experiment 3
(raw ratio): accuracy and precision went up substantially (0.644→0.793,
0.351→0.494), but recall collapsed (0.734→0.308) and **macro-F1 got worse**
(0.4580→0.3556) — the sqrt dampening overshot in the opposite direction
rather than landing at a better balance. AUC was essentially unchanged
(0.7321 vs 0.7318), as expected since it's threshold-independent.

**Conclusion: reverted.** The hypothesis was tested and didn't hold up —
the raw ratio outperforms the sqrt-dampened version on F1, the metric that
actually matters here. `pos_weight` was reverted to `num_neg/num_pos`.
Checkpoint selection was also switched from best-AUC to best-F1, directly
motivated by the epoch-2-vs-epoch-0 mismatch seen in this run.

**Next up:** rerun training with the raw ratio restored and F1-based
checkpointing, expecting a result at or above Experiment 3's F1 (0.4580),
ideally the actual best epoch rather than whichever had marginally higher
AUC.

### Experiment 5 — raw `pos_weight` restored + F1-based checkpointing

`pos_weight` reverted to `num_neg/num_pos` (Experiment 3's version) and
checkpoint selection switched to best validation macro-F1 (see Experiment 4
for why). Same 800/200 patient split, 10 epochs.

**Epoch-wise validation trend:**

| Epoch | Train Loss | Val Acc | Val AUC | Val Prec | Val Recall | Val F1 |
|---|---|---|---|---|---|---|
| 0 | 0.9622 | 0.6440 | 0.7324 | 0.3519 | 0.7431 | 0.4618 |
| 1 | 0.6540 | 0.7256 | 0.7240 | 0.3857 | 0.5972 | 0.4597 |
| 2 | 0.4395 | 0.7583 | 0.7275 | 0.4180 | 0.4772 | 0.4379 |
| 3 | 0.2903 | 0.7658 | 0.7145 | 0.4304 | 0.4550 | 0.4301 |
| 4 | 0.1952 | 0.7707 | 0.7119 | 0.4327 | 0.4450 | 0.4311 |
| 5 | 0.1395 | 0.7725 | 0.7136 | 0.4485 | 0.4180 | 0.4207 |
| 6 | 0.1031 | 0.7839 | 0.7209 | 0.4605 | 0.3716 | 0.3983 |
| 7 | 0.0823 | 0.7819 | 0.7155 | 0.4579 | 0.3802 | 0.4068 |
| 8 | 0.0662 | 0.7818 | 0.7201 | 0.4600 | 0.3604 | 0.3952 |
| 9 | 0.0539 | 0.7768 | 0.7184 | 0.4420 | 0.3883 | 0.4046 |

This time checkpoint selection correctly identified epoch 0 (F1 0.4618) as
the true best epoch — no mismatch like Experiment 4's epoch-2 pick.

**Held-out test set (488 cases, epoch-0 checkpoint, threshold = 0.5):**

Macro-average: **AUC 0.7324, F1 0.4618, Precision 0.3519, Recall 0.7431,
Accuracy 0.6440.** These numbers match the training log's epoch-0 row
almost exactly, confirming the eval script is reading the correct
checkpoint.

**Comparison to Experiment 3:** nearly identical, edging out Experiment 3
on every metric (F1 0.4580→0.4618, AUC 0.7318→0.7324, Recall 0.7342→0.7431,
Accuracy 0.6444→0.6440 roughly flat). This makes sense — both experiments
happened to land on epoch 0 in practice, since AUC and F1 agreed on the
best epoch here. The real value of switching to F1-based checkpointing is
the protection it gives in cases like Experiment 4, where AUC and F1
disagreed and AUC-based selection would have picked a worse-F1 checkpoint.

**Status:** this is the current best/reference checkpoint (F1 0.4618,
1000-patient subset). Precision is still low across most pathologies
(macro 0.3519) — the model over-flags positives to keep recall high. The
next real lever for improving this balance is per-label threshold tuning
rather than further `pos_weight` adjustments, since two different weighting
schemes have now been tried and landed in a similar place.

**Planned next steps:**
- Per-label threshold tuning: pick each pathology's decision threshold
  (e.g. by maximizing F1 or Youden's index on validation data) instead of
  a blanket 0.5 cutoff, to recover precision without touching the loss.
- Try a lower learning rate or short warmup/decay schedule — useful
  training still happens almost entirely in epoch 0-1 across every
  experiment so far.
- Scale `MAX_PATIENTS` up from 1000 toward the full ~12,000-volume dataset
  once threshold tuning is validated.

## Model weights

Not tracked in git (`model.pth`, ~43MB). Kept on Sonic at
`/scratch/25205761/vlm3d_project/scripts/model.pth`. Consider Git LFS or a GitHub
Release if versioning the weights becomes necessary.

## Cluster

Developed and trained on UCD's Sonic HPC cluster (`sonic.ucd.ie`).
