# VLM#D — Lung CT Classification

Vision-language/2.5D model for lung CT classification, developed on the UCD Sonic HPC cluster.

## Structure
- `scripts/train_2p5d.py` — training script
- `scripts/evaluate_2p5d.py` — evaluation script
- `scripts/submit_eval.sh` — SLURM submission script

## Model weights
Not tracked in git (`model.pth`, 43MB). Kept on Sonic at `/scratch/25205761/vlm3d_project/scripts/model.pth`, or add via Git LFS / GitHub Release if you want it versioned.

## Cluster
Trained on Sonic (sonic.ucd.ie).
