#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=lung_eval
#SBATCH --output=eval_%j.out

cd $SLURM_SUBMIT_DIR
source ~/scratch/hugenv/bin/activate
python evaluate_2p5d.py
