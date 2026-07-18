#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=lung_train
#SBATCH --output=train_%j.out
cd /scratch/25205761/vlm3d_project/scripts
source ~/scratch/hugenv/bin/activate
python train_2p5d.py
