#!/bin/bash
#SBATCH --partition=shared
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --job-name=lung_preprocess
#SBATCH --output=preprocess_%j.out
cd /scratch/25205761/vlm3d_project/scripts
source ~/scratch/hugenv/bin/activate
python preprocess_2p5d.py
