#!/bin/bash
#SBATCH -J robot
#SBATCH -A eecs
#SBATCH -p dgx2
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 2-0:00:00
#SBATCH --output=./logs/slurm-%j.out

module load cuda/11.2

python darts_space/search.py \
    --dataset cifar10 \
    --data ../data \
    --save noise4-50% \
    --seed 8836 \
    --poisons_type clean_label \
    --poisons_path /nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/poisons/poisons/noise/noise-cifar10-50.0%.pth \
    --arch_path data/noise_50p_sampled_archs.p