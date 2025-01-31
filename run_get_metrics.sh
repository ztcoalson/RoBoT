#!/bin/bash
#SBATCH -J metrics
#SBATCH -A eecs
#SBATCH -p dgx2
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 2-0:00:00
#SBATCH --output=./logs/slurm-%j.out

module load cuda/11.2

python darts_space/get_metrics.py \
    --data ../data \
    --dataset cifar10 \
    --seed 3589 \
    --n_sample 100 \