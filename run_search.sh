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
    --save noise-diff-denoise4 \
    --seed 14106 \
    --poisons_type diffusion_denoise \
    --poisons_path '/nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/diffusion_denoise/datasets/denoised/gc_cifar10/denoise_gaussian_noise/denoised_w_sigma_0.1.pt' \
    --arch_path /nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/RoBoT/data/noise_50p_diff_denoise_sampled_archs.p