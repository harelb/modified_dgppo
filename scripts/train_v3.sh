#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --array=0-2
#SBATCH -c 8
#SBATCH --gres=gpu:l40s:1
#SBATCH -t 2880
#SBATCH --mem=32GB
#SBATCH -o /home/harelb/orcd/scratch/dgppo/logs/v3_%A_%a.out

module load miniforge
source activate dgppo

cd /home/harelb/code/modified_dgppo
python train.py \
    --env LidarTargetV3 \
    --algo dgppo \
    -n 1 --obs 1 \
    --seed $SLURM_ARRAY_TASK_ID \
    --steps 200000 \
    --n-env-train 256 \
    --batch-size 32768 \
    --n-env-test 64 \
    --log-dir /home/harelb/orcd/scratch/dgppo/logs
