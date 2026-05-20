#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --array=0
#SBATCH -c 8
#SBATCH --gpus=1
#SBATCH -t 2880
#SBATCH --mem=64GB
#SBATCH -o /home/harelb/orcd/scratch/dgppo/logs/bf_lag15_%A_%a.out
#
# GPU-adaptive: batch size scales with detected VRAM.

module load miniforge
source activate dgppo

cd /home/harelb/code/modified_dgppo

# Detect GPU VRAM and choose batch parameters
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
GPU_MEM_GB=$(( GPU_MEM_MB / 1024 ))
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU_NAME  VRAM: ${GPU_MEM_GB}GB"

if   [ "$GPU_MEM_GB" -ge 130 ]; then   # H200 ~141 GB
    N_ENV_TRAIN=512; BATCH_SIZE=131072; N_ENV_TEST=128
elif [ "$GPU_MEM_GB" -ge 70 ]; then    # H100 / A100-80 ~80 GB
    N_ENV_TRAIN=512; BATCH_SIZE=65536;  N_ENV_TEST=128
elif [ "$GPU_MEM_GB" -ge 38 ]; then    # L40S ~44 GB / A100-40 ~40 GB
    N_ENV_TRAIN=256; BATCH_SIZE=32768;  N_ENV_TEST=64
else                                    # RTX6000 ~24 GB or smaller
    N_ENV_TRAIN=128; BATCH_SIZE=16384;  N_ENV_TEST=32
fi

echo "Settings: n-env-train=$N_ENV_TRAIN  batch-size=$BATCH_SIZE  n-env-test=$N_ENV_TEST"

python train.py \
    --env LidarTargetBFLag15 \
    --algo dgppo \
    -n 1 --obs 1 \
    --seed $SLURM_ARRAY_TASK_ID \
    --steps 200000 \
    --rnn-step 32 \
    --n-env-train $N_ENV_TRAIN \
    --batch-size  $BATCH_SIZE \
    --n-env-test  $N_ENV_TEST \
    --log-dir /home/harelb/orcd/scratch/dgppo/logs
