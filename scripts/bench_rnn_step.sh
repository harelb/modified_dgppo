#!/bin/bash
# Benchmark different --rnn-step values to find optimal training throughput.
# Run on aspen: bash scripts/bench_rnn_step.sh
# Results are printed to stdout and saved to logs/bench_rnn_step.log

cd /home/rrg/code/modified_dgppo
VENV_CUDA_BIN=".venv/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin"
export PATH="$VENV_CUDA_BIN:$PATH"
export WANDB_MODE=disabled
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_deterministic_ops=false --xla_gpu_graph_level=3 --xla_gpu_enable_command_buffer=true"

WARMUP_STEPS=100   # steps to skip (JIT compile)
BENCH_STEPS=500    # steps to measure over
TOTAL_STEPS=$((WARMUP_STEPS + BENCH_STEPS))

N_ENV=256
BATCH=32768
LOG=logs/bench_rnn_step.log

mkdir -p logs
echo "=== RNN Step Benchmark $(date) ===" | tee $LOG
echo "n_env_train=$N_ENV  batch_size=$BATCH  warmup=$WARMUP_STEPS  bench=$BENCH_STEPS" | tee -a $LOG
echo "" | tee -a $LOG

for RNN_STEP in 8 16 32 64 128; do
    echo -n "rnn_step=$RNN_STEP ... " | tee -a $LOG

    OUTPUT=$(.venv/bin/python train.py \
        --env LidarTargetV1 \
        --algo dgppo \
        -n 1 --obs 1 \
        --seed 0 \
        --steps $TOTAL_STEPS \
        --n-env-train $N_ENV \
        --batch-size $BATCH \
        --n-env-test 16 \
        --rnn-step $RNN_STEP \
        --log-dir /tmp/bench_rnn \
        2>&1)

    # Extract it/s from steps after warmup
    ITS=$(echo "$OUTPUT" | grep -oP '\d+/'"$TOTAL_STEPS"'.*?(\d+\.\d+)it/s' | tail -1 | grep -oP '(\d+\.\d+)it/s' | head -1)
    SIT=$(echo "$OUTPUT" | grep -oP '\d+/'"$TOTAL_STEPS"'.*?(\d+\.\d+)s/it' | tail -1 | grep -oP '(\d+\.\d+)s/it' | head -1)

    if [ -n "$ITS" ]; then
        echo "$ITS" | tee -a $LOG
    elif [ -n "$SIT" ]; then
        echo "1/$(echo $SIT | tr -d 's/it') it/s (${SIT})" | tee -a $LOG
    else
        echo "could not parse rate" | tee -a $LOG
        echo "$OUTPUT" | tail -5 >> $LOG
    fi
done

echo "" | tee -a $LOG
echo "Done. Full log: $LOG"
