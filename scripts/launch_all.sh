#!/bin/bash
# launch_all.sh — submit all variants with GPU preference + timeout fallback
#
# Tries each GPU type in order; if a job is still pending after WAIT_MIN
# minutes it cancels and tries the next type. All four variants run their
# fallback loops in parallel.
#
# Usage (run inside tmux so it survives logout):
#   tmux new-session -d -s dgppo 'bash ~/code/modified_dgppo/scripts/launch_all.sh'
#   tmux attach -t dgppo          # to monitor progress

# ── Configuration ────────────────────────────────────────────────────────────
GPU_PRIORITY=(h100 a100 l40s a40)       # order of preference; best first
WAIT_MIN=10                              # minutes to wait before trying next GPU
LOG_DIR="${HOME}/orcd/scratch/dgppo/logs"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VARIANTS=(v1 v2 v3 v4)
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$LOG_DIR"

submit_with_fallback() {
    local VARIANT=$1
    local LOG="${LOG_DIR}/launch_${VARIANT}.log"

    for GPU in "${GPU_PRIORITY[@]}"; do
        echo "[$(date '+%H:%M')] ${VARIANT}: submitting with --gpus=${GPU}:1 ..." | tee -a "$LOG"

        SUBMIT_OUT=$(sbatch --gpus=${GPU}:1 "${SCRIPT_DIR}/train_${VARIANT}.sh" 2>&1)
        JOB_ID=$(echo "$SUBMIT_OUT" | awk '/Submitted/{print $4}')

        if [ -z "$JOB_ID" ]; then
            echo "  Submit failed (${SUBMIT_OUT}), trying next GPU..." | tee -a "$LOG"
            continue
        fi

        echo "  Job ${JOB_ID} queued on ${GPU}. Waiting ${WAIT_MIN} min..." | tee -a "$LOG"
        sleep $(( WAIT_MIN * 60 ))

        STATUS=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null)

        if [ "$STATUS" = "RUNNING" ] || [ -z "$STATUS" ]; then
            # Running or already finished (not in queue)
            echo "  [$(date '+%H:%M')] Job ${JOB_ID} is ${STATUS:-DONE} on ${GPU}. Finished." | tee -a "$LOG"
            return
        fi

        echo "  Still ${STATUS} after ${WAIT_MIN} min on ${GPU}. Cancelling..." | tee -a "$LOG"
        scancel "$JOB_ID" 2>/dev/null
    done

    # All preferred types exhausted — accept any GPU
    echo "[$(date '+%H:%M')] ${VARIANT}: all preferred GPUs busy, submitting with any GPU." | tee -a "$LOG"
    sbatch --gpus=1 "${SCRIPT_DIR}/train_${VARIANT}.sh" | tee -a "$LOG"
}

echo "Starting GPU-preference launcher for: ${VARIANTS[*]}"
echo "Priority: ${GPU_PRIORITY[*]}  |  Fallback wait: ${WAIT_MIN} min each"
echo ""

# Launch all variants in parallel background loops
for V in "${VARIANTS[@]}"; do
    submit_with_fallback "$V" &
done

wait
echo ""
echo "All variants submitted. Check job status with: squeue -u $USER"
