#!/bin/bash
# Submit all 4 batching experiment jobs (2x2: serial/batch × cache/nocache)

set -euo pipefail

mkdir -p slurm_logs

echo "Submitting 2x2 batching experiment..."
echo ""

JOB1=$(sbatch scripts/sbatch_serial_cache.sh | awk '{print $4}')
echo "  serial  + cache   : job $JOB1"

JOB2=$(sbatch scripts/sbatch_serial_nocache.sh | awk '{print $4}')
echo "  serial  + nocache : job $JOB2"

JOB3=$(sbatch scripts/sbatch_batch_cache.sh | awk '{print $4}')
echo "  batched + cache   : job $JOB3"

JOB4=$(sbatch scripts/sbatch_batch_nocache.sh | awk '{print $4}')
echo "  batched + nocache : job $JOB4"

echo ""
echo "All submitted. Compare timings in trajectories/test-llama-bm-622-*/*/traj_*.json"
echo "Logs in slurm_logs/"
