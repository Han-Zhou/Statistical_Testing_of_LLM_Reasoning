#!/bin/bash
set -euo pipefail

mkdir -p scripts/qwen_ascend/bm/logs

for part in $(seq -w 2 10); do
    log="scripts/qwen_ascend/bm/logs/part${part}.log"
    bash "scripts/qwen_ascend/bm/qwen_ascend_bm_part${part}.sh" > "$log" 2>&1 &
    echo "Part ${part} started (PID $!) -> $log"
done

echo ""
echo "All 10 parts launched. Monitor with:"
echo "  tail -f scripts/qwen_ascend/bm/logs/part*.log   # all"
echo "  tail -f scripts/qwen_ascend/bm/logs/part01.log   # one"
echo ""
echo "Waiting for all to finish..."
wait
echo "All parts complete."
