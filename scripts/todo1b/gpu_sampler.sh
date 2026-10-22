#!/usr/bin/env bash
# Sample per-GPU memory use every 5 s to a CSV (for peak-memory accounting per lane).
#   bash scripts/todo1b/gpu_sampler.sh <out.csv>
OUT="${1:?usage: gpu_sampler.sh <out.csv>}"
echo "utc,gpu,mem_used_mib,util_pct" > "$OUT"
while true; do
  ts=$(date -u +%FT%TZ)
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | sed "s/^/${ts},/; s/, /,/g" >> "$OUT"
  sleep 5
done
