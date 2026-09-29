#!/usr/bin/env bash
# Step 2: DCP on public models. Usage: bash run_step2.sh <gpu> <kind> <model> [<model> ...]
# Each model resumes from out/<tag>/summary.json if interrupted.
set -u
cd "$(dirname "$0")"
PY=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=<venv>/lib/python3.x/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
gpu=$1; kind=$2; shift 2
mkdir -p logs
for m in "$@"; do
  tag=$(basename "$m")
  for attempt in 1 2 3; do
    echo "[$(date +%F' '%T)] start $m (attempt $attempt) on GPU $gpu" >> logs/lanes.log
    CUDA_VISIBLE_DEVICES=$gpu $PY -u dcp_public.py --model "$m" --kind "$kind" --tag "$tag" >> "logs/$tag.log" 2>&1 && break
    echo "[$(date +%F' '%T)] FAILED $m attempt $attempt (see logs/$tag.log)" >> logs/lanes.log
    sleep 60
  done
  echo "[$(date +%F' '%T)] end $m" >> logs/lanes.log
done
