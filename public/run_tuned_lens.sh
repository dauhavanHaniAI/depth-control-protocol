#!/bin/bash
# Full-rank tuned-lens baselines (tuned_lens.py; lr 1e-4, 8 documents per step, 2 epochs) on the looped models.
cd "$(dirname "$0")"
PY=python
M=<MODELS_DIR>
X=<XLLM_REPO>  # github.com/ifm-ai/xllm-loop, commit 3af99f49
STUBS=$PWD/xllm_stubs:<EXTRA_DEPS>
TF455=<TRANSFORMERS_4.55_DIR>
export CUDA_VISIBLE_DEVICES=${GPU:-6} HF_DATASETS_OFFLINE=1
mkdir -p logs
ifm () { [ -s out/tuned_lens/ifm-$1/summary.json ] || $PY -u tuned_lens.py --family ifm --path $M/LoopedLM-P2-huginn-$1 --xllm $X --stubs $STUBS --tag ifm-$1 --depths 1,3,5 > logs/tl_ifm-$1.log 2>&1; }
ouro () { [ -s out/tuned_lens/ouro-$1/summary.json ] || PYTHONPATH=$TF455 $PY -u tuned_lens.py --family ouro --path $M/Ouro-$1 --tag ouro-$1 > logs/tl_ouro-$1.log 2>&1; }
case "$1" in
  A) ifm l-fixed-r5; ifm l-fixed-pln5 ;;
  B) ifm m-fixed-r5; ifm m-fixed-pln5 ;;
  C) ifm s-fixed-r5; ifm s-fixed-pln5; ouro 1.4B; ouro 2.6B ;;
esac
echo "QUEUE $1 DONE"
