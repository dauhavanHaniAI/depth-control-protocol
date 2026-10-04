#!/bin/bash
# DCP readout-side contrasts on externally trained looped models (predictions: PREDICTIONS_looped.md)
cd "$(dirname "$0")"
PY=python
M=<MODELS_DIR>
X=<XLLM_REPO>  # github.com/ifm-ai/xllm-loop, commit 3af99f49
STUBS=xllm_stubs:<EXTRA_DEPS>
export CUDA_VISIBLE_DEVICES=${GPU:-6}
mkdir -p logs
for t in s-fixed-r5 s-fixed-pln5 m-fixed-r5 m-fixed-pln5 l-fixed-r5 l-fixed-pln5 s-learned-entropy0p01 m-learned-entropy0p01; do
  [ -s out/looped/ifm-$t/summary.json ] && grep -q prediction_inputs out/looped/ifm-$t/summary.json && continue
  $PY -u looped_dcp.py --family ifm --path $M/LoopedLM-P2-huginn-$t --xllm $X --stubs $STUBS --tag ifm-$t > logs/looped_ifm-$t.log 2>&1
done
for t in 1.4B 2.6B; do
  [ -s out/looped/ouro-$t/summary.json ] && grep -q prediction_inputs out/looped/ouro-$t/summary.json && continue
  PYTHONPATH=<TRANSFORMERS_4.55_DIR> $PY -u looped_dcp.py --family ouro --path $M/Ouro-$t --tag ouro-$t > logs/looped_ouro-$t.log 2>&1
done
echo ALLDONE
