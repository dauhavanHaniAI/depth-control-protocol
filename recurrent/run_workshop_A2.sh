#!/bin/bash
# =====================================================================
#  Experiment A, corrected sampler.
#
#  First attempt sampled iterations PER BLOCK, so training always ran both
#  reasoning blocks and produced only 2/4/6/8 total applications. The depth
#  ablation truncates to a PREFIX of the 8 applications, so its k<=4
#  configurations run block 0 alone — configurations training never
#  produced. The output head was therefore never exposed to what was being
#  measured, and the calibration share could not move.
#
#  This run samples k ~ U{1..8} TOTAL applications and runs the prefix plan
#  of k, matching build_plan(mode="prefix") in exp_depth_ablation.py.
#
#  The fixed-depth control arm is unaffected by the sampler, so
#  exp/ws_A_fixed is reused rather than retrained.
# =====================================================================
set -u
cd "$(dirname "$0")"

PY=python
PYT=python
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

BASE=checkpoints/vera_v7_clean_latest.pt
banner () { echo; echo "=============== $* ==============="; date; echo; }

banner "1/2  continued training, PREFIX-plan sampled depth"
$PYT model/pretrain_vera_v2.py \
    --train_bin data/en_math_v4/train.bin \
    --val_bin   data/en_math_v4/val_train.bin \
    --run_name  ws_sampled2 --resume $BASE \
    --sample_depth \
    --epochs 0 --total_steps 37000 --warmup_steps 100 \
    --lr_peak 1.64e-4 --lr_min 1.64e-4 \
    --batch_size 1 --grad_accum 8 --seq_len 2048 \
    > logs/ws_A2b_sampled_prefix.log 2>&1
echo "exit=$?"

banner "2/2  calibration measurement"
CK=checkpoints/ws_sampled2_latest.pt
[ -f "$CK" ] || CK=checkpoints/ws_sampled2_final.pt
$PY exp_depth_ablation.py --ckpt "$CK" --calibrate --boot_T 100 \
    --n_teacher 200 --n_probe 0 --n_gate 0 --iters 1,2,4,8 \
    --out exp/ws_A_sampled_prefix > logs/ws_A2b_measure.log 2>&1
echo "exit=$?"

banner "SUMMARY"
$PY - <<'PYEOF'
import json, pathlib
rows = [("base (unmodified)",              "exp/ws_A_base"),
        ("+1000, fixed depth (control)",   "exp/ws_A_fixed"),
        ("+1000, sampled PER BLOCK (v1)",  "exp/ws_A_sampled"),
        ("+1000, sampled PREFIX (v2)",     "exp/ws_A_sampled_prefix")]
print(f"{'run':32s} {'raw gap':>8s} {'cal gap':>8s} {'cal share':>10s} "
      f"{'T@k=1':>7s} {'T@k=8':>7s} {'NLL@k=1':>8s}")
for name, d in rows:
    f = pathlib.Path(d) / "results.json"
    if not f.exists():
        print(f"{name:32s}  (missing)"); continue
    da = json.load(f.open())["results"]["depth_ablation"]
    ks = sorted(da, key=int); lo, hi = da[ks[0]], da[ks[-1]]
    raw = lo["nll"] - hi["nll"]; cal = lo.get("cal_nll",0) - hi.get("cal_nll",0)
    share = (raw - cal)/raw if raw else float("nan")
    print(f"{name:32s} {raw:8.4f} {cal:8.4f} {100*share:9.1f}% "
          f"{lo.get('cal_temperature',0):7.3f} {hi.get('cal_temperature',0):7.3f} "
          f"{lo['nll']:8.4f}")
print()
print("Thesis prediction: the PREFIX-sampled arm's calibration share collapses")
print("toward 0 and T@k=1 approaches 1.0, while the control stays near base.")
PYEOF
banner "DONE"
