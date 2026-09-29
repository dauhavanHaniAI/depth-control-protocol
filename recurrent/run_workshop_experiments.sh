#!/bin/bash
# =====================================================================
#  Three experiments for the workshop submission, sequential on one GPU.
#
#  A. Sampled-depth intervention (the causal test of the paper's thesis).
#     Two arms continue training from the SAME checkpoint with the SAME
#     data, seed, step count and a CONSTANT learning rate; the only
#     difference is whether the per-step iteration count is sampled.
#     Then re-measure the calibration share of each arm.
#
#  B. Bootstrap intervals on the fitted temperature. Not a separate run:
#     --boot_T is passed to every measurement below, so the four-model
#     table gains uncertainty on the quantity it actually compares.
#
#  C. Domain generalisation. Same measurement on non-mathematical text
#     (the corpus' own FineWeb slice) to check the confound is not a
#     property of mathematical prose.
#
#  Usage:  bash run_workshop_experiments.sh
#  Logs:   logs/ws_*.log ; results under exp/ws_*/
# =====================================================================
set -u
cd "$(dirname "$0")"

PY=python
PYT=python
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

BASE=checkpoints/vera_v7_clean_latest.pt   # step 36000
STEPS=37000                               # 1000 steps of continued training
LR=1.64e-4                                # constant: matches the main run's
                                          # current LR (lr_peak == lr_min)
ACC=8                                     # smaller batch => ~1.6 h per arm
mkdir -p exp logs

banner () { echo; echo "=============== $* ==============="; date; echo; }

# ---------------------------------------------------------------- A, arm 1
banner "A1/4  continued training WITH sampled depth"
$PYT model/pretrain_vera_v2.py \
    --train_bin data/en_math_v4/train.bin \
    --val_bin   data/en_math_v4/val_train.bin \
    --run_name  ws_sampled --resume $BASE \
    --sample_depth \
    --epochs 0 --total_steps $STEPS --warmup_steps 100 \
    --lr_peak $LR --lr_min $LR \
    --batch_size 1 --grad_accum $ACC --seq_len 2048 \
    > logs/ws_A1_sampled.log 2>&1
echo "exit=$?"

# ---------------------------------------------------------------- A, arm 2
banner "A2/4  continued training with FIXED depth (control)"
$PYT model/pretrain_vera_v2.py \
    --train_bin data/en_math_v4/train.bin \
    --val_bin   data/en_math_v4/val_train.bin \
    --run_name  ws_fixed --resume $BASE \
    --epochs 0 --total_steps $STEPS --warmup_steps 100 \
    --lr_peak $LR --lr_min $LR \
    --batch_size 1 --grad_accum $ACC --seq_len 2048 \
    > logs/ws_A2_fixed.log 2>&1
echo "exit=$?"

# ------------------------------------------------------------ A, measure
for ARM in sampled fixed; do
  banner "A3/4  calibration measurement: $ARM arm"
  CK=checkpoints/ws_${ARM}_latest.pt
  if [ ! -f "$CK" ]; then CK=checkpoints/ws_${ARM}_final.pt; fi
  if [ ! -f "$CK" ]; then echo "!! no checkpoint for $ARM, skipping"; continue; fi
  $PY exp_depth_ablation.py --ckpt "$CK" --calibrate --boot_T 100 \
      --n_teacher 200 --n_probe 0 --n_gate 0 --iters 1,2,4,8 \
      --out exp/ws_A_${ARM} > logs/ws_A3_${ARM}.log 2>&1
  echo "exit=$?"
done

# ------------------------------------------- A, baseline for comparison
banner "A4/4  same measurement on the UNMODIFIED base checkpoint"
$PY exp_depth_ablation.py --ckpt $BASE --calibrate --boot_T 100 \
    --n_teacher 200 --n_probe 0 --n_gate 0 --iters 1,2,4,8 \
    --out exp/ws_A_base > logs/ws_A4_base.log 2>&1
echo "exit=$?"

# ---------------------------------------------------------------------- C
banner "C  domain generalisation: non-mathematical held-out text"
$PY exp_depth_ablation.py --ckpt $BASE --calibrate --boot_T 100 \
    --val data/eval/fineweb_val.jsonl \
    --n_teacher 200 --n_probe 0 --n_gate 0 --iters 1,2,4,8 \
    --out exp/ws_C_fineweb > logs/ws_C_fineweb.log 2>&1
echo "exit=$?"

# ------------------------------------------------------------------ report
banner "SUMMARY"
$PY - <<'PYEOF'
import json, pathlib
def load(p):
    f = pathlib.Path(p) / "results.json"
    return json.load(f.open()) if f.exists() else None

rows = [("base (fixed depth, unmodified)", "exp/ws_A_base"),
        ("+1000 steps, fixed depth",       "exp/ws_A_fixed"),
        ("+1000 steps, SAMPLED depth",     "exp/ws_A_sampled"),
        ("base, non-math text",            "exp/ws_C_fineweb")]
print(f"{'run':34s} {'raw gap':>8s} {'cal gap':>8s} {'cal share':>10s} {'T range':>14s}")
for name, d in rows:
    r = load(d)
    if not r:
        print(f"{name:34s} {'--':>8s}  (missing)")
        continue
    da = r["results"]["depth_ablation"]
    ks = sorted(da, key=int)
    lo, hi = da[ks[0]], da[ks[-1]]
    raw = lo["nll"] - hi["nll"]
    cal = lo.get("cal_nll", 0) - hi.get("cal_nll", 0)
    share = (raw - cal) / raw if raw else float("nan")
    Ts = [da[k].get("cal_temperature") for k in ks if "cal_temperature" in da[k]]
    trange = f"{min(Ts):.2f}-{max(Ts):.2f}" if Ts else "n/a"
    ci = ""
    if "cal_T_ci_lo" in lo:
        ci = f"  (k={ks[0]} CI {lo['cal_T_ci_lo']:.2f}-{lo['cal_T_ci_hi']:.2f})"
    print(f"{name:34s} {raw:8.4f} {cal:8.4f} {100*share:9.1f}% {trange:>14s}{ci}")
print()
print("Prediction under the paper's thesis: the SAMPLED arm's calibration")
print("share collapses toward 0 while the FIXED arm stays near the base.")
PYEOF
banner "DONE"
