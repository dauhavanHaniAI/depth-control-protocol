#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# DCB calibration grid: validate the instrument against known ground truth.
#
# Three arms, identical architecture / data / seed / step budget. The ONLY
# difference is the depth schedule seen during training, and for each arm the
# expected verdict is known in advance:
#
#   A  fixed depth 8            -> calibration confound PRESENT
#   B  sampled U{1..8}          -> ABSENT (output has seen every evaluated k)
#   C  sampled U{5..8}          -> PRESENT at k=1,2,4 (evaluated configurations
#                                  fall outside the training distribution)
#
# Arm C is the specificity test: it separates "any sampling was done" from
# "the output saw the configuration being measured". The paper currently
# reports a first attempt that failed in exactly this way as an anecdote;
# this turns it into a designed arm.
#
# Checkpoints are kept at 1500/3000/6000 so the confound can also be plotted
# against training budget. That costs disk, not GPU.
#
# Everything runs on GPU 2. GPUs 0 and 1 carry other users' processes (~20GB
# each) and a full-logit validation step OOMs there; two arms already died
# that way.
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")"
PY=python
export PYTHONPATH=.:./model
export CUDA_VISIBLE_DEVICES=2
# The first attempt peaked at 29.5 GB for a 90M model. The validation step
# materialises [batch, seq, vocab] logits in fp32 (16x1024x32000x4 = 2.1 GB,
# doubled by the reshape), and the allocator then fragments around it -- the
# same pattern that OOMed two arms on GPUs 0/1 and killed an RL run at step 47.
# Halving the microbatch while doubling accumulation keeps tokens/step at
# 32768, so the science is unchanged and only the allocation profile moves.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STEPS=6000
SUM=logs/dcb_grid_SUMMARY.txt
mkdir -p logs exp
: > "$SUM"
say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUM"; }

COMMON="--train_bin data/en_math_v4/train.bin
        --val_bin   data/en_math_v4/val_train.bin
        --dim 640 --n_heads 10 --n_kv_heads 2 --ffn_hidden 1792
        --n_perception_layers 8 --n_reasoning_blocks 2 --max_reasoning_iters 4
        --seq_len 1024 --batch_size 8 --grad_accum 4
        --total_steps $STEPS --epochs 0 --save_at 1500,3000,6000
        --lr_peak 4e-4 --lr_min 4e-5 --diag_batches 2 --no_compile"

train_arm () {   # $1 run_name  $2 extra flags  $3 mô tả
  local name=$1 extra=$2 desc=$3
  if [ -f "checkpoints/${name}_step${STEPS}.pt" ]; then
    say "  $name đã có checkpoint, bỏ qua"; return 0
  fi
  say "TRAIN $name  ($desc)"
  $PY -u -m model.pretrain_vera_v2 $COMMON --run_name "$name" \
      --proc_title python $extra > "logs/dcb_${name}.log" 2>&1
  if [ ! -f "checkpoints/${name}_step${STEPS}.pt" ]; then
    say "  THẤT BẠI: $name không sinh ra checkpoint"
    tail -5 "logs/dcb_${name}.log" | tee -a "$SUM"
    return 1
  fi
  say "  xong: $(grep -oE 'best_val=[0-9.]+' logs/dcb_${name}.log | tail -1)"
}

measure () {     # $1 run_name — chạy đủ ba đối chứng của DCB
  local name=$1
  for st in 1500 3000 6000; do
    local ck="checkpoints/${name}_step${st}.pt"
    [ -f "$ck" ] || continue
    for mode in prefix repeat suffix; do
      local out="exp/dcb_${name}_${st}_${mode}"
      [ -s "$out/results.json" ] && continue
      local flags="--mode $mode"
      [ "$mode" = prefix ] && flags="$flags --calibrate"
      $PY -u exp_depth_ablation.py --ckpt "$ck" $flags --out "$out" \
          >> "logs/dcb_measure_${name}.log" 2>&1
    done
    say "  đo xong $name @ step $st"
  done
}

say "=== LƯỚI HIỆU CHUẨN DCB, $STEPS bước/nhánh, GPU2, tuần tự ==="
train_arm dcb_A_fixed   ""                                    "độ sâu cố định 8"
train_arm dcb_B_full    "--sample_depth"                      "lấy mẫu U{1..8}"
train_arm dcb_C_partial "--sample_depth --sample_depth_min 5" "lấy mẫu U{5..8}"

say "--- đo DCB trên chín checkpoint ---"
for a in dcb_A_fixed dcb_B_full dcb_C_partial; do measure "$a"; done

say "--- tổng hợp ---"
$PY -u summarize_dcb_grid.py 2>&1 | tee -a "$SUM"
say "=== HOÀN TẤT ==="
