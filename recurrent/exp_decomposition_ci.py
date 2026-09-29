#!/usr/bin/env python3
"""
exp_decomposition_ci.py — confidence intervals on the depth decomposition.

The paper's opening claim is a decomposition of the naive depth gap:

    total          = prefix(k_min) - prefix(k_max)
    application    = prefix(k_min) - repeat(k_min)
    distinct depth = repeat(k_min) - repeat(k_mid)
    composition    = repeat(k_mid) - prefix(k_max)
    calibration    = total - (calibrated prefix gap)

Every term is a difference between conditions, and every share is a ratio of
such differences. A per-condition confidence interval says nothing about a
ratio of differences: the conditions are measured on the SAME documents and
are therefore strongly correlated, so independent intervals would badly
overstate the uncertainty of the shares.

This script measures all conditions over one fixed, ordered document list,
keeps the per-document losses, and bootstraps DOCUMENT INDICES ONCE PER
REPLICATE, propagating that single resample through every condition and every
derived term. That respects the correlation and gives honest intervals on the
shares themselves.

Temperatures are held at their full-sample fitted values (passed via
--temps or read from a previous results.json). The reported calibration
interval therefore reflects document sampling, not uncertainty in T.

Usage:
  CUDA_VISIBLE_DEVICES=1 python exp_decomposition_ci.py \\
      --ckpt checkpoints/dpo_v2_best.pt --n_teacher 200 \\
      --temps_from exp/depth_ctrl_calib/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model.vera_psi as _vp                                          # noqa: E402
sys.modules["vera_psi"] = _vp
from model.vera_psi import VERAPsi, VERAArgs                          # noqa: E402
from model.sft_train import StreamTagger                              # noqa: E402
from exp_depth_ablation import (build_plan, forward_instrumented,      # noqa: E402
                                load_examples, PROMPT_END)
from tokenizers import Tokenizer                                      # noqa: E402


@torch.no_grad()
def per_doc_losses(model, tok, tagger, ma, rows, plan, device, max_len,
                   ptdtype, T=1.0):
    """Per-document summed NLL over the response region, raw and at temp T."""
    nll_raw, nll_cal, ntok = [], [], []
    for d in rows:
        text = d["text"]
        ids = tok.encode(text).ids[:max_len]
        if len(ids) < 32:
            continue
        cut = text.find(PROMPT_END)
        p_len = 1
        if cut > 0:
            p_len = min(1 + len(tok.encode(text[:cut + len(PROMPT_END)],
                                           add_special_tokens=False).ids),
                        len(ids) - 1)
        x = torch.tensor([ids], device=device)
        with torch.autocast("cuda", dtype=ptdtype):
            logits, *_ = forward_instrumented(model, x, tagger.tag(x), ma,
                                              plan=plan)
        lg = logits[0, p_len - 1:-1].float()
        tgt = x[0, p_len:]
        if tgt.numel() == 0:
            continue
        nll_raw.append(F.cross_entropy(lg, tgt, reduction="sum").item())
        nll_cal.append(F.cross_entropy(lg / T, tgt, reduction="sum").item())
        ntok.append(int(tgt.numel()))
    return np.array(nll_raw), np.array(nll_cal), np.array(ntok, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/dpo_v2_best.pt")
    ap.add_argument("--tokenizer", default="tokenizer_en_math/tokenizer.json")
    ap.add_argument("--val", default="data/sft/longcot_4k/val.jsonl")
    ap.add_argument("--iters", default="1,2,4,8")
    ap.add_argument("--n_teacher", type=int, default=200)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--temps_from", default="exp/depth_ctrl_calib/results.json")
    ap.add_argument("--out", default="exp/decomposition_ci")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device, ptdtype = "cuda", torch.bfloat16
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ks = [int(k) for k in args.iters.split(",")]

    temps = {}
    tp = Path(args.temps_from)
    if tp.exists():
        prev = json.load(tp.open())["results"]["depth_ablation"]
        temps = {int(k): v.get("cal_temperature", 1.0) for k, v in prev.items()}
        print(f"temperatures from {tp}: {temps}", flush=True)
    else:
        print(f"! {tp} not found — calibration terms will use T=1", flush=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ma = ck["model_args"]
    ma = VERAArgs(**ma) if isinstance(ma, dict) else ma
    ma.gradient_checkpointing = False
    ma.dropout = 0.0
    model = VERAPsi(ma)
    model.load_state_dict(ck["model"], strict=False)
    model = model.to(device).eval()
    step = int(ck.get("step", -1)); del ck
    tok = Tokenizer.from_file(args.tokenizer)
    tagger = StreamTagger(tok)

    rows = load_examples(args.val, args.n_teacher, args.seed)
    print(f"documents: {len(rows)} (one fixed ordering, shared by all "
          f"conditions)", flush=True)

    # ── measure every condition on the SAME documents ──
    raw, cal, tokcount = {}, {}, None
    for mode in ("prefix", "repeat"):
        for k in ks:
            plan = build_plan(ma, k, mode)
            r, c, n = per_doc_losses(model, tok, tagger, ma, rows, plan,
                                     device, args.max_len, ptdtype,
                                     T=temps.get(k, 1.0))
            raw[(mode, k)] = r
            cal[(mode, k)] = c
            if tokcount is None:
                tokcount = n
            assert len(r) == len(tokcount), "condition changed the doc set"
            print(f"   {mode:7s} k={k}  nll={r.sum()/n.sum():.4f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    n_docs = len(tokcount)
    kmin, kmax = ks[0], ks[-1]
    kmid = ks[len(ks) // 2]

    # ── one resample of documents, propagated through every term ──
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n_docs, size=(args.n_boot, n_docs))
    W = tokcount[idx].sum(axis=1)

    def boot(d, key):
        return d[key][idx].sum(axis=1) / W

    p_min, p_max = boot(raw, ("prefix", kmin)), boot(raw, ("prefix", kmax))
    r_min, r_mid = boot(raw, ("repeat", kmin)), boot(raw, ("repeat", kmid))
    c_min, c_max = boot(cal, ("prefix", kmin)), boot(cal, ("prefix", kmax))

    total = p_min - p_max
    terms = {
        "application_count": p_min - r_min,
        "distinct_depth":    r_min - r_mid,
        "block_composition": r_mid - p_max,
        "calibration":       total - (c_min - c_max),
    }

    def ci(v, a=0.05):
        lo, hi = np.percentile(v, [100 * a / 2, 100 * (1 - a / 2)])
        return float(np.mean(v)), float(lo), float(hi)

    rec = {
        "experiment": "depth_decomposition_bootstrap_ci",
        "ckpt": args.ckpt, "ckpt_step": step,
        "n_docs": int(n_docs), "n_boot": args.n_boot,
        "iters": ks, "k_min": kmin, "k_mid": kmid, "k_max": kmax,
        "temperatures_used": {str(k): temps.get(k, 1.0) for k in ks},
        "note": ("Document indices are resampled once per replicate and the "
                 "same resample is used for every condition, so the intervals "
                 "on the shares account for the correlation between "
                 "conditions. Temperatures are held at their full-sample "
                 "values, so the calibration interval reflects document "
                 "sampling only."),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    m, lo, hi = ci(total)
    rec["total_gap"] = {"nats": m, "ci_lo": lo, "ci_hi": hi}
    print(f"\n── decomposition, {n_docs} docs, {args.n_boot} bootstrap "
          f"replicates")
    print(f"   total naive gap   {m:7.4f} nats   95% CI [{lo:.4f}, {hi:.4f}]")
    rec["terms"] = {}
    for name, v in terms.items():
        nm, nlo, nhi = ci(v)
        sm, slo, shi = ci(v / total)          # ratio computed PER REPLICATE
        rec["terms"][name] = {
            "nats": nm, "nats_ci_lo": nlo, "nats_ci_hi": nhi,
            "share": sm, "share_ci_lo": slo, "share_ci_hi": shi,
            "share_excludes_zero": bool(slo > 0 or shi < 0),
        }
        star = "" if (slo <= 0 <= shi) else "  *"
        print(f"   {name:18s} {nm:7.4f} nats  [{nlo:7.4f},{nhi:7.4f}]   "
              f"share {100*sm:6.1f}%  [{100*slo:6.1f}%,{100*shi:6.1f}%]{star}")
    print("   (* share interval excludes zero)")

    (out / "results.json").write_text(json.dumps(rec, indent=2))
    print(f"\n── {out}/results.json   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    raise SystemExit(main())
