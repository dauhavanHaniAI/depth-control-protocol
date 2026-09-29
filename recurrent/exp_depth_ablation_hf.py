#!/usr/bin/env python3
"""
exp_depth_ablation_hf.py — do the depth-ablation confounds generalise?

Companion to exp_depth_ablation.py. That script showed, on a recurrent-depth
model, that naive truncation ablations overstate depth utility ~2x: roughly
half the apparent gap is residual arrival statistics (number of block
applications) and a quarter is output-head miscalibration.

This script asks whether the same confounds appear in ORDINARY transformers
under layer truncation — the ablation used by the layer-pruning and
early-exit literature. If they do, the required controls are not a quirk of
recurrent-depth architectures; they apply to every depth ablation.

Same three interventions, now over transformer layers instead of block
applications:

  prefix   run the first k of L layers, then final norm + lm_head
  repeat   run the first k layers, then repeat layer k until L applications
           (holds the depth the output head sees fixed; varies only DISTINCT
           computation)
  --calibrate  refit a logit temperature per k

SELF-TEST: at k = L the manual forward must reproduce model(input_ids).logits
to within float tolerance. If it does not, the layer plumbing (causal mask,
rotary embeddings) is wrong and every number below is meaningless. The script
refuses to report results unless this passes.

Usage:
  CUDA_VISIBLE_DEVICES=1 python exp_depth_ablation_hf.py \\
      --model Qwen/Qwen2.5-Math-1.5B --n_docs 100 --calibrate
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from calibration import bootstrap_nll, fit_temperature

TAG_RE = re.compile(r"</?(?:problem|think|answer)>")


def build_plan(n_layers, k, mode):
    """Which layer indices to run, in order."""
    k = max(1, min(k, n_layers))
    idx = list(range(n_layers))
    if mode == "prefix":
        return idx[:k]
    if mode == "repeat":
        return idx[:k] + [idx[k - 1]] * (n_layers - k)
    raise ValueError(mode)


@torch.no_grad()
def manual_forward(model, ids, plan):
    """Run an explicit list of layers, then the final norm and lm_head."""
    base = model.model
    h = base.embed_tokens(ids)
    pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
    pe = base.rotary_emb(h, pos)
    for li in plan:
        out = base.layers[li](h, attention_mask=None, position_ids=pos,
                              position_embeddings=pe, use_cache=False)
        h = out[0] if isinstance(out, (tuple, list)) else out
    h = base.norm(h)
    return model.lm_head(h).float()


@torch.no_grad()
def selftest(model, ids, n_layers, tol=2e-2):
    """Manual full-depth forward must match the reference forward."""
    ref = model(ids).logits.float()
    mine = manual_forward(model, ids, build_plan(n_layers, n_layers, "prefix"))
    diff = (ref - mine).abs().max().item()
    ok = diff < tol
    # a stricter check: identical argmax predictions
    agree = (ref.argmax(-1) == mine.argmax(-1)).float().mean().item()
    return ok, diff, agree


def load_docs(path, n, max_chars=4000):
    """Held-out math documents, tags stripped so the text is model-agnostic."""
    docs = []
    for line in Path(path).open():
        try:
            d = json.loads(line)
        except Exception:
            continue
        t = d.get("text") or ""
        t = TAG_RE.sub("", t).strip()
        if len(t) > 300:
            docs.append(t[:max_chars])
        if len(docs) >= n:
            break
    return docs


@torch.no_grad()
def measure(model, tok, docs, plan, device, max_len, calibrate=False):
    tot_nll, tot_tok, tot_correct = 0.0, 0, 0
    cal_logits, cal_tgts = [], []
    doc_nll, doc_tok = [], []          # per-document, for the bootstrap CI
    for t in docs:
        ids = tok(t, return_tensors="pt", truncation=True,
                  max_length=max_len).input_ids.to(device)
        if ids.shape[1] < 32:
            continue
        logits = manual_forward(model, ids, plan)
        lg, tgt = logits[0, :-1], ids[0, 1:]
        d_nll = F.cross_entropy(lg, tgt, reduction="sum").item()
        tot_nll += d_nll
        tot_tok += tgt.numel()
        tot_correct += (lg.argmax(-1) == tgt).sum().item()
        doc_nll.append(d_nll); doc_tok.append(tgt.numel())
        # Cap the calibration subset by TOKENS, not documents, and keep it in
        # bf16: with a 152k vocab one 1k-token document is ~620MB in fp32.
        if calibrate and sum(t.numel() for t in cal_tgts) < 4000:
            cal_logits.append(lg.to(torch.bfloat16).cpu())
            cal_tgts.append(tgt.cpu())
    out = {
        "nll": tot_nll / max(tot_tok, 1),
        "ppl": float(np.exp(min(tot_nll / max(tot_tok, 1), 20))),
        "next_token_acc": tot_correct / max(tot_tok, 1),
        "n_tokens": tot_tok,
    }
    out.update(bootstrap_nll(doc_nll, doc_tok))
    if calibrate and cal_tgts:
        out.update(fit_temperature(cal_logits, cal_tgts, device=device))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    ap.add_argument("--val", default="data/sft/longcot_4k/val.jsonl")
    ap.add_argument("--n_docs", type=int, default=100)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--fracs", default="0.125,0.25,0.5,0.75,1.0",
                    help="fractions of total depth to evaluate")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = getattr(torch, args.dtype)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dt, device_map=None).to(device).eval()
    n_layers = model.config.num_hidden_layers
    name = args.model.split("/")[-1]
    out = Path(args.out or f"exp/hf_depth_{name}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.model}: {n_layers} layers, "
          f"{sum(p.numel() for p in model.parameters())/1e9:.2f}B params",
          flush=True)

    docs = load_docs(args.val, args.n_docs)
    print(f"held-out documents: {len(docs)}", flush=True)

    # ── self-test: manual forward == reference forward ──
    probe = tok(docs[0], return_tensors="pt", truncation=True,
                max_length=256).input_ids.to(device)
    ok, diff, agree = selftest(model, probe, n_layers)
    print(f"self-test: max|Δlogit|={diff:.4g}  argmax agreement={agree*100:.2f}%"
          f"  → {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        print("ABORT: manual forward does not reproduce the reference forward; "
              "the layer plumbing is wrong and results would be meaningless.")
        return 1

    ks = sorted({max(1, int(round(float(f) * n_layers)))
                 for f in args.fracs.split(",")})
    rec = {
        "model": args.model, "n_layers": n_layers,
        "n_docs": len(docs), "max_len": args.max_len,
        "selftest": {"max_abs_logit_diff": diff, "argmax_agreement": agree},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": {},
    }

    for mode in ("prefix", "repeat"):
        rec["results"][mode] = {}
        print(f"\n── {mode}", flush=True)
        for k in ks:
            plan = build_plan(n_layers, k, mode)
            m = measure(model, tok, docs, plan, device, args.max_len,
                        calibrate=args.calibrate and mode == "prefix")
            m["n_applications"] = len(plan)
            m["distinct_layers"] = len(set(plan))
            rec["results"][mode][str(k)] = m
            print(f"   k={k:3d}/{n_layers} ({len(plan)} applications, "
                  f"{len(set(plan))} distinct)  nll={m['nll']:.4f}  "
                  f"ppl={m['ppl']:8.2f}  acc={m['next_token_acc']*100:5.2f}%"
                  + (f"  cal={m['cal_nll']:.4f} (T={m['cal_temperature']:.2f})"
                     if "cal_nll" in m else "")
                  + f"   [{time.time()-t0:.0f}s]", flush=True)

    # ── decomposition, same accounting as the recurrent-depth study ──
    kmin, kmax = str(ks[0]), str(ks[-1])
    pre, rep = rec["results"]["prefix"], rec["results"]["repeat"]
    total = pre[kmin]["nll"] - pre[kmax]["nll"]
    app = pre[kmin]["nll"] - rep[kmin]["nll"]
    mid = str(ks[len(ks) // 2])
    distinct = rep[kmin]["nll"] - rep[mid]["nll"]
    rec["decomposition"] = {
        "total_naive_gap": total,
        "application_count_nats": app,
        "application_count_frac": app / total if total else None,
        "distinct_depth_nats_lowmid": distinct,
        "distinct_depth_frac": distinct / total if total else None,
    }
    if args.calibrate and "cal_nll" in pre[kmin]:
        cal_gap = pre[kmin]["cal_nll"] - pre[kmax]["cal_nll"]
        rec["decomposition"]["calibrated_gap"] = cal_gap
        rec["decomposition"]["calibration_frac"] = (total - cal_gap) / total

    # ── plot ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, [pre[str(k)]["nll"] for k in ks], "o-", label="prefix (naive)",
            color="#2b6cb0")
    ax.plot(ks, [rep[str(k)]["nll"] for k in ks], "s--",
            label="repeat (applications held at L)", color="#c05621")
    if args.calibrate and "cal_nll" in pre[kmin]:
        ax.plot(ks, [pre[str(k)].get("cal_nll", np.nan) for k in ks], "^:",
                label="prefix, calibrated", color="#2f855a")
    ax.set_xlabel("distinct layers run"); ax.set_ylabel("held-out NLL")
    ax.set_title(f"Depth-truncation controls: {name}")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "fig_depth.png", dpi=150)

    (out / "results.json").write_text(json.dumps(rec, indent=2))
    d = rec["decomposition"]
    print(f"\n── {out}")
    print(f"   naive gap                 {d['total_naive_gap']:.4f} nats")
    print(f"   application-count share   {d['application_count_nats']:.4f} nats"
          f"  ({100*(d['application_count_frac'] or 0):.1f}%)")
    if "calibration_frac" in d:
        print(f"   calibration share                        "
              f"  ({100*d['calibration_frac']:.1f}%)")
    print(f"   done [{time.time()-t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
