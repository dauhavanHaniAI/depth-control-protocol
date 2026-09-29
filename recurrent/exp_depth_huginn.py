#!/usr/bin/env python3
"""
exp_depth_huginn.py — does the calibration confound come from training at a
fixed recurrence depth?

Section 5 of the paper found that truncating a fixed-depth recurrent model
leaves its output head miscalibrated: the fitted logit temperature drifts
from 1.42 at k=1 to 0.90 at k=8, and refitting it closes ~26% of the naive
depth gap. Section 6 showed this does NOT happen under layer truncation in
standard transformers.

Huginn-0125 lets us test the mechanism directly. It is a recurrent-depth
model like Sona, but its recurrence count is SAMPLED DURING TRAINING
(config: mean_recurrence=32, sampling_scheme=poisson-lognormal-filling,
verified in randomized_iteration_sampler()). Its output head therefore sees
residuals arriving from many different depths, where Sona's only ever sees
depth 8.

  HYPOTHESIS: the calibration confound is caused by fixed-depth training.
  A depth-randomised model should show little or no temperature drift and a
  near-zero calibration share.

  If confirmed, the paper gains a mechanism and a prescription (sample the
  recurrence depth during training) rather than just a warning.

NOTE ON THE MISSING CONTROL. Huginn's recurrent core is a SINGLE weight-tied
block, so every application is identical and there is no distinct-vs-repeated
axis: `prefix k` and `repeat k` coincide by construction. The application-
count decomposition of Section 5 cannot be replicated here, and this script
does not attempt it. Only the calibration arm transfers.

COMPATIBILITY. Huginn ships remote code written for transformers 4.44.2;
three things must be handled to run it under transformers 5.x / torch 2.11:

  1. `_tied_weights_keys` is a list, but >=5 expects a {target: source} dict
     → install_legacy_tied_weights_shim() below.
  2. flex_attention JIT-compiles a kernel and needs libnvrtc-builtins, which
     ships inside the venv but is not on the loader path
     → export LD_LIBRARY_PATH=<venv>/lib/python3.11/site-packages/nvidia/cu13/lib
  3. iterate_forward() calls len(num_steps) on anything with __len__, which
     raises on a 0-d tensor → we pass a plain Python int.

Usage:
  export LD_LIBRARY_PATH=<venv>/lib/python3.x/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
  CUDA_VISIBLE_DEVICES=1 python exp_depth_huginn.py \\
      --steps 1,2,4,8,16,32 --n_docs 60 --max_len 512
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

from calibration import bootstrap_nll, fit_temperature

TAG_RE = re.compile(r"</?(?:problem|think|answer)>")

# Reference values from the fixed-depth model, for the summary table only.
SONA_REF = {
    "model": "Sona (fixed depth 8)",
    "T_at_min_depth": 1.42, "T_at_max_depth": 0.90,
    "temperature_drift": 0.52, "calibration_share": 0.260,
}


def load_docs(path, n, max_chars=4000):
    docs = []
    for line in Path(path).open():
        try:
            d = json.loads(line)
        except Exception:
            continue
        t = TAG_RE.sub("", d.get("text") or "").strip()
        if len(t) > 300:
            docs.append(t[:max_chars])
        if len(docs) >= n:
            break
    return docs


@torch.no_grad()
def measure(model, tok, docs, r, device, max_len, cal_token_budget=4000):
    """NLL / accuracy at recurrence depth r, plus a fitted logit temperature."""
    tot_nll, tot_tok, tot_correct = 0.0, 0, 0
    cal_logits, cal_tgts = [], []
    doc_nll, doc_tok = [], []          # per-document, for the bootstrap CI
    # Pass a plain int: raven's iterate_forward does
    #   hasattr(num_steps, "__len__") and len(num_steps) > 1
    # and a 0-d tensor HAS __len__ but raises TypeError when called.
    steps = int(r)
    for t in docs:
        ids = tok(t, return_tensors="pt", truncation=True,
                  max_length=max_len).input_ids.to(device)
        if ids.shape[1] < 32:
            continue
        out = model(input_ids=ids, num_steps=steps)
        logits = (out.logits if hasattr(out, "logits") else out[0]).float()
        lg, tgt = logits[0, :-1], ids[0, 1:]
        d_nll = F.cross_entropy(lg, tgt, reduction="sum").item()
        tot_nll += d_nll
        tot_tok += tgt.numel()
        tot_correct += (lg.argmax(-1) == tgt).sum().item()
        doc_nll.append(d_nll); doc_tok.append(tgt.numel())
        if sum(x.numel() for x in cal_tgts) < cal_token_budget:
            cal_logits.append(lg.to(torch.bfloat16).cpu())
            cal_tgts.append(tgt.cpu())

    res = {
        "nll": tot_nll / max(tot_tok, 1),
        "ppl": float(np.exp(min(tot_nll / max(tot_tok, 1), 20))),
        "next_token_acc": tot_correct / max(tot_tok, 1),
        "n_tokens": tot_tok,
    }
    res.update(bootstrap_nll(doc_nll, doc_tok))
    if cal_tgts:
        res.update(fit_temperature(cal_logits, cal_tgts, device=device))
    return res


def install_legacy_tied_weights_shim():
    """Huginn's remote code targets transformers 4.44, which declared
    `_tied_weights_keys` as a flat LIST of tied parameter names. transformers
    >=5 expects a {target: source} MAPPING and calls .keys() on it, so loading
    dies in post_init() with:

        AttributeError: 'list' object has no attribute 'keys'

    We convert the legacy form to the modern one at load time by pairing each
    tied target with the input-embedding parameter, which is what the list
    form implicitly meant. Touches only models that ship the old convention.
    """
    import transformers.modeling_utils as MU

    orig = MU.PreTrainedModel.get_expanded_tied_weights_keys

    def _embedding_param_name(model):
        try:
            emb = model.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight"):
                for n, p in model.named_parameters(remove_duplicate=False):
                    if p is emb.weight:
                        return n
        except Exception:
            pass
        for suffix in ("wte.weight", "embed_tokens.weight", "embeddings.weight"):
            for n, _ in model.named_parameters(remove_duplicate=False):
                if n.endswith(suffix):
                    return n
        return None

    def patched(self, all_submodels=False):
        tw = getattr(self, "_tied_weights_keys", None)
        if isinstance(tw, (list, tuple)):
            src = _embedding_param_name(self)
            self._tied_weights_keys = {k: (src or k) for k in tw}
        return orig(self, all_submodels=all_submodels)

    MU.PreTrainedModel.get_expanded_tied_weights_keys = patched
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tomg-group-umd/huginn-0125")
    ap.add_argument("--val", default="data/sft/longcot_4k/val.jsonl")
    ap.add_argument("--steps", default="1,2,4,8,16,32",
                    help="recurrence depths to evaluate")
    ap.add_argument("--n_docs", type=int, default=60)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="exp/huginn_depth")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    shimmed = install_legacy_tied_weights_shim()
    print(f"loading {args.model} (trust_remote_code; legacy tied-weights "
          f"shim={'on' if shimmed else 'off'})", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        dtype=getattr(torch, args.dtype)).to(device).eval()
    cfg = model.config
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e9:.2f}B params | prelude={getattr(cfg,'n_layers_in_prelude','?')} "
          f"core={getattr(cfg,'n_layers_in_recurrent_block','?')} "
          f"coda={getattr(cfg,'n_layers_in_coda','?')} "
          f"mean_recurrence={getattr(cfg,'mean_recurrence','?')}", flush=True)

    docs = load_docs(args.val, args.n_docs)
    print(f"held-out documents: {len(docs)}", flush=True)

    steps = [int(s) for s in args.steps.split(",")]
    rec = {
        "experiment": "huginn_depth_calibration",
        "model": args.model,
        "params_B": round(n_params / 1e9, 3),
        "mean_recurrence_train": getattr(cfg, "mean_recurrence", None),
        "sampling_scheme": getattr(cfg, "sampling_scheme", None),
        "n_docs": len(docs), "max_len": args.max_len,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": ("Huginn's recurrent core is a single weight-tied block, so "
                 "prefix and repeat controls coincide; only the calibration "
                 "arm of the Section 5 decomposition transfers."),
        "legacy_tied_weights_shim": True,
        "transformers_version_declared": "4.44.2",
        "results": {},
    }

    print("\n── recurrence depth sweep", flush=True)
    for r in steps:
        m = measure(model, tok, docs, r, device, args.max_len)
        rec["results"][str(r)] = m
        print(f"   r={r:3d}  nll={m['nll']:.4f}  ppl={m['ppl']:9.2f}  "
              f"acc={m['next_token_acc']*100:5.2f}%"
              + (f"  cal={m['cal_nll']:.4f} (T={m['cal_temperature']:.2f})"
                 if "cal_nll" in m else "")
              + f"   [{time.time()-t0:.0f}s]", flush=True)

    # ── the hypothesis test ──
    lo, hi = str(steps[0]), str(steps[-1])
    R = rec["results"]
    raw_gap = R[lo]["nll"] - R[hi]["nll"]
    verdict = {}
    if "cal_nll" in R[lo] and "cal_nll" in R[hi]:
        cal_gap = R[lo]["cal_nll"] - R[hi]["cal_nll"]
        temps = [R[str(r)]["cal_temperature"] for r in steps]
        verdict = {
            "raw_gap": raw_gap,
            "calibrated_gap": cal_gap,
            "calibration_share": (raw_gap - cal_gap) / raw_gap if raw_gap else None,
            "T_at_min_depth": temps[0], "T_at_max_depth": temps[-1],
            "temperature_drift": max(temps) - min(temps),
            "temperatures": dict(zip(map(str, steps), temps)),
        }
        share = verdict["calibration_share"]
        drift = verdict["temperature_drift"]
        verdict["verdict"] = (
            "SUPPORTS HYPOTHESIS — depth-randomised training leaves the head "
            "calibrated across depths"
            if (share is not None and abs(share) < 0.10 and drift < 0.25) else
            "DOES NOT SUPPORT — calibration drifts even under randomised-depth "
            "training")
    rec["hypothesis_test"] = verdict
    rec["reference_fixed_depth_model"] = SONA_REF

    # ── plot ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(steps, [R[str(r)]["nll"] for r in steps], "o-", color="#2b6cb0",
               label="raw")
    if verdict:
        ax[0].plot(steps, [R[str(r)]["cal_nll"] for r in steps], "^--",
                   color="#2f855a", label="calibrated")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("recurrence depth r"); ax[0].set_ylabel("held-out NLL")
    ax[0].set_title("Huginn: NLL vs recurrence"); ax[0].legend(); ax[0].grid(alpha=.3)
    if verdict:
        ax[1].plot(steps, [R[str(r)]["cal_temperature"] for r in steps], "o-",
                   color="#b83280", label="Huginn (depth-randomised training)")
        ax[1].axhline(1.0, ls=":", c="k", lw=1)
        ax[1].plot([steps[0], steps[-1]],
                   [SONA_REF["T_at_min_depth"], SONA_REF["T_at_max_depth"]],
                   "s--", color="#c05621", label="Sona (fixed-depth training)")
        ax[1].set_xscale("log", base=2)
        ax[1].set_xlabel("recurrence depth r")
        ax[1].set_ylabel("fitted logit temperature")
        ax[1].set_title("Calibration drift"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "fig_huginn.png", dpi=150)

    rec["runtime_sec"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(rec, indent=2))

    print(f"\n── {out}")
    if verdict:
        print(f"   raw gap {verdict['raw_gap']:.4f} → calibrated "
              f"{verdict['calibrated_gap']:.4f}")
        print(f"   calibration share {100*verdict['calibration_share']:.1f}% "
              f"(Sona, fixed depth: {100*SONA_REF['calibration_share']:.1f}%)")
        print(f"   temperature drift {verdict['temperature_drift']:.3f} "
              f"(Sona: {SONA_REF['temperature_drift']:.3f})")
        print(f"   → {verdict['verdict']}")
    print(f"   [{rec['runtime_sec']}s]")


if __name__ == "__main__":
    raise SystemExit(main())
