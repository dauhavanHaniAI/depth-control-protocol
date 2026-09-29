#!/usr/bin/env python3
"""albert_dcp.py — a public weight-tied model trained at a FIXED depth (ALBERT), under depth truncation and extrapolation.

ALBERT applies one shared layer num_hidden_layers times (12 or 24) and was pretrained only at that depth, so the
depth-schedule account predicts a temperature profile that departs from 1 away from the training depth.
Masked-LM NLL on fixed, seeded low-density mask sets (<=2% and <=6 masks per pass, 6 disjoint passes per document), identical for every depth; temperature and affine
readout-input maps are fitted on calibration documents disjoint from the evaluation documents (same splits as
dcp_public.py) and evaluated on the same tokens as the raw NLL. Single shared layer: prefix == repeat (C4).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import dcp_public as DP

HERE = Path(__file__).parent


def docs_text(domain, split, n=200):
    return [(d["prompt"] + d["text"]) for d in DP.load_split(domain, split)[:n]]


@torch.no_grad()
def masked_batches(tok, texts, seed, max_len=512, frac=0.02, max_masks=6, passes=6):
    """Several disjoint low-density mask sets per document. ALBERT-v2's MLM head collapses to a single token
    when a sequence contains more than a handful of [MASK] tokens (observed from about 5% masking), so each
    pass masks at most frac of the positions and at most max_masks tokens."""
    g = np.random.default_rng(seed)
    out = []
    for t in texts:
        ids = tok(t, truncation=True, max_length=max_len, return_tensors="pt").input_ids[0]
        cand = g.permutation(np.arange(1, ids.numel() - 1))
        per = max(1, min(max_masks, int(frac * len(cand))))
        for k in range(passes):
            pos = np.sort(cand[k * per:(k + 1) * per])
            if len(pos) == 0:
                break
            x = ids.clone()
            x[pos] = tok.mask_token_id
            out.append((x, torch.tensor(pos), ids[pos]))
    return out


@torch.no_grad()
def collect(model, batches, k, cap):
    model.config.num_hidden_layers = k
    Z, Y = [], []
    for x, pos, y in batches:
        model(input_ids=x[None].cuda())
        Z.append(cap["z"][0, pos.cuda()].to(torch.bfloat16).cpu())
        Y.append(y)
    return torch.cat(Z), torch.cat(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="albert/albert-base-v2")
    ap.add_argument("--depths", default="1,2,3,4,6,8,12,16,24")
    args = ap.parse_args()
    from transformers import AlbertForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AlbertForMaskedLM.from_pretrained(args.model).cuda().eval()
    # transformers>=5 does not tie predictions.bias to decoder.bias for these checkpoints (decoder bias stays 0)
    model.predictions.decoder.bias = model.predictions.bias
    L = model.config.num_hidden_layers
    head = model.predictions.decoder
    cap = {}
    head.register_forward_pre_hook(lambda m, a: cap.__setitem__("z", a[0]))
    torch.backends.cuda.matmul.allow_tf32 = True
    res = {"model": args.model, "train_depth": L, "depths": {}}
    data = {(d, s): masked_batches(tok, docs_text(d, s), seed={"math": 1, "web": 2}[d] * 10 + (s == "cal"))
            for d in ("math", "web") for s in ("eval", "cal")}
    for k in [int(v) for v in args.depths.split(",")]:
        rec = {}
        for dom in ("math", "web"):
            Ze, Ye = collect(model, data[(dom, "eval")], k, cap)
            Zc, Yc = collect(model, data[(dom, "cal")], k, cap)
            Zc, Yc, Zg, Yg = Zc.cuda(), Yc.cuda(), Ze.cuda(), Ye.cuda()
            T = DP.fit_T(head, Zc, Yc)
            a, b, lam = DP.fit_affine(head, Zc[:12000], Yc[:12000])
            raw = DP.token_stats(head, Zg, Yg, full=True)
            tc = DP.token_stats(head, Zg, Yg, T=T, full=True)
            af = DP.token_stats(head, Zg, Yg, a=a, b=b)
            rec[dom] = {"nll": float(raw["nll"].mean()), "nll_T": float(tc["nll"].mean()),
                        "nll_affine": float(af["nll"].mean()), "T": T, "acc": float(raw["correct"].mean()),
                        "ece": DP.ece(raw["conf"], raw["correct"]), "ece_T": DP.ece(tc["conf"], tc["correct"]),
                        "n_tokens": int(Ye.numel())}
            DP._WF.clear()
        res["depths"][str(k)] = rec
        print(f"k={k:3d}  math nll {rec['math']['nll']:.3f} T {rec['math']['T']:.3f} nll_T {rec['math']['nll_T']:.3f} "
              f"aff {rec['math']['nll_affine']:.3f} | web nll {rec['web']['nll']:.3f} T {rec['web']['T']:.3f} "
              f"nll_T {rec['web']['nll_T']:.3f} aff {rec['web']['nll_affine']:.3f}", flush=True)
    model.config.num_hidden_layers = L
    out = HERE / "out" / "albert"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.model.split('/')[-1]}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
