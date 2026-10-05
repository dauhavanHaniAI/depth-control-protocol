#!/usr/bin/env python3
"""tuned_lens.py — full-rank (d x d) tuned-lens baselines for the readout-side probes on looped models.

For each truncated depth r, a translator h -> h + A h + b (A, b initialized at zero, i.e. the identity) acts on the
state that enters the model's own final normalization and readout, as in the tuned lens (Belrose et al.). Two
objectives are trained on MATH *train* solutions, selected on the disjoint calibration documents and evaluated on
the evaluation documents used everywhere else (data/math_{cal,eval}.jsonl):

  kl  : tuned-lens objective, KL(p_R || p_lens(h_r)) to the model's own predictions at its training depth R.
        At r = R the translator is the identity by construction.
  nll : next-token NLL. Also trained at r = R (full-depth control), so recoverable shares are net of what the
        same high-capacity probe gains at full depth.

Outputs per-document NLL sums and paired document-bootstrap shares of the r -> R gap.

Example:
  python tuned_lens.py --family ifm --path <DIR>/LoopedLM-P2-huginn-s-fixed-r5 --xllm <XLLM> --stubs <STUBS> --tag ifm-s-fixed-r5
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import dcp_public as DP
from looped_dcp import IFMRunner, OuroRunner

HERE = Path(__file__).parent
dev = "cuda"
SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus"]


def math_train_docs(n, seed=0):
    from datasets import load_dataset
    rows = []
    for s in SUBJECTS:
        for r in load_dataset("EleutherAI/hendrycks_math", s, split="train"):
            if 400 <= len(r["solution"]) <= 3000:  # same filter as the evaluation split
                rows.append({"prompt": f"Problem: {r['problem']}\nSolution:", "text": " " + r["solution"]})
    random.Random(seed).shuffle(rows)
    return rows[:n]


class Readout:
    """Differentiable copy of the model's own final normalization + output layer, and pre-norm state capture."""

    def __init__(self, run, family):
        self.run, self.family = run, family
        self.cap = []
        if family == "ifm":
            fn = run.m.output.final_norm
            self.norm_w = (fn.weight.float() + 1.0).detach()
            self.eps = fn.eps
            self.W = run.m.output.output.weight.detach()
            fn.register_forward_pre_hook(lambda mod, a: self.cap.append(a[0]))
        else:
            nm = run.m.model.norm
            self.norm_w = nm.weight.float().detach()
            self.eps = nm.variance_epsilon
            self.W = run.m.lm_head.weight.detach()
            nm.register_forward_pre_hook(lambda mod, a: self.cap.append(a[0]))

    def logits(self, h):
        h = h.float()
        y = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps) * self.norm_w
        return F.linear(y.to(self.W.dtype), self.W).float()

    @torch.no_grad()
    def states(self, ids, depths, doc_id):
        """Pre-norm states entering the readout after r recurrences (loops), for r in depths."""
        out = {}
        if self.family == "ifm":
            for r in depths:
                self.cap.clear()
                self.run.m.terminal_kv_forward(ids[None].to(dev), torch.tensor([doc_id]), 42, full_logits=True, depth=r)
                out[r] = self.cap[-1][0]
        else:
            self.cap.clear()
            self.run.m.model.total_ut_steps = max(depths)
            self.run.m.model(input_ids=ids[None].to(dev), use_cache=False)
            for r in depths:
                out[r] = self.cap[r - 1][0]
        return out


class Translator(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.A = torch.nn.Parameter(torch.zeros(d, d, device=dev))
        self.b = torch.nn.Parameter(torch.zeros(d, device=dev))

    def forward(self, h):
        h = h.float()
        return h + h @ self.A.T + self.b


def collect(ro, toks, depths):
    S = {r: [] for r in depths}
    Y, D = [], []
    for di, (ids, s) in enumerate(toks):
        if ids.numel() - s - 1 <= 0:
            continue
        st = ro.states(ids, depths, di)
        for r in depths:
            S[r].append(st[r][s:-1].to(torch.bfloat16).cpu())
        Y.append(ids[s + 1:])
        D.append(torch.full((ids.numel() - s - 1,), di))
    return {r: torch.cat(v) for r, v in S.items()}, torch.cat(Y), torch.cat(D)


@torch.no_grad()
def token_nll(ro, H, Y, lens=None, bs=1024):
    out = []
    for i in range(0, Y.numel(), bs):
        h = H[i:i + bs].to(dev)
        if lens is not None:
            h = lens(h)
        out.append(F.cross_entropy(ro.logits(h), Y[i:i + bs].to(dev), reduction="none").cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def mean_kl(ro, H, Href, lens, bs=1024):
    tot, n = 0.0, 0
    for i in range(0, Href.shape[0], bs):
        lt = ro.logits(Href[i:i + bs].to(dev)).log_softmax(-1)
        h = H[i:i + bs].to(dev)
        lp = ro.logits(lens(h) if lens is not None else h).log_softmax(-1)
        tot += F.kl_div(lp, lt, log_target=True, reduction="sum").item()
        n += h.shape[0]
    return tot / n


def per_doc(nll, D, n):
    d = D.numpy()
    return np.bincount(d, weights=nll, minlength=n), np.bincount(d, minlength=n).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["ifm", "ouro"])
    ap.add_argument("--path", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--xllm", default=None)
    ap.add_argument("--stubs", default="")
    ap.add_argument("--depths", default=None)
    ap.add_argument("--train_docs", type=int, default=4000)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8, help="training documents per optimizer step")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--out", default="tuned_lens")
    ap.add_argument("--nb", type=int, default=2000)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    run = IFMRunner(args.path, args.xllm, [p for p in args.stubs.split(":") if p]) if args.family == "ifm" \
        else OuroRunner(args.path)
    R = run.ref
    depths = [int(x) for x in args.depths.split(",")] if args.depths else list(range(1, R + 1))
    assert R in depths
    ro = Readout(run, args.family)
    tok = run.tok
    if run.bos is not None:
        bos = run.bos
    else:
        first = tok("a", add_special_tokens=True).input_ids[0]
        bos = [first] if tok.bos_token_id is not None and first == tok.bos_token_id else []
    out_dir = HERE / "out" / args.out / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # sanity: the differentiable readout reproduces the model's logits at R
    ids0, s0 = DP.tokenize(tok, DP.load_split("math", "eval")[:1], 1024, bos)[0]
    _, lg0 = run.readout_inputs(ids0, [R], 0, want_logits=True)
    st0 = ro.states(ids0, [R], 0)
    sanity = float((ro.logits(st0[R]) - lg0).abs().max())
    print("sanity", sanity, flush=True)

    te = DP.tokenize(tok, DP.load_split("math", "eval"), 1024, bos)
    tc = DP.tokenize(tok, DP.load_split("math", "cal"), 1024, bos)
    tt = DP.tokenize(tok, math_train_docs(args.train_docs), 1024, bos)
    He, Ye, De = collect(ro, te, depths)
    Hc, Yc, _ = collect(ro, tc, depths)
    print(f"cached eval/cal states [{time.time() - t0:.0f}s]", flush=True)
    d = He[R].shape[1]

    lenses = {("nll", r): Translator(d) for r in depths}
    lenses.update({("kl", r): Translator(d) for r in depths if r != R})
    opts = {k: torch.optim.Adam(v.parameters(), lr=args.lr) for k, v in lenses.items()}

    def score(k, lens):
        obj, r = k
        if obj == "nll":
            return float(token_nll(ro, Hc[r], Yc, lens).mean())
        return mean_kl(ro, Hc[r], Hc[R], lens)

    best = {k: (score(k, None), None, 0) for k in lenses}  # identity is always a candidate
    log = [{"step": 0, **{f"{k[0]}_r{k[1]}": best[k][0] for k in lenses}}]
    print(json.dumps(log[0]), flush=True)
    order = []
    for e in range(args.epochs):
        o = list(range(len(tt)))
        random.Random(1 + e).shuffle(o)
        order += o
    for step, i in enumerate(order, 1):
        ids, s = tt[i]
        if ids.numel() - s - 1 <= 0:
            continue
        st = ro.states(ids, depths, 10_000_000 + i)
        y = ids[s + 1:].to(dev)
        with torch.no_grad():
            teacher = ro.logits(st[R][s:-1]).log_softmax(-1)
        for k, lens in lenses.items():
            obj, r = k
            lp = ro.logits(lens(st[r][s:-1])).log_softmax(-1)
            loss = F.nll_loss(lp, y) if obj == "nll" else F.kl_div(lp, teacher, log_target=True, reduction="batchmean")
            (loss / args.accum).backward()
            if step % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(lens.parameters(), 1.0)
                opts[k].step()
                opts[k].zero_grad(set_to_none=True)
        if step % args.eval_every == 0 or step == len(order):
            rec = {"step": step}
            for k, lens in lenses.items():
                v = score(k, lens)
                rec[f"{k[0]}_r{k[1]}"] = v
                if v < best[k][0]:
                    best[k] = (v, {n: p.detach().cpu().clone() for n, p in lens.state_dict().items()}, step)
            log.append(rec)
            print(json.dumps(rec), f"[{time.time() - t0:.0f}s]", flush=True)

    # evaluation with the selected translators
    for k, lens in lenses.items():
        if best[k][1] is None:
            with torch.no_grad():
                lens.A.zero_(); lens.b.zero_()
        else:
            lens.load_state_dict({n: p.to(dev) for n, p in best[k][1].items()})
    n_e = len(te)
    raw = {r: per_doc(token_nll(ro, He[r], Ye), De, n_e) for r in depths}
    nl = {r: per_doc(token_nll(ro, He[r], Ye, lenses[("nll", r)]), De, n_e) for r in depths}
    kl = {r: per_doc(token_nll(ro, He[r], Ye, lenses[("kl", r)]), De, n_e) for r in depths if r != R}
    kl[R] = raw[R]
    rng = np.random.default_rng(0)
    idx = np.vstack([np.arange(n_e)[None], rng.integers(0, n_e, size=(args.nb, n_e))])
    Q = lambda vc: vc[0][idx].sum(1) / vc[1][idx].sum(1)
    ci = lambda x: [float(np.percentile(x[1:], 2.5)), float(np.percentile(x[1:], 97.5))]
    res = {"tag": args.tag, "family": args.family, "ref_depth": R, "depths": depths, "d": d,
           "translator_params": d * d + d, "train_docs": args.train_docs, "lr": args.lr, "accum": args.accum, "epochs": args.epochs,
           "train_tokens": int(sum(max(0, i.numel() - s - 1) for i, s in tt)),
           "sanity_max_abs_logit_diff": sanity, "selected_step": {f"{k[0]}_r{k[1]}": best[k][2] for k in lenses},
           "nll_raw": {str(r): float(Q(raw[r])[0]) for r in depths},
           "nll_lens_nll": {str(r): float(Q(nl[r])[0]) for r in depths},
           "nll_lens_kl": {str(r): float(Q(kl[r])[0]) for r in depths},
           "full_depth_gain_nll_lens": float(Q(raw[R])[0] - Q(nl[R])[0]), "vs_ref": {}, "log": log}
    for r in depths:
        if r == R:
            continue
        gr = Q(raw[r]) - Q(raw[R])
        gn = Q(nl[r]) - Q(nl[R])
        gk = Q(kl[r]) - Q(raw[R])
        res["vs_ref"][str(r)] = {"gap_raw": float(gr[0]),
                                 "share_nll_lens": float(((gr - gn) / gr)[0]), "share_nll_lens_ci95": ci((gr - gn) / gr),
                                 "share_tuned_lens_kl": float(((gr - gk) / gr)[0]), "share_tuned_lens_kl_ci95": ci((gr - gk) / gr)}
    np.savez_compressed(out_dir / "perdoc_math.npz",
                        **{f"{n}_r{r}_{w}": v[r][j] for n, v in (("raw", raw), ("nll", nl), ("kl", kl))
                           for r in depths for j, w in enumerate(("sum", "count"))})
    (out_dir / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("log",)}, indent=1), "done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
