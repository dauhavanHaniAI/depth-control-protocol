#!/usr/bin/env python3
"""looped_dcp.py — DCP readout-side contrasts on externally trained looped language models.

Families:
  ifm  : IFM LoopedLM-P2 (huginn-style prelude / recurrent core / coda), loaded with the xLLM release code.
         Depth r = number of recurrences; the readout is coda + final norm + output layer.
  ouro : ByteDance Ouro (a 48-layer stack looped total_ut_steps times; every loop is read out).

Protocol and predictions are fixed in PREDICTIONS_looped.md (written before any measurement).
For every depth r: raw NLL, NLL after a depth-specific temperature, and NLL after a ridge-regularised affine
readout-input map, both fitted on calibration documents disjoint from the evaluation documents; per-document
sums are stored so that shares relative to the training depth get paired document-bootstrap intervals.
Single-block cores without iteration conditioning admit no repeat control (Condition C4), so only the
readout-side contrasts and extrapolation beyond the trained depth are measured.

Example:
  python looped_dcp.py --family ifm  --path <DIR>/LoopedLM-P2-huginn-s-fixed-r5 --xllm <XLLM_REPO> --tag ifm-s-fixed-r5
  python looped_dcp.py --family ouro --path <DIR>/Ouro-1.4B --tag ouro-1.4b
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import dcp_public as DP

HERE = Path(__file__).parent
dev = "cuda"
DOMAINS = ("math", "web")


class IFMRunner:
    def __init__(self, path, xllm, stubs):
        # xLLM's compiled kernels are not needed for inference: the pure-PyTorch attention path
        # (causal_attn_backend=None) and an fp32 RMSNorm stand-in replace them (see xllm_stubs/).
        sys.path[:0] = [str(xllm)] + [str(p) for p in stubs]
        import xllm_autostub  # noqa: F401
        from transformers import AutoTokenizer
        from xllm.paper_part2.artifacts import build_model, load_artifact
        fields, sd = load_artifact(path)
        fields = dict(fields)
        fields["causal_attn_backend"] = None
        self.tok = AutoTokenizer.from_pretrained(str(Path(path) / "tokenizer"))
        self.m = build_model(fields, sd, self.tok)
        self.ref = int(fields["loop_times"])
        self.info = {k: fields.get(k) for k in ("loop_times", "huginn_sampling_scheme", "huginn_backprop_depth",
                                                "huginn_depth_prior", "model_dim")}
        self.head = self.m.output.output
        self.cap = {}
        self.head.register_forward_pre_hook(lambda mod, a: self.cap.__setitem__("z", a[0]))
        self.batched = False
        self.bos = [self.tok.bos_token_id]  # training documents were encoded with BOS

    @torch.no_grad()
    def readout_inputs(self, ids, depths, doc_id, want_logits=False):
        out, lg = {}, None
        for r in depths:
            self.cap.clear()
            lg, _ = self.m.terminal_kv_forward(ids[None].to(dev), torch.tensor([doc_id]), 42,
                                               full_logits=True, depth=r)
            out[r] = self.cap["z"][0]
        return (out, lg[0].float()) if want_logits else out


class OuroRunner:
    def __init__(self, path):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        # Ouro's remote code targets transformers 4.55; run it with that version (PYTHONPATH)
        self.m = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
                                                      torch_dtype=torch.bfloat16).to(dev).eval()
        self.ref = int(self.m.config.total_ut_steps)
        self.info = {"total_ut_steps": self.ref, "hidden_size": self.m.config.hidden_size}
        self.head = self.m.lm_head
        self.batched = True
        self.bos = None  # tokenizer default, as for the public dense models

    @torch.no_grad()
    def readout_inputs(self, ids, depths, doc_id, want_logits=False):
        # loop r's readout input is the normalised state after r loops; it does not depend on later loops
        self.m.model.total_ut_steps = max(depths)
        _, hs, _ = self.m.model(input_ids=ids[None].to(dev), use_cache=False)
        out = {r: hs[r - 1][0] for r in depths}
        if want_logits:
            self.m.model.total_ut_steps = self.ref
            lg = self.m(input_ids=ids[None].to(dev), use_cache=False, exit_at_step=self.ref - 1).logits[0].float()
            return out, lg
        return out


def collect(run, toks, depths):
    Z = {r: [] for r in depths}
    Y, D = [], []
    for di, (ids, s) in enumerate(toks):
        if ids.numel() - s - 1 <= 0:
            continue
        z = run.readout_inputs(ids, depths, di)
        for r in depths:
            Z[r].append(z[r][s:-1].to(torch.bfloat16).cpu())
        Y.append(ids[s + 1:])
        D.append(torch.full((ids.numel() - s - 1,), di))
    return {r: torch.cat(v) for r, v in Z.items()}, torch.cat(Y), torch.cat(D)


def per_doc(nll, D, n):
    d = D.numpy()
    return np.bincount(d, weights=nll, minlength=n), np.bincount(d, minlength=n).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["ifm", "ouro"])
    ap.add_argument("--path", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--xllm", default=None)
    ap.add_argument("--stubs", default="", help="colon-separated dirs with xllm_extension stand-ins and extra deps")
    ap.add_argument("--depths", default=None, help="comma list; default IFM 1..8,10 / Ouro 1..8")
    ap.add_argument("--n_docs", type=int, default=200)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--aff_tokens", type=int, default=12000)
    ap.add_argument("--nb", type=int, default=2000)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    run = IFMRunner(args.path, args.xllm, [p for p in args.stubs.split(":") if p]) if args.family == "ifm" else OuroRunner(args.path)
    tok, ref = run.tok, run.ref
    depths = ([int(x) for x in args.depths.split(",")] if args.depths else
              ([1, 2, 3, 4, 5, 6, 7, 8, 10] if args.family == "ifm" else list(range(1, 9))))
    assert ref in depths
    if run.bos is not None:
        bos = run.bos
    else:
        first = tok("a", add_special_tokens=True).input_ids[0]
        bos = [first] if tok.bos_token_id is not None and first == tok.bos_token_id else []
    out_dir = HERE / "out" / "looped" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"tag": args.tag, "family": args.family, "path": Path(args.path).name, "ref_depth": ref,
           "depths": depths, "model_info": run.info, "bos": bos, "domains": {}}

    # sanity: captured readout input reproduces the model's own logits at the training depth
    ids0, s0 = DP.tokenize(tok, DP.load_split("math", "eval")[:1], args.max_len, bos)[0]
    z0, lg0 = run.readout_inputs(ids0, [ref], 0, want_logits=True)
    res["sanity_max_abs_logit_diff"] = float((DP.head_logits(run.head, z0[ref]) - lg0).abs().max())
    print("sanity", res["sanity_max_abs_logit_diff"], flush=True)

    t0 = time.time()
    rng = np.random.default_rng(0)
    for dom in DOMAINS:
        te = DP.tokenize(tok, DP.load_split(dom, "eval")[:args.n_docs], args.max_len, bos)
        tc = DP.tokenize(tok, DP.load_split(dom, "cal")[:args.n_docs], args.max_len, bos)
        n_e = len(te)
        groups = [depths] if run.batched else [[r] for r in depths]
        raw, cal, aff, T, lam = {}, {}, {}, {}, {}
        for g in groups:
            Ze, Ye, De = collect(run, te, g)
            Zc, Yc, _ = collect(run, tc, g)
            for r in g:
                zc, yc = Zc[r].to(dev), Yc.to(dev)
                T[r] = DP.fit_T(run.head, zc, yc)
                na = min(args.aff_tokens, yc.numel())
                a, b, lam[r] = DP.fit_affine(run.head, zc[:na], yc[:na])
                ze, ye = Ze[r].to(dev), Ye.to(dev)
                raw[r] = per_doc(DP.token_stats(run.head, ze, ye)["nll"], De, n_e)
                cal[r] = per_doc(DP.token_stats(run.head, ze, ye, T=T[r])["nll"], De, n_e)
                aff[r] = per_doc(DP.token_stats(run.head, ze, ye, a=a, b=b)["nll"], De, n_e)
                del zc, ze
                print(f"{args.tag} {dom} r={r} T={T[r]:.3f} raw={raw[r][0].sum() / raw[r][1].sum():.4f} "
                      f"[{time.time() - t0:.0f}s]", flush=True)
            del Ze, Zc
            torch.cuda.empty_cache()
        np.savez_compressed(out_dir / f"perdoc_{dom}.npz",
                            **{f"{k}_r{r}_{w}": v[r][i] for k, v in (("raw", raw), ("cal", cal), ("aff", aff))
                               for r in depths for i, w in enumerate(("sum", "count"))})
        idx = np.vstack([np.arange(n_e)[None], rng.integers(0, n_e, size=(args.nb, n_e))])
        Q = lambda vc: vc[0][idx].sum(1) / vc[1][idx].sum(1)
        ci = lambda x: [float(np.percentile(x[1:], 2.5)), float(np.percentile(x[1:], 97.5))]
        dres = {"T": {str(r): T[r] for r in depths}, "affine_lambda": {str(r): lam[r] for r in depths},
                "nll_raw": {str(r): float(Q(raw[r])[0]) for r in depths},
                "nll_cal": {str(r): float(Q(cal[r])[0]) for r in depths},
                "nll_aff": {str(r): float(Q(aff[r])[0]) for r in depths}, "vs_ref": {}}
        for r in depths:
            if r == ref:
                continue
            gr = Q(raw[r]) - Q(raw[ref])
            gc = Q(cal[r]) - Q(cal[ref])
            ga = Q(aff[r]) - Q(aff[ref])
            dres["vs_ref"][str(r)] = {
                "gap_raw": float(gr[0]), "gap_raw_ci95": ci(gr),
                "temp_nats": float((gr - gc)[0]), "temp_nats_ci95": ci(gr - gc),
                "temp_share": float(((gr - gc) / gr)[0]), "temp_share_ci95": ci((gr - gc) / gr),
                "affine_share": float(((gr - ga) / gr)[0]), "affine_share_ci95": ci((gr - ga) / gr)}
        res["domains"][dom] = dres
        (out_dir / "summary.json").write_text(json.dumps(res, indent=1))

    # predictions of PREDICTIONS_looped.md, evaluated on mathematical text
    m = res["domains"]["math"]
    Tm = {int(k): v for k, v in m["T"].items()}
    share1 = m["vs_ref"]["1"]["temp_share"]
    inside = [r for r in depths if r <= ref]
    drift = max(Tm[r] for r in inside) - min(Tm[r] for r in inside)
    outside = [r for r in depths if ref < r <= 8]
    res["prediction_inputs"] = {
        "share_r1": share1, "drift_inside": drift,
        "max_abs_T_minus_1_below_ref": max(abs(Tm[r] - 1) for r in inside if r < ref),
        "abs_T_ref_minus_1": abs(Tm[ref] - 1),
        "max_abs_T_minus_1_outside": max(abs(Tm[r] - 1) for r in outside) if outside else None,
        "max_abs_T_minus_1_inside": max(abs(Tm[r] - 1) for r in inside)}
    (out_dir / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["prediction_inputs"], indent=1), "done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
