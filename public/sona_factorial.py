#!/usr/bin/env python3
"""sona_factorial.py — factorial and calibration-uncertainty analyses on the public checkpoint (math text).

(1) 2x2 factorial on the first block: application count a in {4, 8} x iteration conditioning in
    {one index (g=0 for every application), four indices (g=0..3)}, plus two baselines:
      same4    : (0,0) x4                               a=4, one index
      prefix4  : (0,0)(0,1)(0,2)(0,3)                   a=4, four indices
      repeat1  : (0,0) x8                               a=8, one index
      repeat4  : prefix4 + (0,3) x4                     a=8, four indices
      rfirst8  : (0,0) x5 + (0,1)(0,2)(0,3)             a=8, four indices, extra applications first
      cyclic8  : (0,0)(0,1)(0,2)(0,3) twice             a=8, four indices, indices cycled
    Interaction = [Q(same4) - Q(prefix4)] - [Q(repeat1) - Q(repeat4)], with a paired document bootstrap, on raw
    and on temperature-calibrated NLL (temperatures fitted on the disjoint calibration documents).
(2) Temperature-correctable share of the prefix(1) -> full gap with the temperature REFIT in every bootstrap
    replicate: calibration documents and evaluation documents are both resampled.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import dcp_public as DP
import sona_adapter as S
import sona_c as C

HERE = Path(__file__).parent
OUT = HERE / "out" / "public-sft" / "reviewer_c"
dev = "cuda"
B0 = [(0, t) for t in range(4)]
PLANS = {
    "same4": [(0, 0)] * 4,
    "prefix4": B0,
    "repeat1": [(0, 0)] * 8,
    "repeat4": B0 + [(0, 3)] * 4,
    "rfirst8": [(0, 0)] * 5 + [(0, 1), (0, 2), (0, 3)],
    "cyclic8": B0 + B0,
    "prefix1": [(0, 0)],
    "full": [(b, t) for b in range(2) for t in range(4)],
}


def per_doc(W, Z, Y, D, n_docs, T=1.0):
    nll = C.nll_of(W, Z, Y, T=T).double().numpy()
    d = D.numpy()
    return (np.bincount(d, weights=nll, minlength=n_docs), np.bincount(d, minlength=n_docs).astype(float))


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    tok, m, step = S.load("<checkpoint>.pt", dev)
    ad = S.SonaAdapter(m, tok)
    head = ad.head
    W = m.output.weight.float()
    te, tc = C.load_tokens(tok, "eval"), C.load_tokens(tok, "cal")
    n_e, n_c = len(te), len(tc)
    rng = np.random.default_rng(0)
    res = {"factorial": {}, "calibration_refit_bootstrap": {}}
    feats = {}
    t0 = time.time()
    for name, plan in PLANS.items():
        Ze, _, Ye, De = C.collect(ad, te, {"plan": plan}, want_h=False)
        Zc, _, Yc, Dc = C.collect(ad, tc, {"plan": plan}, want_h=False)
        T = DP.fit_T(head, Zc.to(dev), Yc.to(dev))
        feats[name] = (Ze, Ye, De, Zc, Yc, Dc, T)
        print(f"{name:8s} T={T:.3f} [{time.time() - t0:.0f}s]", flush=True)

    # (1) factorial with paired document bootstrap
    NB = 5000
    idx = np.vstack([np.arange(n_e)[None], rng.integers(0, n_e, size=(NB, n_e))])
    for cal in ("raw", "T"):
        Q = {}
        for name in PLANS:
            Ze, Ye, De, *_, T = feats[name]
            v, c = per_doc(W, Ze, Ye, De, n_e, T=(T if cal == "T" else 1.0))
            Q[name] = v[idx].sum(1) / c[idx].sum(1)
        eff = {
            "conditioning_at_a4": Q["same4"] - Q["prefix4"],
            "conditioning_at_a8": Q["repeat1"] - Q["repeat4"],
            "application_at_1idx": Q["same4"] - Q["repeat1"],
            "application_at_4idx": Q["prefix4"] - Q["repeat4"],
            "interaction": (Q["same4"] - Q["prefix4"]) - (Q["repeat1"] - Q["repeat4"]),
            "rfirst8_minus_repeat4": Q["rfirst8"] - Q["repeat4"],
            "cyclic8_minus_repeat4": Q["cyclic8"] - Q["repeat4"],
        }
        res["factorial"][cal] = {
            "nll": {k: float(v[0]) for k, v in Q.items()},
            "effects": {k: {"nats": float(v[0]), "ci95": [float(np.percentile(v[1:], 2.5)), float(np.percentile(v[1:], 97.5))],
                            "ci90": [float(np.percentile(v[1:], 5)), float(np.percentile(v[1:], 95))]} for k, v in eff.items()},
        }
        print(cal, json.dumps(res["factorial"][cal]["effects"], indent=0)[:1200], flush=True)

    # (2) temperature refit inside the bootstrap (prefix(1) vs full)
    NR = 200
    Zc1, Yc1, Dc1 = feats["prefix1"][3:6]
    Zcf, Ycf, Dcf = feats["full"][3:6]
    raw1 = per_doc(W, *feats["prefix1"][0:3], n_e)
    rawf = per_doc(W, *feats["full"][0:3], n_e)
    shares, T1s, Tfs = [], [], []
    for r in range(NR):
        cdocs = torch.tensor(rng.integers(0, n_c, size=n_c))
        edocs = rng.integers(0, n_e, size=n_e) if r > 0 else np.arange(n_e)
        Ts = []
        for Zc, Yc, Dc in ((Zc1, Yc1, Dc1), (Zcf, Ycf, Dcf)):
            rows = torch.cat([torch.nonzero(Dc == d).flatten() for d in cdocs.tolist()])
            Ts.append(DP.fit_T(head, Zc[rows].to(dev), Yc[rows].to(dev)))
        cal1 = per_doc(W, *feats["prefix1"][0:3], n_e, T=Ts[0])
        calf = per_doc(W, *feats["full"][0:3], n_e, T=Ts[1])
        q = lambda vc: vc[0][edocs].sum() / vc[1][edocs].sum()
        gap = q(raw1) - q(rawf)
        shares.append((gap - (q(cal1) - q(calf))) / gap)
        T1s.append(Ts[0]); Tfs.append(Ts[1])
        if r % 20 == 0:
            print(f"refit bootstrap {r}/{NR} share={shares[-1]:.4f} T1={Ts[0]:.3f} Tf={Ts[1]:.3f}", flush=True)
    sh = np.array(shares)
    res["calibration_refit_bootstrap"] = {
        "replicates": NR, "share_point": float(sh[0]),
        "share_ci95": [float(np.percentile(sh[1:], 2.5)), float(np.percentile(sh[1:], 97.5))],
        "T_prefix1_ci95": [float(np.percentile(T1s[1:], 2.5)), float(np.percentile(T1s[1:], 97.5))],
        "T_full_ci95": [float(np.percentile(Tfs[1:], 2.5)), float(np.percentile(Tfs[1:], 97.5))]}
    print(res["calibration_refit_bootstrap"], flush=True)
    (OUT / "factorial.json").write_text(json.dumps(res, indent=1))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
