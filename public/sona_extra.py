#!/usr/bin/env python3
"""sona_extra.py — offline analyses on the public checkpoint of our architecture (no model runs).

  * TOST equivalence test for the distinct-iteration contrast (90% paired-bootstrap CI inside +-delta);
  * order sensitivity / Shapley over the two feasible orders of the calibration and structural axes;
  * stratification of every contrast by document difficulty, document length and token position.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
D = HERE / "out" / "public-sft"
NB = 5000
DELTAS = (0.02, 0.05)  # nats; see paper for justification
CFGS = ("full", "prefix1", "repeat1", "repeat4")
rng = np.random.default_rng(0)
out = {}
for dom in ("math", "web"):
    Z = {c: np.load(D / f"{c}.npz") for c in CFGS}
    doc = Z["full"][f"{dom}_doc"]
    pos = Z["full"][f"{dom}_pos"]
    n = int(doc.max()) + 1
    cnt = np.bincount(doc, minlength=n).astype(float)

    def dsum(c, cal, mask=None):
        w = Z[c][f"{dom}_{cal}_nll"].astype(float)
        if mask is not None:
            w = w * mask
        return np.bincount(doc, weights=w, minlength=n)

    def contrasts(mask=None, docs=None):
        """Point estimates (nats) of every contrast on a token mask and/or document subset."""
        m = np.ones_like(doc, dtype=float) if mask is None else mask.astype(float)
        c = np.bincount(doc, weights=m, minlength=n)
        sel = np.ones(n, bool) if docs is None else docs
        Q = {(k, cal): dsum(k, cal, m)[sel].sum() / c[sel].sum() for k in CFGS for cal in ("raw", "T")}
        naive = Q[("prefix1", "raw")] - Q[("full", "raw")]
        return {"naive": naive,
                "apply": Q[("prefix1", "raw")] - Q[("repeat1", "raw")],
                "distinct": Q[("repeat1", "raw")] - Q[("repeat4", "raw")],
                "compose": Q[("repeat4", "raw")] - Q[("full", "raw")],
                "calib_T": naive - (Q[("prefix1", "T")] - Q[("full", "T")]),
                "tokens": int(c[sel].sum())}

    # --- bootstrap of distinct (raw and T-calibrated) and of both orders
    S = {(k, cal): dsum(k, cal) for k in CFGS for cal in ("raw", "T")}
    idx = np.vstack([np.arange(n)[None], rng.integers(0, n, size=(NB, n))])
    C = cnt[idx].sum(1)
    Q = {key: v[idx].sum(1) / C for key, v in S.items()}
    res = {}
    for cal in ("raw", "T"):
        dist = Q[("repeat1", cal)] - Q[("repeat4", cal)]
        lo90, hi90 = np.percentile(dist[1:], [5, 95])
        res[f"distinct_{cal}"] = {"nats": float(dist[0]), "ci90": [float(lo90), float(hi90)],
                                  "tost": {str(d): bool(lo90 > -d and hi90 < d) for d in DELTAS}}
    # Shapley over the two feasible orders: structure-then-calibration (raw path) and
    # calibration-then-structure (T-calibrated path); each component's value is the mean of both
    for comp, (a, b) in {"apply": ("prefix1", "repeat1"), "distinct": ("repeat1", "repeat4"),
                         "compose": ("repeat4", "full")}.items():
        raw = Q[(a, "raw")] - Q[(b, "raw")]
        cal = Q[(a, "T")] - Q[(b, "T")]
        sh = 0.5 * (raw + cal)
        res[f"shapley_{comp}"] = {"raw_order": float(raw[0]), "cal_order": float(cal[0]), "shapley": float(sh[0]),
                                  "shapley_ci95": [float(np.percentile(sh[1:], 2.5)), float(np.percentile(sh[1:], 97.5))],
                                  "order_diff": float(raw[0] - cal[0])}
    # --- strata
    full_doc_nll = S[("full", "raw")] / np.maximum(cnt, 1)
    ok = cnt > 0
    q = np.nanquantile(np.where(ok, full_doc_nll, np.nan), [1 / 3, 2 / 3])
    diff_t = np.digitize(full_doc_nll, q)
    ql = np.quantile(cnt[ok], [1 / 3, 2 / 3])
    len_t = np.digitize(cnt, ql)
    strata = {}
    for t, name in enumerate(("easy", "medium", "hard")):
        strata[f"difficulty_{name}"] = contrasts(docs=ok & (diff_t == t))
    for t, name in enumerate(("short", "medium", "long")):
        strata[f"length_{name}"] = contrasts(docs=ok & (len_t == t))
    for lo, hi in ((0, 64), (64, 256), (256, 512), (512, 10 ** 6)):
        m = (pos >= lo) & (pos < hi)
        if m.sum() >= 200:
            strata[f"position_{lo}-{hi if hi < 10**6 else 'end'}"] = contrasts(mask=m)
    res["strata"] = strata
    out[dom] = res
(D / "sona_extra.json").write_text(json.dumps(out, indent=1))
for dom, r in out.items():
    print(f"== {dom}")
    for k in ("distinct_raw", "distinct_T"):
        print(f"  {k}: {r[k]['nats']:+.4f}  90% CI [{r[k]['ci90'][0]:+.4f}, {r[k]['ci90'][1]:+.4f}]  TOST {r[k]['tost']}")
    for c in ("apply", "distinct", "compose"):
        s = r[f"shapley_{c}"]
        print(f"  {c:8s} raw-order {s['raw_order']:+.4f}  cal-order {s['cal_order']:+.4f}  Shapley {s['shapley']:+.4f} "
              f"[{s['shapley_ci95'][0]:+.4f}, {s['shapley_ci95'][1]:+.4f}]")
    for k, v in r["strata"].items():
        print(f"  {k:22s} tok={v['tokens']:6d} naive={v['naive']:.3f} apply={v['apply']:+.3f} "
              f"distinct={v['distinct']:+.4f} compose={v['compose']:+.3f} calibT={v['calib_T']:+.3f}")
