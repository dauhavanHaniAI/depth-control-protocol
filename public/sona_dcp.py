#!/usr/bin/env python3
"""sona_dcp.py — full DCP decomposition for the recurrent model from dcp_public.py outputs.

Paired document bootstrap (documents resampled once per replicate, shared by every configuration and
calibrator). Components, each in nats and as a share of the naive gap:
  apply    = prefix(1) - repeat(1)          distinct = repeat(1) - repeat(4)
  compose  = repeat(4) - full               calib_T / calib_aff: temperature / affine-correctable part
The same structural decomposition is repeated on temperature-calibrated NLL (each configuration with its
own fitted T) to measure the interaction between the calibration and structural axes (Property 2).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
TAG = "public-sft"
D = HERE / "out" / TAG
S = json.loads((D / "summary.json").read_text())
NB = 5000
rng = np.random.default_rng(0)
res = {}
for dom in ("math", "web"):
    cache = {}

    def sums(cfg, cal):
        if (cfg, cal) not in cache:
            z = np.load(D / f"{cfg}.npz")
            d = z[f"{dom}_doc"]
            n = int(d.max()) + 1
            cache[(cfg, cal)] = (np.bincount(d, weights=z[f"{dom}_{cal}_nll"].astype(np.float64), minlength=n),
                                 np.bincount(d, minlength=n).astype(np.float64))
        return cache[(cfg, cal)]

    n = len(sums("full", "raw")[1])
    idx = np.vstack([np.arange(n)[None], rng.integers(0, n, size=(NB, n))])

    def Q(cfg, cal="raw"):
        v, c = sums(cfg, cal)
        return v[idx].sum(1) / c[idx].sum(1)

    comp = {}
    for cal, tag in (("raw", ""), ("T", "_Tcal")):
        naive = Q("prefix1", cal) - Q("full", cal)
        comp["naive" + tag] = naive
        comp["apply" + tag] = Q("prefix1", cal) - Q("repeat1", cal)
        comp["distinct" + tag] = Q("repeat1", cal) - Q("repeat4", cal)
        comp["compose" + tag] = Q("repeat4", cal) - Q("full", cal)
    naive = comp["naive"]
    comp["calib_T"] = naive - comp["naive_Tcal"]
    comp["calib_aff"] = naive - (Q("prefix1", "aff") - Q("full", "aff"))
    comp["permute8_minus_full"] = Q("permute8") - Q("full")
    out = {}
    for k, v in comp.items():
        base = comp["naive_Tcal"] if k.endswith("_Tcal") else naive
        rec = {"nats": float(v[0]), "ci": [float(np.percentile(v[1:], 2.5)), float(np.percentile(v[1:], 97.5))]}
        if not k.startswith("naive") and k != "permute8_minus_full":
            sh = v / base
            rec["share"] = float(sh[0])
            rec["share_ci"] = [float(np.percentile(sh[1:], 2.5)), float(np.percentile(sh[1:], 97.5))]
        out[k] = rec
    prof = {}
    for cfg, c in S["configs"].items():
        g = c["domains"][dom]["geometry"]
        prof[cfg] = {"nll": c["domains"][dom]["nll"], "nll_T": c["domains"][dom]["nll_T"],
                     "nll_aff": c["domains"][dom]["nll_affine"], "T": c["T"][dom], "ece": c["domains"][dom]["ece"],
                     "ece_T": c["domains"][dom]["ece_T"], "z_cos": g["z_cos_to_full"], "h_cka": g["h_cka_to_full"],
                     "h_norm_ratio": g["h_norm"] / g["h_norm_ref"], "h_frechet": g["h_frechet_norm"],
                     "update_norms": c.get("update_norms")}
    res[dom] = {"components": out, "profile": prof}
(D / "sona_dcp.json").write_text(json.dumps(res, indent=1))
for dom in ("math", "web"):
    print(f"== {dom}")
    for k, v in res[dom]["components"].items():
        s = f"  share {v['share']*100:6.1f}% [{v['share_ci'][0]*100:.1f}, {v['share_ci'][1]*100:.1f}]" if "share" in v else ""
        print(f"  {k:22s} {v['nats']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]{s}")
p = res["math"]["profile"]
print("== geometry (math): cfg  NLL  T  cos(z)  CKA(h)  |h|/|h_full|  Frechet  ECE->T")
for cfg in ("prefix1", "repeat1", "suffix1", "prefix4", "repeat4", "suffix4", "permute8", "full"):
    q = p[cfg]
    print(f"  {cfg:9s} {q['nll']:.3f} {q['T']:.3f} {q['z_cos']:.2f} {q['h_cka']:.2f} {q['h_norm_ratio']:.2f} {q['h_frechet']:.2f} {q['ece']:.3f}->{q['ece_T']:.3f}")
print("update norms full    :", [round(x, 2) for x in p["full"]["update_norms"]])
print("update norms repeat1 :", [round(x, 2) for x in p["repeat1"]["update_norms"]])
print("update norms repeat4 :", [round(x, 2) for x in p["repeat4"]["update_norms"]])
