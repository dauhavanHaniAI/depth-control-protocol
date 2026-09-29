#!/usr/bin/env python3
"""
analyze.py — offline analysis of dcp_public.py outputs (no model runs).

For every model and domain it computes, with a paired document bootstrap (documents resampled once
per replicate and the same resample used for every configuration and calibrator):
  naive gap, application-count contrast (dense), temperature-correctable NLL under four calibrators
  (same-domain T, cross-domain T, in-sample T, affine readout map), in nats and as shares;
plus temperature profiles, calibration metrics, residual geometry, difficulty terciles and
token-position buckets. Writes out/analysis.json and out/tables.tex.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "out"
DOMAINS = ("math", "web")
CALS = ("raw", "T", "Tx", "Tin", "aff")
NBOOT = 2000
ORDER = ["huginn-0125", "Qwen2.5-Math-1.5B", "Qwen3-1.7B", "Qwen2.5-1.5B", "Qwen2.5-1.5B-Instruct",
         "SmolLM2-1.7B", "pythia-1.4b", "opt-350m"]
PRETTY = {"huginn-0125": "Huginn-0125 (recurrent, sampled)", "Qwen2.5-Math-1.5B": "Qwen2.5-Math-1.5B",
          "Qwen3-1.7B": "Qwen3-1.7B", "Qwen2.5-1.5B": "Qwen2.5-1.5B (base)",
          "Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B-Instruct", "SmolLM2-1.7B": "SmolLM2-1.7B",
          "pythia-1.4b": "Pythia-1.4B", "opt-350m": "OPT-350m (post-LN)"}


def doc_sums(z, dom, key):
    """Per-document summed value of per-token array `key`, and token counts."""
    d = z[f"{dom}_doc"]
    n = int(d.max()) + 1
    v = np.bincount(d, weights=z[key].astype(np.float64), minlength=n)
    c = np.bincount(d, minlength=n).astype(np.float64)
    return v, c


class Model:
    def __init__(self, path):
        self.path = path
        self.tag = path.name
        self.S = json.loads((path / "summary.json").read_text())
        self.kind = self.S["kind"]
        self.names = [n for n in self.S["configs"] if (path / f"{n}.npz").exists()]
        self._cache = {}

    def npz(self, name):
        if name not in self._cache:
            self._cache[name] = dict(np.load(self.path / f"{name}.npz"))
        return self._cache[name]

    def ref(self):
        return "r32" if self.kind == "huginn" else "full"

    def mild(self):
        """Least truncated prefix configuration (dense) / r16 (huginn)."""
        if self.kind == "huginn":
            return "r16" if "r16" in self.names else None
        ks = sorted(int(n[6:]) for n in self.names if n.startswith("prefix"))
        return f"prefix{ks[-1]}" if ks else None

    def low(self):
        """Most truncated configuration, used for the headline gap."""
        if self.kind == "huginn":
            return "r1"
        ks = sorted(int(n[6:]) for n in self.names if n.startswith("prefix"))
        return f"prefix{ks[0]}" if ks else None


def boot_components(M, dom, low, rng, ref=None):
    """Paired document bootstrap of all headline components for one model/domain."""
    ref = ref or M.ref()
    confs = [ref, low]
    rep = low.replace("prefix", "repeat") if low.startswith("prefix") else None
    if rep and rep in M.names:
        confs.append(rep)
    sums, cnt = {}, None
    for c in confs:
        z = M.npz(c)
        for cal in CALS:
            key = f"{dom}_{cal}_nll"
            if key in z:
                v, cnt = doc_sums(z, dom, key)
                sums[(c, cal)] = v
    n = len(cnt)
    idx = np.vstack([np.arange(n)[None], rng.integers(0, n, size=(NBOOT, n))])  # row 0 = point estimate
    C = cnt[idx].sum(1)

    def nll(c, cal):
        return sums[(c, cal)][idx].sum(1) / C

    naive = nll(low, "raw") - nll(ref, "raw")
    comp = {"naive": naive}
    for cal in ("T", "Tx", "Tin", "aff"):
        if (low, cal) in sums:
            comp[f"calib_{cal}"] = naive - (nll(low, cal) - nll(ref, cal))
    if rep and (rep, "raw") in sums:
        comp["apply"] = nll(low, "raw") - nll(rep, "raw")
    res = {}
    for k, v in comp.items():
        res[k] = {"nats": float(v[0]), "ci": [float(np.percentile(v[1:], 2.5)), float(np.percentile(v[1:], 97.5))]}
        if k != "naive":
            sh = v / naive
            res[k]["share"] = float(sh[0])
            res[k]["share_ci"] = [float(np.percentile(sh[1:], 2.5)), float(np.percentile(sh[1:], 97.5))]
    return res


def ece_from(z, dom, cal, bins=15):
    conf = z[f"{dom}_{cal}_conf"].astype(np.float64)
    cor = z[f"{dom}_{cal}_correct"].astype(np.float64)
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - cor[m].mean())
    return float(e)


def difficulty_and_position(M, dom, low):
    """Gap by full-depth difficulty tercile (documents) and by token-position bucket."""
    zr, zl = M.npz(M.ref()), M.npz(low)
    v_ref, c = doc_sums(zr, dom, f"{dom}_raw_nll")
    v_low, _ = doc_sums(zl, dom, f"{dom}_raw_nll")
    vT_ref, _ = doc_sums(zr, dom, f"{dom}_T_nll")
    vT_low, _ = doc_sums(zl, dom, f"{dom}_T_nll")
    ok = c > 0  # documents with no scored tokens (e.g. prompt fills max_len) are excluded
    diff = np.where(ok, v_ref / np.maximum(c, 1), np.nan)
    terc = np.digitize(diff, np.nanquantile(diff, [1 / 3, 2 / 3]))
    terc[~ok] = -1
    out = {"tercile": [], "position": []}
    for t in range(3):
        m = terc == t
        gap = (v_low[m].sum() - v_ref[m].sum()) / c[m].sum()
        gapT = (vT_low[m].sum() - vT_ref[m].sum()) / c[m].sum()
        out["tercile"].append({"tercile": t, "full_nll": float(v_ref[m].sum() / c[m].sum()), "gap": float(gap),
                               "calib_share": float((gap - gapT) / gap) if gap else None})
    pos = zr[f"{dom}_pos"]
    for lo, hi in ((0, 64), (64, 256), (256, 512), (512, 10 ** 6)):
        m = (pos >= lo) & (pos < hi)
        if m.sum() < 100:
            continue
        g = zl[f"{dom}_raw_nll"][m].mean() - zr[f"{dom}_raw_nll"][m].mean()
        gT = zl[f"{dom}_T_nll"][m].mean() - zr[f"{dom}_T_nll"][m].mean()
        out["position"].append({"bucket": f"{lo}-{hi if hi < 10 ** 6 else 'end'}", "n": int(m.sum()),
                                "gap": float(g), "calib_share": float((g - gT) / g) if g else None})
    return out


def analyze_model(M, rng):
    low = M.low()
    if low is None or low not in M.names or M.ref() not in M.names:
        return None
    rec = {"kind": M.kind, "low": low, "ref": M.ref(), "n_configs": len(M.names), "domains": {}}
    for dom in DOMAINS:
        r = {"components": boot_components(M, dom, low, rng)}
        mild = M.mild()
        if mild and mild != low:
            r["components_mild"] = boot_components(M, dom, mild, rng)
            r["mild"] = mild
        if M.kind == "huginn":  # extrapolation beyond the training mean, relative to r32
            r["extrapolation"] = {n: boot_components(M, dom, n, rng)["naive"]
                                  for n in ("r48", "r64") if n in M.names}
        prof = {}
        for n in M.names:
            c = M.S["configs"][n]
            z = M.npz(n)
            d = c["domains"][dom]
            prof[n] = {"T": c["T"][dom], "T_insample": c["T_insample"][dom], "nll": d["nll"],
                       "nll_T": d["nll_T"], "nll_T_cross": d["nll_T_cross"], "nll_affine": d["nll_affine"],
                       "acc": d["acc"], "ece": ece_from(z, dom, "raw"), "ece_T": ece_from(z, dom, "T"),
                       "ece_affine": ece_from(z, dom, "aff"), "brier": d["brier"], "brier_T": d["brier_T"],
                       "entropy": d["entropy"], "conf": d["conf"], "logit_norm": d["logit_norm"],
                       "geometry": d["geometry"], "update_norms": c.get("update_norms"),
                       "affine_lambda": c.get("affine_lambda", {}).get(dom)}
        r["profile"] = prof
        r["difficulty_position"] = difficulty_and_position(M, dom, low)
        rec["domains"][dom] = r
    return rec


def fmt_share(v):
    return f"${v['share']*100:.1f}$ \\scriptsize{{[{v['share_ci'][0]*100:.1f}, {v['share_ci'][1]*100:.1f}]}}"


def write_tables(A):
    L = []
    for dom in DOMAINS:
        L += [f"% Temperature-correctable share (%) under four calibrators, {dom} domain; 95% paired doc-bootstrap CIs",
              "\\begin{tabular}{llrrrrrr}", "\\toprule",
              "Model & Truncated & Naive gap & Same-dom.\\ $T$ & Cross-dom.\\ $T$ & In-sample $T$ & Affine & Application \\\\",
              "\\midrule"]
        for tag, r in A.items():
            c = r["domains"][dom]["components"]
            cells = [fmt_share(c[k]) if k in c else "--" for k in ("calib_T", "calib_Tx", "calib_Tin", "calib_aff", "apply")]
            L.append(f"{PRETTY.get(tag, tag)} & {r['low']} & ${c['naive']['nats']:.3f}$ & " + " & ".join(cells) + " \\\\")
        L += ["\\bottomrule", "\\end{tabular}", ""]
    L += ["% Temperature range, calibration and readout-input geometry at the truncated configuration (math)",
          "\\begin{tabular}{lrrrrrrr}", "\\toprule",
          "Model & $T$ range & ECE raw $\\to$ $T$ & $\\cos(z, z_{\\text{ref}})$ & CKA$(h, h_{\\text{ref}})$ & "
          "$\\|h\\| / \\|h_{\\text{ref}}\\|$ & Eff.\\ rank ratio & Fr\\'echet \\\\", "\\midrule"]
    for tag, r in A.items():
        p = r["domains"]["math"]["profile"]
        Ts = [v["T"] for n, v in p.items() if v["T"] < 999 and not n.startswith(("repeat", "suffix"))]
        q = p[r["low"]]
        g = q["geometry"]
        L.append(f"{PRETTY.get(tag, tag)} & ${min(Ts):.2f}$--${max(Ts):.2f}$ & ${q['ece']:.3f} \\to {q['ece_T']:.3f}$ & "
                 f"${g['z_cos_to_full']:.2f}$ & ${g['h_cka_to_full']:.2f}$ & ${g['h_norm']/g['h_norm_ref']:.2f}$ & "
                 f"${g['h_eff_rank']/g['h_eff_rank_ref']:.2f}$ & ${g['h_frechet_norm']:.2f}$ \\\\")
    L += ["\\bottomrule", "\\end{tabular}", ""]
    (OUT / "tables.tex").write_text("\n".join(L))


def main():
    rng = np.random.default_rng(0)
    models = [Model(p) for p in sorted(OUT.iterdir()) if (p / "summary.json").exists()]
    models.sort(key=lambda m: ORDER.index(m.tag) if m.tag in ORDER else 99)
    A = {}
    for M in models:
        rec = analyze_model(M, rng)
        if rec is None:
            print(f"{M.tag}: incomplete ({len(M.names)} configs), skipped")
            continue
        A[M.tag] = rec
        for dom in DOMAINS:
            c = rec["domains"][dom]["components"]
            print(f"{M.tag:22s} {dom:4s} {rec['low']:9s} naive={c['naive']['nats']:.3f}  "
                  + "  ".join(f"{k}={v['share']*100:.1f}% [{v['share_ci'][0]*100:.1f}, {v['share_ci'][1]*100:.1f}]"
                              for k, v in c.items() if k != "naive"))
    if not A:
        print("no per-token outputs (*.npz) found under out/: run dcp_public.py first; "
              "existing out/analysis.json left unchanged")
        return
    (OUT / "analysis.json").write_text(json.dumps(A, indent=1))
    write_tables(A)
    print("wrote", OUT / "analysis.json", OUT / "tables.tex")


if __name__ == "__main__":
    main()
