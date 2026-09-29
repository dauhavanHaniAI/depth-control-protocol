#!/usr/bin/env python3
"""make_appendix.py — LaTeX tables for the public-model appendix, generated from out/analysis.json."""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
A = json.loads((HERE / "out" / "analysis.json").read_text())
NAME = {"huginn-0125": "Huginn-0125$^{\\ast}$", "Qwen2.5-Math-1.5B": "Qwen2.5-Math-1.5B",
        "Qwen3-1.7B": "Qwen3-1.7B", "Qwen2.5-1.5B": "Qwen2.5-1.5B", "Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B-Instruct",
        "SmolLM2-1.7B": "SmolLM2-1.7B", "pythia-1.4b": "Pythia-1.4B", "opt-350m": "OPT-350m (post-LN)"}


def depth_of(name):
    m = re.fullmatch(r"(?:prefix|r)(\d+)", name)
    return int(m.group(1)) if m else None


def trunc_label(r, name):
    if r["kind"] == "huginn":
        return f"$r={depth_of(name)}$"
    kmax = max(depth_of(n) for n in r["domains"]["math"]["profile"] if n.startswith("prefix"))
    L = round(kmax / 0.93)
    return f"${depth_of(name)}/{L}$"


def sh(c, k, digits=1):
    if k not in c:
        return "--"
    v = c[k]
    return f"${v['share']*100:.{digits}f}$ {{\\tiny[{v['share_ci'][0]*100:.{digits}f}, {v['share_ci'][1]*100:.{digits}f}]}}"


def components_table(dom):
    rows = []
    for panel, key in (("Severe truncation ($k \\approx L/4$; Huginn $r = 1$ vs.\\ $32$)", "components"),
                       ("Mild truncation ($k \\approx 0.93L$)", "components_mild")):
        rows.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{panel}}}}} \\\\")
        for t, r in A.items():
            d = r["domains"][dom]
            c = d.get(key)
            if c is None:
                continue
            cfg = r["low"] if key == "components" else d["mild"]
            if r["kind"] == "huginn" and key == "components_mild":
                continue  # r16 vs r32 differ by < 0.005 nats: shares are undefined in practice
            rows.append(f"{NAME[t]} & {trunc_label(r, cfg)} & ${c['naive']['nats']:.3f}$ & {sh(c,'calib_T')} & "
                        f"{sh(c,'calib_Tx')} & {sh(c,'calib_Tin')} & {sh(c,'calib_aff')} & {sh(c,'apply')} \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    return "\n".join([
        "\\begin{table}[t]", "\\centering", "\\scriptsize\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llrrrrrr}", "\\toprule",
        "Model & $k$ & Gap (nats) & Same-dom.\\ $T$ & Cross-dom.\\ $T$ & In-sample $T$ & Affine readout & Application \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}",
        f"\\caption{{Public models, {'mathematical (MATH solutions)' if dom == 'math' else 'general (FineWeb)'} text. "
        "Temperature-correctable share (\\%) of the naive gap under four recalibrations of the truncated configuration, "
        "and the application-count contrast (dense models, prefix($k$) vs.\\ repeat($k$)). "
        "Calibration maps are fitted on $200$ calibration documents disjoint from the $200$ evaluation documents "
        "(cross-domain: fitted on the other domain; in-sample: fitted on the evaluation documents). "
        "$95\\%$ paired document-bootstrap intervals, $2{,}000$ replicates. "
        "$^{\\ast}$Recurrent, trained with sampled depth; at $r = 16$ its gap to $r = 32$ is below $0.005$~nats, so no mild row is shown.}",
        f"\\label{{tab:public-{dom}}}", "\\end{table}"])


def geometry_table():
    rows = []
    for t, r in A.items():
        p = r["domains"]["math"]["profile"]
        Ts = [v["T"] for n, v in p.items() if (n == "full" or re.fullmatch(r"(prefix|r)\d+", n)) and v["T"] < 999]
        q = p[r["low"]]
        g = q["geometry"]
        rep = r["low"].replace("prefix", "repeat")
        gr = p[rep]["geometry"] if rep in p else None
        hn = "--" if r["kind"] == "huginn" else f"${g['h_norm']/g['h_norm_ref']:.2f}$"
        hr = f"${gr['h_norm']/gr['h_norm_ref']:.2f}$" if gr and r["kind"] != "huginn" else "--"
        rows.append(f"{NAME[t]} & ${min(Ts):.2f}$--${max(Ts):.2f}$ & ${q['ece']:.3f} / {q['ece_T']:.3f} / {q['ece_affine']:.3f}$ & "
                    f"${g['z_cos_to_full']:.2f}$ & ${g['h_cka_to_full']:.2f}$ & {hn} & {hr} & ${g['h_frechet_norm']:.2f}$ \\\\")
    return "\n".join([
        "\\begin{table}[t]", "\\centering", "\\scriptsize\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{lrrrrrrr}", "\\toprule",
        "Model & $T$ range & ECE raw / $T$ / affine & $\\cos(z, z_{\\text{ref}})$ & CKA$(h, h_{\\text{ref}})$ & "
        "$\\|h\\|/\\|h_{\\text{ref}}\\|$ prefix & $\\|h\\|/\\|h_{\\text{ref}}\\|$ repeat & Fr\\'echet \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}",
        "\\caption{Readout-side measurements at the severe truncation of Table~\\ref{tab:public-math} (mathematical text). "
        "$T$ range: fitted temperature over all prefix depths and full depth. ECE: $15$-bin expected calibration error of the top-1 "
        "prediction before and after recalibration. $z$: input to the readout (after the final normalization); $h$: residual stream "
        "before the final normalization; the reference is the full-depth model (Huginn: $r = 32$) on the same tokens. "
        "Fr\\'echet: distance between Gaussian fits of $h$ and $h_{\\text{ref}}$ in the top-$64$ principal subspace of $h_{\\text{ref}}$, "
        "normalized by the reference total variance. Huginn's residual norm is fixed by its sandwich normalization, and OPT-350m is post-LN, "
        "so norm ratios are uninformative for them.}",
        "\\label{tab:public-geometry}", "\\end{table}"])


def huginn_table():
    r = A["huginn-0125"]
    pm, pw = r["domains"]["math"]["profile"], r["domains"]["web"]["profile"]
    rows = []
    for n in sorted(pm, key=depth_of):
        a, w = pm[n], pw[n]
        g = a["geometry"]
        rows.append(f"${depth_of(n)}$ & ${a['nll']:.4f}$ & ${a['nll_T']:.4f}$ & ${a['nll_affine']:.4f}$ & ${a['T']:.3f}$ & "
                    f"${w['T']:.3f}$ & ${a['ece']:.3f}$ & ${g['z_cos_to_full']:.2f}$ & ${g['h_cka_to_full']:.2f}$ \\\\")
    return "\n".join([
        "\\begin{table}[t]", "\\centering", "\\scriptsize",
        "\\begin{tabular}{rrrrrrrrr}", "\\toprule",
        "$r$ & NLL & NLL, $T$ & NLL, affine & $T$ (math) & $T$ (web) & ECE & $\\cos(z, z_{32})$ & CKA$(h, h_{32})$ \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}",
        "\\caption{Huginn-0125 across recurrence depths, including extrapolation beyond its training mean ($r = 48, 64$). "
        "NLL on the $200$ mathematical evaluation documents with disjoint calibration documents. Recalibration changes NLL by at most "
        "$0.103$~nats at any depth, and the temperature stays within $0.95$--$1.07$ on both domains.}",
        "\\label{tab:public-huginn}", "\\end{table}"])


out = "\n\n".join([components_table("math"), components_table("web"), geometry_table(), huginn_table()])
(HERE / "out" / "appendix_tables.tex").write_text(out + "\n")
print(out[:3000])
