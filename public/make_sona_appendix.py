#!/usr/bin/env python3
"""make_sona_appendix.py — LaTeX tables for the public-checkpoint appendix, from out/public-sft/sona_dcp.json."""
import json
from pathlib import Path

HERE = Path(__file__).parent
R = json.loads((HERE / "out" / "public-sft" / "sona_dcp.json").read_text())


def cell(c, k, base_ci=True):
    v = c[k]
    s = f"${v['nats']:+.3f}$"
    if "share" in v:
        s += f" & ${v['share']*100:.1f}\\%$ {{\\tiny[{v['share_ci'][0]*100:.1f}, {v['share_ci'][1]*100:.1f}]}}"
    return s


rows = []
for label, k, kT in (("Application count", "apply", "apply_Tcal"),
                     ("Distinct iterations", "distinct", "distinct_Tcal"),
                     ("Inter-block composition", "compose", "compose_Tcal")):
    line = label
    for dom in ("math", "web"):
        c = R[dom]["components"]
        line += f" & {cell(c, k)} & ${c[kT]['share']*100:.1f}\\%$"
    rows.append(line + " \\\\")
extra = []
for label, k in (("Temperature-correctable", "calib_T"), ("Affine-readout-correctable", "calib_aff")):
    line = label
    for dom in ("math", "web"):
        line += f" & {cell(R[dom]['components'], k)} & --"
    extra.append(line + " \\\\")
nm, nw = R["math"]["components"], R["web"]["components"]
t1 = "\n".join([
    "\\begin{table}[t]", "\\centering", "\\scriptsize\\setlength{\\tabcolsep}{3pt}",
    "\\begin{tabular}{lrrrrrr}", "\\toprule",
    "& \\multicolumn{3}{c}{Mathematical text} & \\multicolumn{3}{c}{General text} \\\\",
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
    "Component & nats & share & share, $T$-cal. & nats & share & share, $T$-cal. \\\\", "\\midrule",
    f"Naive gap, prefix(1) vs.\\ prefix(8) & ${nm['naive']['nats']:.3f}$ & $100\\%$ & -- & ${nw['naive']['nats']:.3f}$ & $100\\%$ & -- \\\\",
    "\\midrule", *rows, "\\midrule", *extra, "\\bottomrule", "\\end{tabular}",
    "\\caption{DCP on a public SFT checkpoint of our architecture (step $7{,}000$), with the disjoint calibration and "
    "evaluation documents of Appendix~\\ref{app:public}. Shares with $95\\%$ paired document-bootstrap intervals "
    "($5{,}000$ replicates). \\emph{Share, $T$-cal.}: the same univariate path computed on temperature-calibrated NLL "
    "(each configuration with its own fitted temperature), which measures the interaction between the calibration and "
    "structural axes of Property~2.}",
    "\\label{tab:sonapub-decomp}", "\\end{table}"])

p = R["math"]["profile"]
g_rows = []
for cfg, lab in (("prefix1", "prefix(1)"), ("repeat1", "repeat(1)"), ("suffix1", "suffix(1)"),
                 ("prefix4", "prefix(4)"), ("repeat4", "repeat(4)"), ("suffix4", "suffix(4)"),
                 ("permute8", "permuted indices (8)"), ("full", "full (8)")):
    q = p[cfg]
    g_rows.append(f"{lab} & ${q['nll']:.3f}$ & ${q['T']:.3f}$ & ${q['ece']:.3f} / {q['ece_T']:.3f}$ & "
                  f"${q['z_cos']:.2f}$ & ${q['h_cka']:.2f}$ & ${q['h_norm_ratio']:.2f}$ & ${q['h_frechet']:.2f}$ \\\\")
t2 = "\n".join([
    "\\begin{table}[t]", "\\centering", "\\scriptsize\\setlength{\\tabcolsep}{3pt}",
    "\\begin{tabular}{lrrrrrrr}", "\\toprule",
    "Schedule & NLL & $T$ & ECE raw / $T$ & $\\cos(z, z_{\\text{full}})$ & CKA$(h, h_{\\text{full}})$ & "
    "$\\|h\\| / \\|h_{\\text{full}}\\|$ & Fr\\'echet \\\\", "\\midrule", *g_rows, "\\bottomrule", "\\end{tabular}",
    "\\caption{Readout-side measurements for each schedule on the public SFT checkpoint (mathematical text). "
    "$z$: readout input; $h$: residual stream before the final normalization; reference: the full schedule on the same "
    "tokens. The permuted schedule runs the full eight applications with the global iteration indices permuted "
    "within each block.}",
    "\\label{tab:sonapub-geometry}", "\\end{table}"])

un = {k: p[k]["update_norms"] for k in ("full", "repeat1", "repeat4")}
t3 = "\n".join([
    "\\begin{table}[t]", "\\centering", "\\scriptsize",
    "\\begin{tabular}{lrrrrrrrr}", "\\toprule",
    "Application & $1$ & $2$ & $3$ & $4$ & $5$ & $6$ & $7$ & $8$ \\\\", "\\midrule",
    *[f"{lab} & " + " & ".join(f"${x:.2f}$" for x in un[k]) + " \\\\"
      for k, lab in (("full", "full"), ("repeat1", "repeat(1)"), ("repeat4", "repeat(4)"))],
    "\\bottomrule", "\\end{tabular}",
    "\\caption{Relative update $\\|x^{(j)} - x^{(j-1)}\\| / \\|x^{(j-1)}\\|$ of the real-token stream at each block "
    "application (mean over $20$ mathematical documents). Under repeat schedules the update of the repeated application "
    "shrinks towards a fixed point, whereas in the full schedule the second block's updates grow.}",
    "\\label{tab:sonapub-dyn}", "\\end{table}"])

(HERE / "out" / "public-sft" / "appendix_sona.tex").write_text("\n\n".join([t1, t2, t3]) + "\n")
print("\n\n".join([t1, t2, t3]))
