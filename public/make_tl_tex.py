#!/usr/bin/env python3
"""make_tl_tex.py — table comparing the diagonal affine probe with full-rank tuned-lens translators (r = 1)."""
import json
from pathlib import Path

HERE = Path(__file__).parent
ROWS = [("ifm-s-fixed-r5", "IFM-S", "fixed"), ("ifm-s-fixed-pln5", "IFM-S", "sampled"),
        ("ifm-m-fixed-r5", "IFM-M", "fixed"), ("ifm-m-fixed-pln5", "IFM-M", "sampled"),
        ("ifm-l-fixed-r5", "IFM-L", "fixed"), ("ifm-l-fixed-pln5", "IFM-L", "sampled"),
        ("ouro-1.4B", "Ouro-1.4B", "per-loop loss"), ("ouro-2.6B", "Ouro-2.6B", "per-loop loss")]


def load(sub, tag):
    f = HERE / "out" / sub / tag / "summary.json"
    return json.loads(f.read_text()) if f.exists() else None


def main():
    lines = []
    for tag, label, sched in ROWS:
        tl, lp = load("tuned_lens", tag), load("looped", tag)
        if tl is None or lp is None:
            continue
        v = tl["vs_ref"]["1"]
        a = lp["domains"]["math"]["vs_ref"]["1"]
        ci = lambda x: f"{{\\tiny[{100 * x[0]:.1f}, {100 * x[1]:.1f}]}}"
        pm = tl["translator_params"] / 1e6
        lines.append(f"{label} & {sched} & ${v['gap_raw']:.3f}$ & ${100 * a['affine_share']:.1f}$ & "
                     f"${100 * v['share_nll_lens']:.1f}$ {ci(v['share_nll_lens_ci95'])} & "
                     f"${100 * v['share_tuned_lens_kl']:.1f}$ {ci(v['share_tuned_lens_kl_ci95'])} & "
                     f"${tl['full_depth_gain_nll_lens']:.3f}$ & ${pm:.1f}$M \\\\")
    tex = "\n".join([
        "\\begin{table}[t]", "\\centering", "\\footnotesize\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llrrrrrr}", "\\toprule",
        "& & & Diagonal & \\multicolumn{2}{c}{Full-rank translator ($d \\times d$)} & & \\\\",
        "\\cmidrule(lr){5-6}",
        "Model & Training depth & Gap $r{=}1$ & affine (\\%) & NLL objective (\\%) & Tuned lens, KL (\\%) & Gain at $R$ & Params \\\\",
        "\\midrule", *lines, "\\bottomrule", "\\end{tabular}",
        "\\caption{Linear recoverability of the $r = 1$ truncation gap under probes of increasing capacity, mathematical "
        "text. Diagonal affine: the probe of Table~\\ref{tab:looped} ($2d$ parameters, fitted on calibration documents). "
        "Full-rank translators $h \\mapsto h + Ah + b$ act on the state entering the model's own final normalization and "
        "readout, are initialized at the identity, and are trained on MATH training solutions (disjoint from the "
        "calibration and evaluation documents; Adam, learning rate $10^{-4}$, $8$ documents per step, $2$ epochs), with "
        "the checkpoint selected on the calibration documents. NLL objective: next-token loss, full-depth controlled "
        "(the translator trained at $R$ is subtracted; its gain at $R$ is listed). Tuned lens: KL to the model's own "
        "predictions at $R$, as in \\citet{belrose2023tuned}; at $R$ it is the identity. Shares are of the raw gap "
        "between $r = 1$ and $R$, with $95\\%$ paired document-bootstrap intervals over the $200$ evaluation documents.}",
        "\\label{tab:tunedlens}", "\\end{table}"])
    (HERE / "out" / "tuned_lens" / "tuned_lens_table.tex").write_text(tex + "\n")
    print(tex)


if __name__ == "__main__":
    main()
