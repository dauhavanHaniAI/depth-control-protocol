#!/usr/bin/env python3
"""make_looped_tex.py — LaTeX table and prediction check for the externally trained looped models."""
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "out" / "looped"
ROWS = [  # tag, model label, training depth support
    ("ifm-s-fixed-r5", "IFM-S (0.30B)", "$\\{5\\}$"),
    ("ifm-s-fixed-pln5", "IFM-S (0.30B)", "PLN, mean $5$, $\\le 64$"),
    ("ifm-s-learned-entropy0p01", "IFM-S (0.30B)", "learned prior"),
    ("ifm-m-fixed-r5", "IFM-M (0.82B)", "$\\{5\\}$"),
    ("ifm-m-fixed-pln5", "IFM-M (0.82B)", "PLN, mean $5$, $\\le 64$"),
    ("ifm-m-learned-entropy0p01", "IFM-M (0.82B)", "learned prior"),
    ("ifm-l-fixed-r5", "IFM-L (2.49B)", "$\\{5\\}$"),
    ("ifm-l-fixed-pln5", "IFM-L (2.49B)", "PLN, mean $5$, $\\le 64$"),
    ("ouro-1.4B", "Ouro-1.4B", "$\\{1,\\dots,4\\}$, every loop read out"),
    ("ouro-2.6B", "Ouro-2.6B", "$\\{1,\\dots,4\\}$, every loop read out"),
]


def load(tag):
    f = OUT / tag / "summary.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return d if "prediction_inputs" in d else None


def row(d, label, sched, dom="math"):
    m = d["domains"][dom]
    ref = d["ref_depth"]
    v = m["vs_ref"]["1"]
    nll = {int(k): x for k, x in m["nll_raw"].items()}
    aff_gain_ref = nll[ref] - m["nll_aff"][str(ref)]
    beyond = [r for r in nll if ref < r <= 8]
    ext = max(nll[r] - nll[ref] for r in beyond)
    ci = lambda a: f"{{\\tiny[{100 * a[0]:.1f}, {100 * a[1]:.1f}]}}"
    return (f"{label} & {sched} & ${nll[ref]:.3f}$ & ${v['gap_raw']:.3f}$ & "
            f"${100 * v['temp_share']:.1f}$ {ci(v['temp_share_ci95'])} & "
            f"${100 * v['affine_share']:.1f}$ {ci(v['affine_share_ci95'])} & ${aff_gain_ref:.3f}$ & ${ext:+.3f}$ \\\\")


def predictions(d):
    p = d["prediction_inputs"]
    tag = d["tag"]
    if "fixed-r5" in tag:
        return {"P1": p["share_r1"] >= 0.05 and p["max_abs_T_minus_1_below_ref"] > 0.1 and p["abs_T_ref_minus_1"] < 0.05,
                "P4": p["max_abs_T_minus_1_outside"] > p["abs_T_ref_minus_1"]}
    if "pln5" in tag:
        return {"P2_share": p["share_r1"] < 0.05}
    if tag.startswith("ouro"):
        return {"P3": p["share_r1"] < 0.05 and p["max_abs_T_minus_1_inside"] <= 0.1,
                "P4": p["max_abs_T_minus_1_outside"] > p["max_abs_T_minus_1_inside"]}
    return {}


def main():
    lines, preds = [], {}
    for tag, label, sched in ROWS:
        d = load(tag)
        if d is None:
            continue
        lines.append(row(d, label, sched))
        preds[tag] = {**predictions(d), "inputs": d["prediction_inputs"]}
    for scale in "sml":
        a, b = load(f"ifm-{scale}-fixed-r5"), load(f"ifm-{scale}-fixed-pln5")
        if a and b:
            preds[f"P2_drift_{scale}"] = b["prediction_inputs"]["drift_inside"] < a["prediction_inputs"]["drift_inside"]
    tex = "\n".join([
        "\\begin{table}[t]", "\\centering", "\\footnotesize\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llrrrrrr}", "\\toprule",
        "Model & Training depths & NLL at $R$ & Gap $r{=}1$ & Temperature (\\%) & Affine (\\%) & Gain at $R$ & Beyond $R$ \\\\",
        "\\midrule", *lines, "\\bottomrule", "\\end{tabular}",
        "\\caption{Externally trained looped models, mathematical text ($200$ evaluation and $200$ disjoint calibration "
        "documents). $R$: training depth ($5$ for IFM, $4$ for Ouro). Gap: NLL at $r = 1$ minus NLL at $R$ (nats). "
        "Temperature and Affine: shares of that gap removed by a depth-specific temperature or affine readout-input map, "
        "with $95\\%$ paired document-bootstrap intervals ($2{,}000$ replicates); both shares are net of what the same "
        "probe recovers at $R$. Gain at $R$: NLL removed by the affine probe at the training depth itself (the "
        "full-depth control). PLN: Poisson-lognormal, capped at $64$. Beyond $R$: largest NLL increase over $R < r \\le 8$. Within each IFM scale, the "
        "fixed and sampled models share architecture, data and token budget \\citep{huang2026fixedpoints}.}",
        "\\label{tab:looped}", "\\end{table}"])
    (OUT / "looped_table.tex").write_text(tex + "\n")
    (OUT / "predictions_check.json").write_text(json.dumps(preds, indent=1))
    print(tex)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "inputs"} if isinstance(v, dict) else v
                      for k, v in preds.items()}, indent=1))


if __name__ == "__main__":
    main()
