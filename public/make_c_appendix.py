#!/usr/bin/env python3
"""make_c_appendix.py — LaTeX tables for the reviewer-requested analyses (out/public-sft/reviewer_c, gsm8k, albert)."""
import collections
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).parent
RC = HERE / "out" / "public-sft" / "reviewer_c"
T = []


def table(spec, header, rows, caption, label, size="\\scriptsize\\setlength{\\tabcolsep}{3pt}"):
    return "\n".join(["\\begin{table}[t]", "\\centering", size, f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
                      header + " \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}",
                      f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"])


# geometry
g = json.loads((RC / "geometry.json").read_text())
lab = {"full": "full (8)", "prefix1": "prefix(1)", "prefix4": "prefix(4)", "repeat1": "repeat(1)",
       "repeat4": "repeat(4)", "suffix1": "suffix(1)", "suffix4": "suffix(4)", "extend12": "extend to 12",
       "extend16": "extend to 16", "cycle16": "second block twice (16)"}
rows = []
for n in ("prefix1", "prefix4", "repeat1", "repeat4", "suffix1", "suffix4", "full", "extend12", "cycle16", "extend16"):
    v = g[n]
    rows.append(f"{lab[n]} & ${v['nll_subsample']:.2f}$ & ${v['h_norm_ratio']:.2f}$ & ${v['ln_in_std']:.2f}$ & "
                f"${v['h_eff_rank']:.0f}$ & ${v['top1_var_frac']:.2f}$ & ${v['h_anisotropy']:.2f}$ & "
                f"${v['principal_cos_mean32']:.2f}$ & ${v['z_cos_to_full']:.2f}$ & ${v['h_cka_to_full']:.2f}$ & "
                f"${v['h_frechet_norm']:.2f}$ & ${v['logit_norm']:.0f}$ & ${v['entropy']:.2f}$ \\\\")
T.append(table("lrrrrrrrrrrrr",
               "Schedule & NLL & $\\|h\\|$ ratio & LN-in std & Eff.\\ rank & Top-1 var. & Anisotropy & Princ.\\ cos & "
               "$\\cos(z)$ & CKA & Fr\\'echet & Logit norm & Entropy",
               rows,
               "Residual-stream diagnostics per schedule on the public checkpoint (mathematical text, $8{,}192$ sampled "
               "evaluation tokens). $h$: residual stream entering the final normalization (\\emph{LN-in std}: mean per-token "
               "standard deviation of $h$; full-depth value $4.11$); \\emph{Eff.\\ rank}: participation ratio of the covariance "
               "spectrum of $h$; \\emph{Top-1 var.}: variance fraction of the first principal component; \\emph{Princ.\\ cos}: "
               "mean cosine of the principal angles between the top-$32$ principal subspaces of $h$ and of the full-depth $h$; "
               "$\\cos(z)$, CKA and Fr\\'echet are measured against the full schedule on the same tokens.",
               "tab:c-geometry"))

# dynamics
d = json.loads((RC / "dynamics.json").read_text())
rows = []
for n, lab_ in (("full", "full"), ("repeat1", "repeat(1)"), ("repeat4", "repeat(4)"), ("extend16", "extend to 16")):
    v = d[n]
    k = len(v["rel_update"])
    for key, name in (("rel_update", "rel.\\ update"), ("state_cos", "state cos"), ("gate_mean", "mean gate")):
        vals = v[key][:16]
        rows.append(f"{lab_} & {name} & " + " & ".join(f"${x:.2f}$" for x in vals) + " & " * (16 - len(vals)) + " \\\\")
    rows.append("\\midrule")
rows = rows[:-1]
T.append(table("ll" + "r" * 16, "Schedule & Quantity & " + " & ".join(f"${j}$" for j in range(1, 17)), rows,
               "Per-application dynamics of the real-token stream (mean over $40$ documents): relative update "
               "$\\|x_j - x_{j-1}\\| / \\|x_{j-1}\\|$, cosine between successive states, and mean update gate. Under "
               "repeat the update of the repeated application shrinks and successive states become nearly identical "
               "(a contraction toward a fixed point); beyond the trained eight applications the gates saturate at $1$.",
               "tab:c-dynamics", size="\\tiny\\setlength{\\tabcolsep}{2pt}"))

# gate vs NLL
gn = json.loads((RC / "gate_nll.json").read_text())
rows = [f"${r['application']}$ & ${r['mean_gate']:.2f}$ & ${r['mean_update']:.2f}$ & ${r['mean_dnll']:.3f}$ & "
        f"${r['rho_gate_dnll']:.2f}$ & ${r['rho_update_dnll']:.2f}$ & ${r['rho_gate_difficulty']:.2f}$ \\\\" for r in gn]
T.append(table("rrrrrrr", "Application & Mean gate & Mean rel.\\ update & Mean $\\Delta$NLL & "
               "$\\rho$(gate, $\\Delta$NLL) & $\\rho$(update, $\\Delta$NLL) & $\\rho$(gate, difficulty)", rows,
               "Gate behaviour and predictive contribution, full schedule, all mathematical evaluation tokens. "
               "$\\Delta$NLL: per-token decrease in NLL from reading out after application $j-1$ to after application $j$ "
               "(i.e.\\ from prefix($j-1$) to prefix($j$)); difficulty: full-depth token NLL. $\\rho$: Spearman correlation "
               "over tokens.", "tab:c-gate", size="\\small"))

# calibration-set size
c = json.loads((RC / "calsize.json").read_text())
by = collections.defaultdict(list)
for r in c["runs"]:
    by[r["n_docs"]].append(r)
rows = []
for n, rs in sorted(by.items()):
    def ms(k):
        if k not in rs[0]:
            return "--"
        vals = [r[k] for r in rs]
        m = st.mean(vals)
        s = st.pstdev(vals) if len(vals) > 1 else 0.0
        scale = 100 if k.startswith("share") else 1
        return f"${m*scale:.{1 if scale == 100 else 3}f} \\pm {s*scale:.{1 if scale == 100 else 3}f}$"
    rows.append(f"${n}$ & ${rs[0]['tokens']}$ & ${len(rs)}$ & {ms('T_prefix1')} & {ms('T_full')} & {ms('share_T')} & {ms('share_affine')} \\\\")
T.append(table("rrrrrrr", "Calib.\\ docs & Tokens & Subsets & $T$, prefix(1) & $T$, full & Temp.-corr.\\ (\\%) & Affine (\\%)",
               rows, "Sensitivity to the number of calibration documents (public checkpoint, mathematical text; mean "
               "$\\pm$ s.d.\\ over random subsets). Shares are of the $0.996$-nat naive gap, evaluated on the fixed "
               "$200$ evaluation documents.", "tab:c-calsize", size="\\small"))

# readouts
r = json.loads((RC / "readouts.json").read_text())
raw, per, dc = r["raw"], r["per_depth"], r["depth_conditioned"]
rows = []
for n, lab_ in (("prefix1", "prefix(1)"), ("prefix2", "prefix(2)"), ("prefix4", "prefix(4)"), ("full", "full (8)")):
    rows.append(f"{lab_} & ${raw[n]:.3f}$ & ${per[n]:.3f}$ & ${dc[n]:.3f}$ \\\\")
rows.append("\\midrule")
rows.append(f"Gap to full & ${raw['prefix1'] - raw['full']:.3f}$ & ${per['prefix1'] - per['full']:.3f}$ & "
            f"${dc['prefix1'] - dc['full']:.3f}$ \\\\")
T.append(table("lrrr", "Schedule & Frozen readout & Per-depth readout & One depth-conditioned readout", rows,
               "Lightweight readouts on the frozen recurrent body (public checkpoint, mathematical text; NLL on the "
               "evaluation documents). Each readout adds a per-feature affine map and a rank-$64$ residual correction "
               "to the readout input and is trained on the disjoint calibration documents. \\emph{Per-depth}: a separate "
               "correction for each schedule. \\emph{Depth-conditioned}: one shared correction that receives a learned "
               "embedding of the depth, trained on all four schedules.", "tab:c-readouts", size="\\small"))
(RC / "appendix_c.tex").write_text("\n\n".join(T) + "\n")
print("\n\n".join(T)[:1500])


# ALBERT (public, weight-tied, fixed training depth)
def albert_table():
    rows = []
    for f in sorted((HERE / "out" / "albert").glob("*.json")):
        a = json.loads(f.read_text())
        L = a["train_depth"]
        name = a["model"].split("/")[-1]
        ks = sorted(a["depths"], key=int)
        full = a["depths"][str(L)]
        for k in ks:
            v = a["depths"][k]
            m, w = v["math"], v["web"]
            def share(x, key):
                gap = x["nll"] - (full["math"] if x is m else full["web"])["nll"]
                ref = (full["math"] if x is m else full["web"])[key]
                if abs(gap) < 1e-6:
                    return "--"
                val = round(100 * (gap - (x[key] - ref)) / gap)
                return f"${0 if val == 0 else val}$"
            mark = " (train)" if int(k) == L else ""
            rows.append(f"{name} & ${k}${mark} & ${m['nll']:.2f}$ & ${m['T']:.2f}$ & {share(m, 'nll_T')} & {share(m, 'nll_affine')} & "
                        f"${w['nll']:.2f}$ & ${w['T']:.2f}$ & {share(w, 'nll_T')} & {share(w, 'nll_affine')} \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    return table("llrrrrrrrr",
                 "Model & Applications & NLL & $T$ & Temp.\\ (\\%) & Affine (\\%) & NLL & $T$ & Temp.\\ (\\%) & Affine (\\%)",
                 ["& & \\multicolumn{4}{c}{Mathematical text} & \\multicolumn{4}{c}{General text} \\\\", "\\midrule"] + rows,
                 "ALBERT, a public weight-tied model trained at a single depth ($12$ applications for base, $24$ for large), "
                 "under truncation and extrapolation of the number of applications. Masked-LM NLL on seeded low-density mask "
                 "sets; temperature and affine maps fitted on disjoint calibration documents. Temp.\\ and Affine: part of the "
                 "gap to the training depth that each recalibration removes (for extrapolated depths, the gap is to the "
                 "training depth as well).", "tab:c-albert")


T.append(albert_table())


# GSM8K by schedule (paired on common items, exact McNemar vs. full)
def gsm8k_table():
    from math import comb
    G = HERE / "out" / "gsm8k_depth"
    R = {f.stem: {json.loads(l)["idx"]: json.loads(l) for l in f.open()} for f in G.glob("*.jsonl")}
    if "full" not in R:
        return None
    order = [s for s in ("full", "repeat4", "repeat1", "prefix4", "prefix1") if s in R]
    common = sorted(set.intersection(*(set(R[s]) for s in order)))
    n = len(common)
    lab = {"full": "full (8)", "repeat4": "repeat(4)", "repeat1": "repeat(1)", "prefix4": "prefix(4)", "prefix1": "prefix(1)"}
    rows = []
    for sch in order:
        acc = sum(R[sch][i]["correct"] for i in common) / n
        parse = sum(R[sch][i]["parsed"] for i in common) / n
        closed = sum(R[sch][i]["closed"] for i in common) / n
        if sch == "full":
            rows.append(f"{lab[sch]} & ${100*acc:.1f}$ & ${100*parse:.0f}$ & ${100*closed:.0f}$ & -- & -- & -- \\\\")
            continue
        b = sum(R["full"][i]["correct"] and not R[sch][i]["correct"] for i in common)
        c = sum((not R["full"][i]["correct"]) and R[sch][i]["correct"] for i in common)
        k, m = min(b, c), b + c
        p = min(1.0, 2 * sum(comb(m, j) for j in range(k + 1)) / 2 ** m) if m else 1.0
        rows.append(f"{lab[sch]} & ${100*acc:.1f}$ & ${100*parse:.0f}$ & ${100*closed:.0f}$ & ${b}$ & ${c}$ & ${p:.2g}$ \\\\")
    return table("lrrrrrr", "Schedule & Accuracy (\\%) & Parsed (\\%) & Closed (\\%) & Lost vs.\\ full & Gained vs.\\ full & McNemar $p$",
                 rows, f"GSM8K accuracy under each execution schedule on the public checkpoint (first {n} test problems, greedy "
                 "decoding, at most $512$ new tokens, prompt format of the model's training data). \\emph{{Lost}}/\\emph{{Gained}}: "
                 "problems solved by the full schedule but not by the truncated one, and vice versa; exact two-sided McNemar test "
                 "on these paired outcomes. \\emph{{Closed}}: generations that ended with \\texttt{{</answer>}}.", "tab:c-gsm8k", size="\\small")


_g = gsm8k_table()
if _g:
    T.append(_g)
(RC / "appendix_c.tex").write_text("\n\n".join(T) + "\n")
(RC / "appendix_c.tex").write_text("\n\n".join(T) + "\n")
