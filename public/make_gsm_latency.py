#!/usr/bin/env python3
"""make_gsm_latency.py — GSM8K-by-schedule and latency tables for the appendix."""
import json
from math import comb, sqrt
from pathlib import Path

HERE = Path(__file__).parent
G = HERE / "out" / "gsm8k_depth"
RC = HERE / "out" / "public-sft" / "reviewer_c"
R = {f.stem: {json.loads(l)["idx"]: json.loads(l) for l in f.open()} for f in G.glob("*.jsonl")}
order = ["full", "prefix4", "repeat4", "repeat1"]
common = sorted(set.intersection(*(set(R[s]) for s in order)))
n = len(common)
lab = {"full": "full (8)", "repeat4": "repeat(4)", "repeat1": "repeat(1)", "prefix4": "prefix(4)"}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def mcnemar(a, b):
    """a, b: schedule names. Returns (#a right & b wrong, #a wrong & b right, exact two-sided p)."""
    x = sum(R[a][i]["correct"] and not R[b][i]["correct"] for i in common)
    y = sum((not R[a][i]["correct"]) and R[b][i]["correct"] for i in common)
    m = x + y
    p = min(1.0, 2 * sum(comb(m, j) for j in range(min(x, y) + 1)) / 2 ** m) if m else 1.0
    return x, y, p


rows = []
for s in order:
    k = sum(R[s][i]["correct"] for i in common)
    lo, hi = wilson(k, n)
    parse = 100 * sum(R[s][i]["parsed"] for i in common) / n
    closed = 100 * sum(R[s][i]["closed"] for i in common) / n
    rows.append(f"{lab[s]} & ${k}$ & ${100 * k / n:.1f}$ & $[{lo:.1f}, {hi:.1f}]$ & ${parse:.0f}$ & ${closed:.0f}$ \\\\")
pairs = [("Application count at $d = 4$", "repeat4", "prefix4"),
         ("Distinct iterations of block 0", "repeat4", "repeat1"),
         ("Engaging the second block", "full", "repeat4"),
         ("Full vs.\\ naive truncation", "full", "prefix4")]
prow = []
for name, a, b in pairs:
    x, y, p = mcnemar(a, b)
    acc_a = sum(R[a][i]["correct"] for i in common) / n
    acc_b = sum(R[b][i]["correct"] for i in common) / n
    prow.append(f"{name} & {lab[a]} vs.\\ {lab[b]} & ${100 * (acc_a - acc_b):+.1f}$ & ${x}$ & ${y}$ & ${p:.2g}$ \\\\")
t1 = "\n".join([
    "\\begin{table}[t]", "\\centering", "\\small",
    "\\begin{tabular}{lrrrrr}", "\\toprule",
    "Schedule & Solved & Accuracy (\\%) & $95\\%$ Wilson CI & Parsed (\\%) & Closed (\\%) \\\\", "\\midrule", *rows,
    "\\bottomrule", "\\end{tabular}", "\\\\[0.6em]",
    "\\begin{tabular}{llrrrr}", "\\toprule",
    "Contrast & Schedules $A$ vs.\\ $B$ & $\\Delta$ acc.\\ (pts) & $A$ only & $B$ only & McNemar $p$ \\\\", "\\midrule", *prow,
    "\\bottomrule", "\\end{tabular}",
    f"\\caption{{GSM8K on the public checkpoint under each execution schedule (first ${n}$ test problems, greedy decoding, "
    "at most $512$ new tokens, the model's training prompt format). Top: accuracy with Wilson intervals; \\emph{Closed}: "
    "generations that ended with \\texttt{</answer>}. Bottom: the DCP contrasts at the task level, as paired item-level "
    "comparisons with an exact two-sided McNemar test ($A$ only / $B$ only: problems solved by one schedule and not the other).}",
    "\\label{tab:c-gsm8k}", "\\end{table}"])

L = json.loads((RC / "latency.json").read_text())
lr = []
for s, lbl in (("prefix1", "prefix(1)"), ("prefix4", "prefix(4)"), ("suffix4", "suffix(4)"), ("repeat1", "repeat(1)"),
               ("repeat4", "repeat(4)"), ("full", "full (8)"), ("extend16", "extend to 16")):
    v = L["results"][s]
    lr.append(f"{lbl} & ${v['applications']}$ & ${v['ms_median']:.0f}$ & ${v['tokens_per_s'] / 1000:.1f}$ & "
              f"${v['ms_median'] / L['results']['full']['ms_median']:.2f}$ & ${v['peak_mem_gb']:.2f}$ \\\\")
t2 = "\n".join([
    "\\begin{table}[t]", "\\centering", "\\small",
    "\\begin{tabular}{lrrrrr}", "\\toprule",
    "Schedule & Applications & Time (ms) & k tokens/s & Relative time & Peak memory (GB) \\\\", "\\midrule", *lr,
    "\\bottomrule", "\\end{tabular}",
    f"\\caption{{Wall-clock cost of each schedule on the public checkpoint: one teacher-forced forward pass over "
    f"$8 \\times 1{{,}}024$ tokens in bfloat16 on a single {L['gpu'].replace('NVIDIA ', '')}, median of $10$ runs after "
    "warm-up. Each block application costs about $27$~ms; the perception layers, which every schedule runs, account "
    "for about $60\\%$ of full-depth time, so truncating to one application is only $1.5\\times$ faster. Peak memory "
    "is unchanged because no per-application activations are stored at inference.}",
    "\\label{tab:c-latency}", "\\end{table}"])
(RC / "gsm_latency.tex").write_text(t1 + "\n\n" + t2 + "\n")
print(t1); print(t2)
