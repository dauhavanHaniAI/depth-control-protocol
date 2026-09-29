#!/usr/bin/env python3
"""
exp_depth_ablation.py — does VERA's latent recursion actually do work?

Gates the "RL over latent depth" research direction. Three experiments, all on
a frozen checkpoint, all writing structured records for later paper use.

  A. DEPTH ABLATION. Run the reasoning stack truncated to k of 8 total
     iterations (k = 1,2,4,8) and measure held-out teacher-forced NLL and
     next-token accuracy, plus (optionally) greedy GSM8K accuracy.
     A flat curve ⇒ the recursion contributes nothing and there is no credit
     to assign — fix that before building any RL theory on it.

  B. GATE DISTRIBUTION. Collect the per-(position, iteration) halt gate
     g = σ(-(E + α|ΔE|)/τ). If gate_reg has pinned everything at the target
     (0.5), the halt "decision" is degenerate and there is no policy to learn.

  C. DEPTH-WISE SEPARABILITY. The inline verifier head is NOT trained in the
     current checkpoints (use_verifier=False, no verifier.* weights), so its
     monotonicity cannot be measured directly. Instead we ask the question
     that actually decides whether proposal P1 is viable: is the information
     a value head would need PRESENT in the depth-i latent state? We fit a
     linear probe on the frozen thought_state at each iteration to separate
     correct solutions from answer-corrupted ones, and report AUC vs depth.
     Rising AUC ⇒ a per-depth value head is learnable ⇒ P1 is nearly free.

Truncation keeps the ORIGINAL global iteration numbering (global_iter =
block_idx * max_reasoning_iters + it) so the learned per-iteration embeddings
stay aligned with training. Note that with n_reasoning_blocks=2 and
max_reasoning_iters=4, k<=4 runs only block 0 — depth is confounded with block
identity, and the per-k block usage is recorded in the log.

Outputs (--out, default exp/depth_ablation/):
    results.json        every metric, config, and environment fact
    summary.md          paper-ready prose summary of the three verdicts
    fig1_depth.png      accuracy/NLL vs iterations
    fig2_gates.png      gate distribution per iteration
    fig3_probe.png      probe AUC vs depth, correct vs corrupted

Usage:
  CUDA_VISIBLE_DEVICES=2 python exp_depth_ablation.py \\
      --ckpt checkpoints/vera_psi_v2_latest.pt --n_teacher 200
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model.vera_psi as _vp                                          # noqa: E402
sys.modules["vera_psi"] = _vp
from model.vera_psi import VERAPsi, VERAArgs                          # noqa: E402
from model.sft_train import StreamTagger                              # noqa: E402
from mathverify_util import extract_boxed, verify_answer              # noqa: E402
from calibration import bootstrap_nll, fit_temperature               # noqa: E402
from tokenizers import Tokenizer                                      # noqa: E402


# ── instrumented forward ──────────────────────────────────────────────────
def build_plan(ma, k, mode="prefix"):
    """Which (block, iter) applications to run, in order.

    prefix  first k of the 8 — the naive ablation
    suffix  last k — controls for "which iterations" vs "how many"
    repeat  first k, then repeat the k-th to 8 applications — matches the
            residual arrival statistics the output head was trained on while
            holding DISTINCT computation at k. If this recovers the full-depth
            NLL, the prefix curve was measuring output-head miscalibration,
            not reasoning.
    """
    pairs = [(bi, it) for bi in range(ma.n_reasoning_blocks)
             for it in range(ma.max_reasoning_iters)]
    total = len(pairs)
    k = max(1, min(k, total))
    if mode == "prefix":
        return pairs[:k]
    if mode == "suffix":
        return pairs[-k:]
    if mode == "repeat":
        return pairs[:k] + [pairs[k - 1]] * (total - k)
    raise ValueError(f"unknown mode {mode}")


@torch.no_grad()
def forward_instrumented(model, ids, sids, ma, plan=None, capture=False, last_only=False):
    """Mirror of sft_train.sft_forward, with (a) an explicit iteration plan and
    (b) capture of per-iteration gate / energy / latent state."""
    B, seq = ids.shape
    x = model.tok_embeddings(ids)

    causal = model._mask_cache.causal(seq, x.device)
    for layer in model.perception_layers:
        x = layer(x, sids, 0, causal)
    x = model.mid_norm(x)

    n_t = ma.n_thought_tokens
    ts = torch.zeros(B, n_t, device=sids.device, dtype=sids.dtype)
    fs = torch.cat([ts, sids], dim=1)
    tm = model._mask_cache.thought_causal(n_t, seq, x.device)

    thought_state = model.thought_tokens.init_state(B, x.device, x.dtype)
    prev_energy = None
    gates, energies, states = [], [], []
    blocks_used = []

    melt_states = {}
    if ma.use_melt_kv:
        for bi, blk in enumerate(model.reasoning_blocks):
            if getattr(blk, "use_melt", False):
                melt_states[bi] = blk.melt_kv.init_state(
                    B, n_t + seq, x.device, x.dtype)

    if plan is None:
        plan = build_plan(ma, ma.n_reasoning_blocks * ma.max_reasoning_iters)
    for bi, it in plan:
        block = model.reasoning_blocks[bi]
        # keep ORIGINAL numbering so iter_embeds match training
        g_idx = bi * ma.max_reasoning_iters + it
        thought_state, _, branches = model.thought_tokens.step_thought(
            thought_state, g_idx, hard=False)
        if getattr(model, "thought_bridge", None) is not None:
            x = model.thought_bridge.apply_to_real(thought_state, x, branches)
        x_with = model.thought_tokens.inject(x, thought_state)
        ms = melt_states.get(bi)
        x_with, gate, E, ms_new, _ = block(x_with, fs, tm, prev_energy, ms)
        x, thought_state = model.thought_tokens.strip(x_with)
        if E is not None:
            prev_energy = E
        if ms_new is not None:
            melt_states[bi] = ms_new
        if capture:
            gates.append(gate.detach().float().flatten().cpu())
            energies.append(
                E.detach().float().mean().item() if E is not None else None)
            v_in = thought_state
            if getattr(model, "thought_bridge", None) is not None:
                v_in = model.thought_bridge.thought_for_verifier(
                    thought_state, x)
            states.append(v_in.detach().float().mean(dim=1)[0].cpu())
        blocks_used.append(bi)

    x = model.final_norm(x)
    # last_only: autoregressive sampling needs the final position only, but the
    # full projection materialises [B, S, vocab] in fp32 on EVERY generated
    # token. At B=16, S~700, vocab 32000 that is 1.4 GB allocated and freed per
    # token, which fragmented the allocator badly enough to OOM a run at step 47
    # with 18 GB sitting reserved-but-unallocated.
    if last_only:
        x = x[:, -1:, :]
    logits = model.output(x).float()
    return logits, gates, energies, states, sorted(set(blocks_used))


# ── data ──────────────────────────────────────────────────────────────────
PROMPT_END = "</problem>"
_ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def load_examples(path, n, seed=0):
    rows = []
    for line in Path(path).open():
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("text") and d.get("answer"):
            rows.append(d)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def corrupt(d):
    """Answer-corrupted twin: reasoning unchanged, final answer replaced.
    Creates a genuine reasoning/answer inconsistency for the probe."""
    ans = d["answer"].strip()
    m = re.fullmatch(r"-?\d+", ans)
    wrong = str(int(ans) + 7) if m else "42"
    if wrong == ans:
        wrong = "13"
    txt = _ANS_RE.sub(f"<answer>{wrong}</answer>", d["text"])
    box = extract_boxed(txt)
    if box:
        txt = txt.replace(f"\\boxed{{{box}}}", f"\\boxed{{{wrong}}}")
    return txt, wrong


def teacher_metrics(model, tok, tagger, ma, rows, plan, device, max_len,
                    ptdtype, calibrate=False, boot_T=0):
    """Held-out NLL + next-token accuracy over the RESPONSE region only.

    calibrate=True also fits a single logit temperature T (shared across the
    whole set) and reports the temperature-calibrated NLL. If truncating the
    recursion mainly leaves the residual off-distribution for a final_norm /
    output head trained at full depth, calibration recovers most of the gap —
    which would mean the raw curve measures miscalibration, not reasoning.
    """
    tot_nll, tot_tok, tot_correct = 0.0, 0, 0
    cal_logits, cal_tgts = [], []
    doc_nll, doc_tok = [], []          # per-document, for the bootstrap CI
    for d in rows:
        text = d["text"]
        ids = tok.encode(text).ids[:max_len]
        if len(ids) < 32:
            continue
        cut = text.find(PROMPT_END)
        p_len = 1
        if cut > 0:
            p_len = min(1 + len(tok.encode(text[:cut + len(PROMPT_END)],
                                           add_special_tokens=False).ids),
                        len(ids) - 1)
        x = torch.tensor([ids], device=device)
        sids = tagger.tag(x)
        with torch.autocast("cuda", dtype=ptdtype):
            logits, *_ = forward_instrumented(model, x, sids, ma, plan=plan)
        lg = logits[0, p_len - 1:-1]
        tgt = x[0, p_len:]
        if tgt.numel() == 0:
            continue
        nll = F.cross_entropy(lg.float(), tgt, reduction="sum")
        doc_nll.append(nll.item()); doc_tok.append(tgt.numel())
        if calibrate and len(cal_tgts) < 40:      # cap: full logits are large
            cal_logits.append(lg.float().cpu())
            cal_tgts.append(tgt.cpu())
        tot_nll += nll.item()
        tot_tok += tgt.numel()
        tot_correct += (lg.argmax(-1) == tgt).sum().item()
    out = {
        "nll": tot_nll / max(tot_tok, 1),
        "ppl": float(np.exp(tot_nll / max(tot_tok, 1))),
        "next_token_acc": tot_correct / max(tot_tok, 1),
        "n_tokens": tot_tok,
    }
    out.update(bootstrap_nll(doc_nll, doc_tok))
    if calibrate and cal_tgts:
        out.update(fit_temperature(cal_logits, cal_tgts, device=device,
                                   n_boot=boot_T))
    return out


@torch.no_grad()
def gsm8k_accuracy(model, tok, tagger, ma, plan, device, n, max_new, ptdtype):
    """Greedy generation accuracy (slow: no KV cache). 0 to skip."""
    rows = []
    for line in Path("data/raw/gsm8k_test.jsonl").open():
        d = json.loads(line)
        if d.get("question") and d.get("answer"):
            rows.append(d)
        if len(rows) >= n:
            break
    bos = tok.token_to_id("<bos>")
    stop = {tok.token_to_id("</answer>"), tok.token_to_id("<eos>")}
    stop = {s for s in stop if s is not None}
    correct = 0
    for d in rows:
        prompt = f"<problem>\n{d['question']}\n</problem>\n<think>\n"
        ids = [bos] + tok.encode(prompt, add_special_tokens=False).ids
        x = torch.tensor([ids], device=device)
        gen = []
        for _ in range(max_new):
            sids = tagger.tag(x)
            with torch.autocast("cuda", dtype=ptdtype):
                logits, *_ = forward_instrumented(model, x, sids, ma, plan=plan)
            nxt = int(logits[0, -1].argmax())
            if nxt in stop:
                break
            gen.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], device=device)], 1)
            if len(gen) >= 24 and len(set(gen[-24:])) <= 2:
                break                                    # degenerate loop
        out = tok.decode(gen)
        if verify_answer(str(d["answer"]), out):
            correct += 1
    return {"acc": correct / max(len(rows), 1), "n": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/vera_psi_v2_latest.pt")
    ap.add_argument("--tokenizer", default="tokenizer_en_math/tokenizer.json")
    ap.add_argument("--val", default="data/sft/longcot_4k/val.jsonl")
    ap.add_argument("--iters", default="1,2,4,8")
    ap.add_argument("--mode", choices=["prefix", "suffix", "repeat"],
                    default="prefix",
                    help="prefix=first k (naive); suffix=last k (which vs how "
                         "many); repeat=first k padded to 8 by repeating the "
                         "k-th (controls output-head calibration)")
    ap.add_argument("--boot_T", type=int, default=0,
                    help="bootstrap replicates for the fitted temperature "
                         "(0 = point estimate only)")
    ap.add_argument("--calibrate", action="store_true",
                    help="also fit a per-k logit temperature and report the "
                         "calibrated NLL")
    ap.add_argument("--n_teacher", type=int, default=200)
    ap.add_argument("--n_probe", type=int, default=150)
    ap.add_argument("--n_gate", type=int, default=60)
    ap.add_argument("--gsm8k_n", type=int, default=0,
                    help="greedy GSM8K problems per setting (slow; 0 skips)")
    ap.add_argument("--max_new", type=int, default=400)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--out", default="exp/depth_ablation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    ptdtype = torch.bfloat16
    t0 = time.time()
    torch.manual_seed(args.seed)

    # ── model ──
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ma = ck["model_args"]
    ma = VERAArgs(**ma) if isinstance(ma, dict) else ma
    ma.gradient_checkpointing = False
    ma.dropout = 0.0
    model = VERAPsi(ma)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    model = model.to(device).eval()
    tok = Tokenizer.from_file(args.tokenizer)
    tagger = StreamTagger(tok)
    n_total_iters = ma.n_reasoning_blocks * ma.max_reasoning_iters
    ks = [int(k) for k in args.iters.split(",") if int(k) <= n_total_iters]

    rec = {
        "experiment": "vera_depth_ablation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ckpt": args.ckpt,
        "ckpt_step": int(ck.get("step", -1)),
        "ckpt_metrics": ck.get("metrics", {}),
        "config": {
            "n_perception_layers": ma.n_perception_layers,
            "n_reasoning_blocks": ma.n_reasoning_blocks,
            "max_reasoning_iters": ma.max_reasoning_iters,
            "total_iters": n_total_iters,
            "n_thought_tokens": ma.n_thought_tokens,
            "halt_mode": getattr(ma, "halt_mode", "?"),
            "energy_temperature": getattr(ma, "energy_temperature", None),
            "use_verifier": getattr(ma, "use_verifier", None),
            "use_melt_kv": getattr(ma, "use_melt_kv", None),
            "n_branches": getattr(ma, "n_branches", None),
        },
        "load": {"missing": len(missing), "unexpected": len(unexpected)},
        "mode": args.mode,
        "calibrate": args.calibrate,
        "args": vars(args),
        "results": {},
    }
    del ck
    print(f"loaded {args.ckpt} step={rec['ckpt_step']} "
          f"missing={len(missing)} | total_iters={n_total_iters}", flush=True)

    rows = load_examples(args.val, max(args.n_teacher, args.n_probe, args.n_gate),
                         args.seed)
    print(f"held-out examples: {len(rows)}", flush=True)

    # ── A. depth ablation ──
    print("\n── A. depth ablation", flush=True)
    depth = {}
    print(f"   mode={args.mode}  calibrate={args.calibrate}", flush=True)
    for k in ks:
        plan = build_plan(ma, k, args.mode)
        m = teacher_metrics(model, tok, tagger, ma, rows[:args.n_teacher], plan,
                            device, args.max_len, ptdtype,
                            calibrate=args.calibrate, boot_T=args.boot_T)
        m["plan"] = [list(p) for p in plan]
        m["n_applications"] = len(plan)
        m["distinct_iters"] = len({tuple(p) for p in plan})
        if args.gsm8k_n:
            m.update({"gsm8k": gsm8k_accuracy(model, tok, tagger, ma, plan,
                                              device, args.gsm8k_n,
                                              args.max_new, ptdtype)})
        depth[k] = m
        print(f"   k={k:2d} ({args.mode}: {len(plan)} applications, "
              f"{m['distinct_iters']} distinct)  nll={m['nll']:.4f}  "
              f"ppl={m['ppl']:7.2f}  next_tok_acc={m['next_token_acc']*100:5.2f}%"
              + (f"  cal_nll={m['cal_nll']:.4f} (T={m['cal_temperature']:.2f})"
                 if args.calibrate and "cal_nll" in m else "")
              + (f"  gsm8k={m['gsm8k']['acc']*100:.1f}%" if args.gsm8k_n else "")
              + f"   [{time.time()-t0:.0f}s]", flush=True)
    rec["results"]["depth_ablation"] = {str(k): v for k, v in depth.items()}

    nlls = [depth[k]["nll"] for k in ks]
    rel = (nlls[0] - nlls[-1]) / max(abs(nlls[0]), 1e-9)
    rec["results"]["depth_verdict"] = {
        "nll_at_min_iters": nlls[0], "nll_at_max_iters": nlls[-1],
        "relative_nll_reduction": rel,
        "verdict": ("RECURSION DOES WORK" if rel > 0.02 else
                    "FLAT — recursion contributes ~nothing"),
    }

    # ── B. gate distribution ──
    print("\n── B. gate distribution", flush=True)
    per_iter = [[] for _ in range(n_total_iters)]
    for d in rows[:args.n_gate]:
        ids = tok.encode(d["text"]).ids[:args.max_len]
        if len(ids) < 32:
            continue
        x = torch.tensor([ids], device=device)
        with torch.autocast("cuda", dtype=ptdtype):
            _, gates, energies, _, _ = forward_instrumented(
                model, x, tagger.tag(x), ma, capture=True)
        for i, g in enumerate(gates):
            per_iter[i].append(g.numpy())
    gate_stats = []
    for i, chunks in enumerate(per_iter):
        if not chunks:
            continue
        v = np.concatenate(chunks)
        gate_stats.append({
            "iter": i, "mean": float(v.mean()), "std": float(v.std()),
            "p05": float(np.percentile(v, 5)), "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95)),
            "frac_within_0.05_of_target": float(np.mean(np.abs(v - 0.5) < 0.05)),
            "n": int(v.size),
        })
        print(f"   iter {i}: mean={v.mean():.3f} std={v.std():.3f} "
              f"p05={np.percentile(v,5):.3f} p95={np.percentile(v,95):.3f}",
              flush=True)
    rec["results"]["gate_stats"] = gate_stats
    if not gate_stats:
        rec["results"]["gate_verdict"] = {"verdict": "SKIPPED (--n_gate 0)"}
    else:
        spread = float(np.mean([g["std"] for g in gate_stats]))
        rec["results"]["gate_verdict"] = {
            "mean_within_iter_std": spread,
            "verdict": ("DEGENERATE — gates pinned near target, no policy to "
                        "learn" if spread < 0.05 else
                        "NON-DEGENERATE — gates vary, an allocation policy is "
                        "learnable"),
        }

    # ── C. depth-wise separability probe ──
    print("\n── C. depth-wise separability probe", flush=True)
    if args.n_probe == 0:
        print("   skipped (--n_probe 0)", flush=True)
    X = [[] for _ in range(n_total_iters)]
    y = []
    for d in rows[:args.n_probe]:
        for text, label in ((d["text"], 1), (corrupt(d)[0], 0)):
            ids = tok.encode(text).ids[:args.max_len]
            if len(ids) < 32:
                continue
            x = torch.tensor([ids], device=device)
            with torch.autocast("cuda", dtype=ptdtype):
                _, _, _, states, _ = forward_instrumented(
                    model, x, tagger.tag(x), ma, capture=True)
            for i, s in enumerate(states):
                X[i].append(s.numpy())
            y.append(label)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    y = np.array(y)
    probe = []
    for i in range(n_total_iters):
        if len(X[i]) != len(y) or len(y) < 20:
            continue
        Xi = np.stack(X[i])
        Xi = (Xi - Xi.mean(0)) / (Xi.std(0) + 1e-6)
        auc = cross_val_score(
            LogisticRegression(max_iter=2000, C=0.1), Xi, y,
            cv=5, scoring="roc_auc")
        probe.append({"iter": i, "auc_mean": float(auc.mean()),
                      "auc_std": float(auc.std()), "n": int(len(y))})
        print(f"   iter {i}: probe AUC = {auc.mean():.3f} ± {auc.std():.3f}",
              flush=True)
    rec["results"]["probe"] = probe
    if probe:
        first, last = probe[0]["auc_mean"], probe[-1]["auc_mean"]
        rec["results"]["probe_verdict"] = {
            "auc_first_iter": first, "auc_last_iter": last, "delta": last - first,
            "verdict": ("SEPARABLE and RISING — a per-depth value head is "
                        "learnable (P1 viable)" if last > 0.65 and last - first > 0.02
                        else "WEAK — depth-wise value head has little to learn"),
        }

    # ── plots ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ks, [depth[k]["nll"] for k in ks], "o-", color="#2b6cb0")
    ax[0].set_xlabel("reasoning iterations"); ax[0].set_ylabel("held-out NLL")
    ax[0].set_title("A. NLL vs latent depth"); ax[0].grid(alpha=.3)
    ax[1].plot(ks, [depth[k]["next_token_acc"] * 100 for k in ks], "o-",
               color="#2f855a", label="next-token acc")
    if args.gsm8k_n:
        ax[1].plot(ks, [depth[k]["gsm8k"]["acc"] * 100 for k in ks], "s--",
                   color="#c05621", label="GSM8K greedy")
    ax[1].set_xlabel("reasoning iterations"); ax[1].set_ylabel("accuracy (%)")
    ax[1].set_title("A. accuracy vs latent depth"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "fig1_depth.png", dpi=150); plt.close(fig)

    if gate_stats:
        fig, ax = plt.subplots(figsize=(7, 4))
        it = [g["iter"] for g in gate_stats]
        mu = np.array([g["mean"] for g in gate_stats])
        lo = np.array([g["p05"] for g in gate_stats])
        hi = np.array([g["p95"] for g in gate_stats])
        ax.plot(it, mu, "o-", color="#6b46c1", label="mean gate")
        ax.fill_between(it, lo, hi, alpha=.2, color="#6b46c1", label="p5–p95")
        ax.axhline(0.5, ls="--", c="k", lw=1, label="gate_reg target")
        ax.set_xlabel("reasoning iteration"); ax.set_ylabel("halt gate g")
        ax.set_ylim(0, 1); ax.set_title("B. halt-gate distribution")
        ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out / "fig2_gates.png", dpi=150); plt.close(fig)

    if probe:
        fig, ax = plt.subplots(figsize=(7, 4))
        it = [p["iter"] for p in probe]
        mu = np.array([p["auc_mean"] for p in probe])
        sd = np.array([p["auc_std"] for p in probe])
        ax.errorbar(it, mu, yerr=sd, fmt="o-", color="#b83280", capsize=3)
        ax.axhline(0.5, ls="--", c="k", lw=1, label="chance")
        ax.set_xlabel("reasoning iteration"); ax.set_ylabel("probe AUC")
        ax.set_ylim(0.4, 1.0)
        ax.set_title("C. correct-vs-corrupted separability by depth")
        ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out / "fig3_probe.png", dpi=150); plt.close(fig)

    rec["runtime_sec"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(rec, indent=2))

    # ── paper-ready summary ──
    dv = rec["results"]["depth_verdict"]
    gv = rec["results"]["gate_verdict"]
    pv = rec["results"].get("probe_verdict", {})
    md = [
        f"# VERA depth-ablation experiment\n",
        f"- checkpoint: `{args.ckpt}` (step {rec['ckpt_step']})",
        f"- architecture: {ma.n_perception_layers} perception layers, "
        f"{ma.n_reasoning_blocks} reasoning blocks x {ma.max_reasoning_iters} "
        f"iters = {n_total_iters} total, {ma.n_thought_tokens} thought tokens",
        f"- held-out: `{args.val}` (n={args.n_teacher} teacher-forced, "
        f"{args.n_probe} probe pairs, {args.n_gate} gate)",
        f"- runtime: {rec['runtime_sec']}s\n",
        f"- iteration plan mode: **{args.mode}**"
        + ("  (calibrated NLL reported)" if args.calibrate else "") + "\n",
        "## A. Does latent depth do work?\n",
        "| iters | NLL | ppl | next-token acc |" +
        (" GSM8K |" if args.gsm8k_n else ""),
        "|---|---|---|---|" + ("---|" if args.gsm8k_n else ""),
    ]
    for k in ks:
        m = depth[k]
        row = (f"| {k} | {m['nll']:.4f} | {m['ppl']:.2f} | "
               f"{m['next_token_acc']*100:.2f}% |")
        if args.gsm8k_n:
            row += f" {m['gsm8k']['acc']*100:.1f}% |"
        md.append(row)
    md += [
        f"\nRelative NLL reduction from {ks[0]} to {ks[-1]} iterations "
        f"(mode={args.mode}): "
        f"**{dv['relative_nll_reduction']*100:.2f}%** → **{dv['verdict']}**\n",
        "## B. Is the halt gate a real decision?\n",
        (f"Mean within-iteration gate std = "
         f"{gv['mean_within_iter_std']:.4f} → **{gv['verdict']}**\n"
         if "mean_within_iter_std" in gv else f"**{gv['verdict']}**\n"),
        "## C. Is a per-depth value head learnable?\n",
    ]
    if not pv:
        md.append("Skipped.\n")
    if pv:
        md.append(f"Probe AUC {pv['auc_first_iter']:.3f} (iter 0) → "
                  f"{pv['auc_last_iter']:.3f} (iter {n_total_iters-1}), "
                  f"delta {pv['delta']:+.3f} → **{pv['verdict']}**\n")
        md.append("Note: the inline verifier head is untrained in this "
                  "checkpoint (`use_verifier=False`), so this probes whether "
                  "the *information* a value head would need is present in the "
                  "depth-i latent state.\n")
    (out / "summary.md").write_text("\n".join(md))

    print(f"\n── {out}")
    print(f"   A: {dv['verdict']}")
    print(f"   B: {gv['verdict']}")
    if pv:
        print(f"   C: {pv['verdict']}")
    print(f"   results.json / summary.md / fig1-3.png   [{rec['runtime_sec']}s]")


if __name__ == "__main__":
    main()
