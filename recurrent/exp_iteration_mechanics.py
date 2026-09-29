#!/usr/bin/env python3
"""
exp_iteration_mechanics.py — why does distinct depth contribute ~0?

The depth-ablation paper measures that extra distinct iterations inside a
reasoning block buy essentially nothing, but never says why. It reports the
learned GATE value per iteration (0.36 -> 0.69), which is not the same thing:
gamma = 0.74 means "write 74% of the block output", yet if that output is close
to the incoming residual the state barely moves. This measures the state.

Three probes, all single forward passes:

  A. CONTRIBUTION.  ||x^(g) - x^(g-1)|| / ||x^(g-1)|| per application, i.e. how
     much each iteration actually changes the real stream.

  B. NEW vs OLD.  cos(dx^(g), dx^(g-1)), and the participation ratio of the
     singular values of [dx^(1) ... dx^(N)]. If successive updates are nearly
     parallel and the set spans ~1 direction, then N applications act like one
     application with a larger step -- which is exactly what the repeat control
     found behaviourally, and would explain it mechanistically.

  C. NOISE RESPONSE.  Inject Gaussian noise into x at iteration g0 and track
     ||delta^(g)|| / ||delta^(g0)||. A contraction factor below 1 means the
     recurrence converges to a fixed point, so late iterations cannot add
     information no matter how open the gate is.

Usage:
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
    python exp_iteration_mechanics.py --n 100
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model.vera_psi as _vp
sys.modules["vera_psi"] = _vp
from model.vera_psi import VERAPsi, VERAArgs
from model.sft_train import StreamTagger
from exp_depth_ablation import build_plan, load_examples
from tokenizers import Tokenizer


@torch.no_grad()
def run_capture(model, ids, sids, ma, noise_at=None, noise_scale=0.0, gen=None):
    """Mirror of forward_instrumented, but capturing the real stream x after
    every block application, and optionally injecting noise at one of them."""
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
    melt_states = {}
    if ma.use_melt_kv:
        for bi, blk in enumerate(model.reasoning_blocks):
            if getattr(blk, "use_melt", False):
                melt_states[bi] = blk.melt_kv.init_state(B, n_t + seq, x.device, x.dtype)

    plan = build_plan(ma, ma.n_reasoning_blocks * ma.max_reasoning_iters)
    xs = [x.detach().float().clone()]
    gates = []
    for step, (bi, it) in enumerate(plan):
        if noise_at is not None and step == noise_at:
            eps = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)
            eps = eps / eps.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            x = x + noise_scale * x.norm(dim=-1, keepdim=True) * eps
        block = model.reasoning_blocks[bi]
        g_idx = bi * ma.max_reasoning_iters + it
        thought_state, _, branches = model.thought_tokens.step_thought(
            thought_state, g_idx, hard=False)
        if getattr(model, "thought_bridge", None) is not None:
            x = model.thought_bridge.apply_to_real(thought_state, x, branches)
        x_with = model.thought_tokens.inject(x, thought_state)
        ms = melt_states.get(bi)
        x_with, gate, E, ms_new, _ = block(x_with, fs, tm, prev_energy, ms)
        x, thought_state = model.thought_tokens.strip(x_with)
        if E is not None: prev_energy = E
        if ms_new is not None: melt_states[bi] = ms_new
        gates.append(float(gate.detach().float().mean()))
        xs.append(x.detach().float().clone())
    return xs, gates


def participation_ratio(M):
    """(sum s)^2 / sum s^2 over the singular values of M.

    Computed from the Gram matrix M M^T, which is [k, k] for k updates, rather
    than by an SVD of M itself: M is [k, seq*dim] and a direct SVD allocates
    gigabytes for a long sequence. The singular values of M are the square
    roots of the eigenvalues of M M^T, so the two are identical.
    """
    G = (M @ M.T).double()
    ev = torch.linalg.eigvalsh(G).clamp_min(0)
    s = ev.sqrt()
    return float((s.sum() ** 2 / (s ** 2).sum().clamp_min(1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/vera_v7_clean_step40000.pt")
    ap.add_argument("--tokenizer", default="tokenizer_en_math/tokenizer.json")
    ap.add_argument("--val", default="data/sft/longcot_4k/val.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--noise_scale", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="exp/iteration_mechanics")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev, ptd = "cuda", torch.bfloat16
    t0 = time.time()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ma = ck["model_args"]; ma = VERAArgs(**ma) if isinstance(ma, dict) else ma
    ma.gradient_checkpointing = False; ma.dropout = 0.0; ma.use_verifier = False
    model = VERAPsi(ma); model.load_state_dict(ck["model"], strict=False)
    step = int(ck.get("step", -1)); del ck
    model = model.to(dev).eval()
    tok = Tokenizer.from_file(args.tokenizer); tagger = StreamTagger(tok)
    N = ma.n_reasoning_blocks * ma.max_reasoning_iters
    docs = load_examples(args.val, args.n, args.seed)
    gen = torch.Generator(device=dev); gen.manual_seed(args.seed)
    print(f"ckpt {args.ckpt} step={step} | N={N} lần áp dụng | {len(docs)} tài liệu",
          flush=True)

    rel, cos_adj, cos_first, pr, contract, gate_acc = [], [], [], [], [], []
    for di, d in enumerate(docs):
        ids = tok.encode(d["text"], add_special_tokens=False).ids[:args.max_len]
        if len(ids) < 64: continue
        x0 = torch.tensor([[1] + ids], device=dev)
        sids = tagger.tag(x0)
        with torch.autocast("cuda", dtype=ptd):
            xs, gates = run_capture(model, x0, sids, ma)
        gate_acc.append(gates)
        dx = [(xs[i + 1] - xs[i])[0] for i in range(N)]          # [seq, dim] mỗi vòng
        rel.append([float(dx[i].norm() / xs[i][0].norm()) for i in range(N)])
        cos_adj.append([float(torch.nn.functional.cosine_similarity(
            dx[i].flatten(), dx[i - 1].flatten(), dim=0)) for i in range(1, N)])
        cos_first.append([float(torch.nn.functional.cosine_similarity(
            dx[i].flatten(), dx[0].flatten(), dim=0)) for i in range(1, N)])
        M = torch.stack([v.flatten() for v in dx])
        M = M / M.norm(dim=1, keepdim=True).clamp_min(1e-9)
        pr.append(participation_ratio(M))

        # --- C: bơm nhiễu ở vòng 0, theo dõi nhiễu loạn lan truyền ---
        with torch.autocast("cuda", dtype=ptd):
            xs_n, _ = run_capture(model, x0, sids, ma, noise_at=0,
                                  noise_scale=args.noise_scale, gen=gen)
        d0 = (xs_n[1] - xs[1]).norm()
        contract.append([float((xs_n[i + 1] - xs[i + 1]).norm() / d0.clamp_min(1e-9))
                         for i in range(N)])
        if (di + 1) % 20 == 0:
            print(f"   {di+1}/{len(docs)} [{time.time()-t0:.0f}s]", flush=True)

    A = lambda v: np.array(v).mean(0).tolist()
    rec = {"ckpt": args.ckpt, "ckpt_step": step, "N": N, "n_docs": len(rel),
           "noise_scale": args.noise_scale,
           "A_relative_update": A(rel), "gate_mean": A(gate_acc),
           "B_cos_adjacent": A(cos_adj), "B_cos_with_first": A(cos_first),
           "B_participation_ratio": float(np.mean(pr)),
           "B_participation_ratio_std": float(np.std(pr)),
           "C_noise_ratio": A(contract),
           "note": "A/B/C computed on the real stream x, not the gate value",
           "runtime_sec": round(time.time() - t0, 1)}
    (out / "results.json").write_text(json.dumps(rec, indent=2))

    f = lambda v: "  ".join(f"{x:.3f}" for x in v)
    print("\n" + "=" * 68)
    print("A. cập nhật tương đối ||dx||/||x||  :", f(rec["A_relative_update"]))
    print("   giá trị cổng (để đối chiếu)     :", f(rec["gate_mean"]))
    print("\nB. cos(dx_g, dx_{g-1})            :", f(rec["B_cos_adjacent"]))
    print("   cos(dx_g, dx_1)                 :", f(rec["B_cos_with_first"]))
    print(f"   participation ratio của {N} cập nhật: "
          f"{rec['B_participation_ratio']:.2f} ± {rec['B_participation_ratio_std']:.2f}"
          f"   (1 = mọi vòng cùng một hướng, {N} = trực giao hoàn toàn)")
    print("\nC. ||nhiễu_g|| / ||nhiễu_0||       :", f(rec["C_noise_ratio"]))
    r = rec["C_noise_ratio"]
    print(f"   hệ số co trung bình mỗi vòng    : {(r[-1]/1.0)**(1/max(N-1,1)):.3f}"
          "   (<1 = ánh xạ co, hội tụ điểm bất động)")
    print("=" * 68)
    print(f"-> {out}/results.json  [{rec['runtime_sec']}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
