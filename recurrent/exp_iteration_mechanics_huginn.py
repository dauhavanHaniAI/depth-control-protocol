#!/usr/bin/env python3
"""
exp_iteration_mechanics_huginn.py — the same three probes as
exp_iteration_mechanics.py, on Huginn-0125.

Huginn's recurrent core is one weight-tied stack of 4 SandwichBlocks applied r
times, so a "recurrence step" is one pass through all four. State after each
step is captured with a forward hook on the last core block; noise is injected
with a forward pre-hook on the first one at a chosen step index.

The point of running this second model: the probes on Sona produced a result
that contradicts the obvious mechanism (updates are near-orthogonal, the map is
expansive). Either that is a property of depth-recurrent stacks in general, or
it is specific to a fixed-depth-trained model. Huginn is the depth-sampled
counterpart, so it separates the two.

Usage:
  CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=... \
    python exp_iteration_mechanics_huginn.py --n 40
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_depth_huginn import install_legacy_tied_weights_shim, load_docs


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
    ap.add_argument("--model", default="tomg-group-umd/huginn-0125")
    ap.add_argument("--val", default="data/sft/longcot_4k/val.jsonl")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--noise_scale", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="exp/iteration_mechanics_huginn")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    install_legacy_tied_weights_shim()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()
    core = model.transformer.core_block
    print(f"nạp xong | lõi {len(core)} block, r={args.steps} bước đệ quy "
          f"[{time.time()-t0:.0f}s]", flush=True)

    captured, inject = [], {"at": None, "scale": 0.0, "count": 0, "gen": None}

    def cap_hook(mod, inp, outp):
        h = outp[0] if isinstance(outp, (tuple, list)) else outp
        captured.append(h.detach().float().clone())

    def pre_hook(mod, inp):
        if inject["at"] is None:
            return None
        i = inject["count"]; inject["count"] += 1
        if i != inject["at"]:
            return None
        x = inp[0]
        eps = torch.randn(x.shape, device=x.device, dtype=x.dtype,
                          generator=inject["gen"])
        eps = eps / eps.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        x = x + inject["scale"] * x.norm(dim=-1, keepdim=True) * eps
        return (x,) + tuple(inp[1:])

    h1 = core[-1].register_forward_hook(cap_hook)
    h2 = core[0].register_forward_pre_hook(pre_hook)

    docs = load_docs(args.val, args.n)
    gen = torch.Generator(device="cuda"); gen.manual_seed(args.seed)
    R = args.steps
    rel, cos_adj, cos_first, pr, contract = [], [], [], [], []

    for di, t in enumerate(docs):
        ids = tok(t, return_tensors="pt", truncation=True,
                  max_length=args.max_len).input_ids.to("cuda")
        if ids.shape[1] < 64:
            continue
        captured.clear(); inject["at"] = None
        with torch.no_grad():
            model(input_ids=ids, num_steps=int(R))
        xs = list(captured)
        if len(xs) < R:
            print(f"   cảnh báo: bắt được {len(xs)} trạng thái, kỳ vọng {R}")
            if len(xs) < 3: continue
        n = len(xs)
        dx = [(xs[i + 1] - xs[i])[0] for i in range(n - 1)]
        rel.append([float(dx[i].norm() / xs[i][0].norm()) for i in range(n - 1)])
        cos_adj.append([float(torch.nn.functional.cosine_similarity(
            dx[i].flatten(), dx[i - 1].flatten(), dim=0)) for i in range(1, n - 1)])
        cos_first.append([float(torch.nn.functional.cosine_similarity(
            dx[i].flatten(), dx[0].flatten(), dim=0)) for i in range(1, n - 1)])
        M = torch.stack([v.flatten() for v in dx])
        M = M / M.norm(dim=1, keepdim=True).clamp_min(1e-9)
        pr.append(participation_ratio(M))

        clean = [x.clone() for x in xs]
        captured.clear()
        inject.update(at=0, scale=args.noise_scale, count=0, gen=gen)
        with torch.no_grad():
            model(input_ids=ids, num_steps=int(R))
        noisy = list(captured); inject["at"] = None
        k = min(len(clean), len(noisy))
        d0 = (noisy[0] - clean[0]).norm().clamp_min(1e-9)
        contract.append([float((noisy[i] - clean[i]).norm() / d0) for i in range(k)])
        if (di + 1) % 10 == 0:
            print(f"   {di+1}/{len(docs)} [{time.time()-t0:.0f}s]", flush=True)

    h1.remove(); h2.remove()
    trim = lambda L: min(len(x) for x in L)
    A = lambda L: np.array([x[:trim(L)] for x in L]).mean(0).tolist()
    rec = {"model": args.model, "recurrence_steps": R, "n_docs": len(rel),
           "noise_scale": args.noise_scale,
           "A_relative_update": A(rel),
           "B_cos_adjacent": A(cos_adj), "B_cos_with_first": A(cos_first),
           "B_participation_ratio": float(np.mean(pr)),
           "B_participation_ratio_std": float(np.std(pr)),
           "B_pr_normalised": float(np.mean(pr)) / max(R - 1, 1),
           "C_noise_ratio": A(contract),
           "runtime_sec": round(time.time() - t0, 1)}
    (out / "results.json").write_text(json.dumps(rec, indent=2))

    f = lambda v, k=10: "  ".join(f"{x:.3f}" for x in v[:k])
    print("\n" + "=" * 70)
    print(f"A. cập nhật tương đối (10 bước đầu): {f(rec['A_relative_update'])}")
    print(f"   (10 bước cuối)                  : {f(rec['A_relative_update'][-10:])}")
    print(f"\nB. cos(dx_g, dx_1) (10 đầu)        : {f(rec['B_cos_with_first'])}")
    print(f"   participation ratio             : {rec['B_participation_ratio']:.2f}"
          f" / {R-1}  = {100*rec['B_pr_normalised']:.0f}% chiều tối đa")
    print(f"\nC. nhiễu (10 đầu)                  : {f(rec['C_noise_ratio'])}")
    print(f"   (10 cuối)                       : {f(rec['C_noise_ratio'][-10:])}")
    r = rec["C_noise_ratio"]
    print(f"   hệ số/bước                      : {(r[-1]/max(r[0],1e-9))**(1/max(len(r)-1,1)):.4f}")
    print("=" * 70)
    print(f"-> {out}/results.json  [{rec['runtime_sec']}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
