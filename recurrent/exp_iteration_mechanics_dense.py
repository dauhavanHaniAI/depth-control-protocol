#!/usr/bin/env python3
"""
exp_iteration_mechanics_dense.py — the same three dynamics probes on a dense
transformer, where one "iteration" is one layer.

WHY. On Sona (weight-shared, fixed-depth trained) the recurrence is expansive
(noise factor 1.067 per application); on Huginn (weight-shared, depth-sampled)
it is contractive (0.892). Continuing Sona for 1000 steps with sampled depth
moved it only 1.069 -> 1.053, so depth sampling is not what separates the two.
That leaves the question of what does. Measuring dense stacks answers half of
it: if a dense transformer is also expansive, expansion is generic to deep
stacks and only Huginn is unusual; if dense stacks contract, expansion is
specific to Sona.

Layer outputs come from output_hidden_states, so probes A and B need no hooks.
Probe C injects noise with a forward pre-hook on the first layer.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_depth_huginn import load_docs


def participation_ratio(M):
    G = (M @ M.T).double()
    ev = torch.linalg.eigvalsh(G).clamp_min(0)
    s = ev.sqrt()
    return float((s.sum() ** 2 / (s ** 2).sum().clamp_min(1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--val", default="data/sft/longcot_4k/val.jsonl")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--noise_scale", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to("cuda").eval()
    layers = (getattr(model, "model", model)).layers
    print(f"{args.model}: {len(layers)} lớp [{time.time()-t0:.0f}s]", flush=True)

    inj = {"on": False, "scale": 0.0, "gen": None}

    def pre_hook(mod, inp):
        if not inj["on"]:
            return None
        x = inp[0]
        eps = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=inj["gen"])
        eps = eps / eps.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x + inj["scale"] * x.norm(dim=-1, keepdim=True) * eps,) + tuple(inp[1:])

    h = layers[0].register_forward_pre_hook(pre_hook)
    docs = load_docs(args.val, args.n)
    gen = torch.Generator(device="cuda"); gen.manual_seed(args.seed)
    rel, cos_first, pr, contract = [], [], [], []

    for di, t in enumerate(docs):
        ids = tok(t, return_tensors="pt", truncation=True,
                  max_length=args.max_len).input_ids.to("cuda")
        if ids.shape[1] < 64:
            continue
        inj["on"] = False
        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states
        xs = [x.detach().float() for x in hs]
        n = len(xs) - 1
        dx = [(xs[i + 1] - xs[i])[0] for i in range(n)]
        rel.append([float(dx[i].norm() / xs[i][0].norm().clamp_min(1e-9)) for i in range(n)])
        cos_first.append([float(torch.nn.functional.cosine_similarity(
            dx[i].flatten(), dx[0].flatten(), dim=0)) for i in range(1, n)])
        M = torch.stack([v.flatten() for v in dx])
        M = M / M.norm(dim=1, keepdim=True).clamp_min(1e-9)
        pr.append(participation_ratio(M))

        inj.update(on=True, scale=args.noise_scale, gen=gen)
        with torch.no_grad():
            hs2 = model(input_ids=ids, output_hidden_states=True).hidden_states
        inj["on"] = False
        ys = [x.detach().float() for x in hs2]
        d0 = (ys[1] - xs[1]).norm().clamp_min(1e-9)
        contract.append([float((ys[i + 1] - xs[i + 1]).norm() / d0) for i in range(n)])
        if (di + 1) % 20 == 0:
            print(f"   {di+1}/{len(docs)} [{time.time()-t0:.0f}s]", flush=True)

    h.remove()
    A = lambda L: np.array(L).mean(0).tolist()
    N = len(rel[0])
    rec = {"model": args.model, "n_layers": N, "n_docs": len(rel),
           "noise_scale": args.noise_scale,
           "A_relative_update": A(rel), "B_cos_with_first": A(cos_first),
           "B_participation_ratio": float(np.mean(pr)),
           "B_pr_normalised": float(np.mean(pr)) / N,
           "C_noise_ratio": A(contract),
           "runtime_sec": round(time.time() - t0, 1)}
    (out / "results.json").write_text(json.dumps(rec, indent=2))
    r = rec["C_noise_ratio"]
    f = lambda v, k=8: "  ".join(f"{x:.3f}" for x in v[:k])
    print("\n" + "=" * 66)
    print(f"A. cập nhật (8 lớp đầu) : {f(rec['A_relative_update'])}")
    print(f"   (8 lớp cuối)         : {f(rec['A_relative_update'][-8:])}")
    print(f"B. participation ratio  : {rec['B_participation_ratio']:.2f}/{N}"
          f" = {100*rec['B_pr_normalised']:.0f}%")
    print(f"C. nhiễu (8 đầu)        : {f(r)}")
    print(f"   (8 cuối)             : {f(r[-8:])}")
    print(f"   hệ số/lớp            : {(r[-1]/max(r[0],1e-9))**(1/max(len(r)-1,1)):.4f}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
