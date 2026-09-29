#!/usr/bin/env python3
"""sona_c.py — reviewer-requested analyses on the public checkpoint of our architecture (math text).

  geometry   : residual-stream diagnostics per schedule vs. full (norms, covariance spectrum, principal angles,
               anisotropy, final-normalization input statistics, logit norm, entropy, Frechet distance)
  dynamics   : per-application relative update, successive-state cosine, successive-update cosine, mean gate
  gate_nll   : token-level association between gate / update size and the NLL change of each application
  calsize    : fitted temperature and recovered shares as a function of the number of calibration documents
  readouts   : lightweight per-depth readouts and one depth-conditioned readout, frozen body, trained on the
               calibration documents and evaluated on the disjoint evaluation documents
Usage: python sona_c.py --parts geometry,dynamics,gate_nll,calsize,readouts
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dcp_public as DP
import sona_adapter as S

HERE = Path(__file__).parent
OUT = HERE / "out" / "public-sft" / "reviewer_c"
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda"


def fmt(docs):
    out = []
    for d in docs:
        prob = d["prompt"][len("Problem: "):-len("\nSolution:")]
        out.append({**d, "prompt": f"<problem>\n{prob}\n</problem>\n<think>\n", "text": d["text"].lstrip()})
    return out


def load_tokens(tok, split, n=200):
    return DP.tokenize(tok, fmt(DP.load_split("math", split)[:n]), 1024, [tok.bos_token_id])


@torch.no_grad()
def collect(ad, toks, cfg, want_h=True):
    """Readout input z (post final norm), pre-norm state h, targets y, doc ids, for all scored tokens."""
    Z, H, Y, D = [], [], [], []
    for di, (ids, s) in enumerate(toks):
        if ids.numel() - s - 1 <= 0:  # prompt fills max_len: no scored tokens (same rule as dcp_public)
            continue
        x, _ = ad._run(ids[None].to(dev), cfg)
        h = x[0, s:-1]
        Z.append(ad.m.final_norm(h[None])[0].to(torch.bfloat16).cpu())
        if want_h:
            H.append(h.to(torch.bfloat16).cpu())
        Y.append(ids[s + 1:])
        D.append(torch.full((ids.numel() - s - 1,), di))
    return torch.cat(Z), (torch.cat(H) if want_h else None), torch.cat(Y), torch.cat(D)


def nll_of(head_w, Z, Y, T=1.0, bs=4096, adapter=None, depth=None):
    tot = []
    with torch.no_grad():
        for i in range(0, Y.numel(), bs):
            z = Z[i:i + bs].to(dev).float()
            if adapter is not None:
                z = adapter(z, depth)
            lg = F.linear(z, head_w) / T
            tot.append(F.cross_entropy(lg, Y[i:i + bs].to(dev), reduction="none").cpu())
    return torch.cat(tot)


# ----------------------------------------------------------------------------- geometry
def geometry_part(ad, toks):
    cf = dict(ad.configs())
    names = ["full", "prefix1", "prefix4", "repeat1", "repeat4", "suffix1", "suffix4", "extend12", "extend16", "cycle16"]
    rng = np.random.default_rng(0)
    feats = {}
    for n in names:
        Z, H, Y, _ = collect(ad, toks, cf[n])
        if "idx" not in feats:
            feats["idx"] = torch.tensor(np.sort(rng.choice(Y.numel(), min(8192, Y.numel()), replace=False)))
        idx = feats["idx"]
        W = ad.m.output.weight.float()
        with torch.no_grad():
            lg = F.linear(Z[idx].to(dev).float(), W)
            lp = lg.log_softmax(-1)
            ent = float((-(lp.exp() * lp).sum(-1)).mean())
            lnorm = float(lg.norm(dim=-1).mean())
            nll = float(F.cross_entropy(lg, Y[idx].to(dev)))
        feats[n] = dict(H=H[idx].float(), Z=Z[idx].float(), ent=ent, lnorm=lnorm, nll=nll)
        print(f"  geometry collected {n}", flush=True)
    R = feats["full"]
    Hr, Zr = R["H"], R["Z"]

    def pca(X, k):
        Xc = X - X.mean(0)
        U, Sv, Vh = torch.linalg.svd(Xc, full_matrices=False)
        ev = Sv ** 2
        return Vh[:k].T, ev / ev.sum()

    Pr, evr = pca(Hr, 64)
    out = {}
    for n in names:
        f = feats[n]
        H, Z = f["H"], f["Z"]
        P, ev = pca(H, 64)
        cosang = torch.linalg.svdvals(P[:, :32].T @ Pr[:, :32])  # cosines of principal angles, top-32 subspaces
        g = DP.geometry(H, Hr, Z, Zr)
        out[n] = {**g,
                  "h_norm_ratio": g["h_norm"] / g["h_norm_ref"],
                  "top1_var_frac": float(ev[0]), "top10_var_frac": float(ev[:10].sum()),
                  "principal_cos_mean32": float(cosang.mean()), "principal_cos_min32": float(cosang.min()),
                  "ln_in_std": float(H.std(dim=-1).mean()), "ln_in_std_ref": float(Hr.std(dim=-1).mean()),
                  "ln_in_mean_abs": float(H.mean(dim=-1).abs().mean()),
                  "logit_norm": f["lnorm"], "entropy": f["ent"], "nll_subsample": f["nll"]}
    (OUT / "geometry.json").write_text(json.dumps(out, indent=1))
    for n, v in out.items():
        print(f"{n:9s} nll {v['nll_subsample']:.3f} |h|r {v['h_norm_ratio']:.2f} cos(z) {v['z_cos_to_full']:.2f} "
              f"CKA {v['h_cka_to_full']:.2f} PA {v['principal_cos_mean32']:.2f} top1 {v['top1_var_frac']:.3f} "
              f"aniso {v['h_anisotropy']:.2f} lnstd {v['ln_in_std']:.2f}/{v['ln_in_std_ref']:.2f} "
              f"logit {v['logit_norm']:.1f} ent {v['entropy']:.2f} Fr {v['h_frechet_norm']:.2f}", flush=True)


# ----------------------------------------------------------------------------- dynamics + gate/NLL
@torch.no_grad()
def dynamics_part(ad, toks, n_docs=40):
    cf = dict(ad.configs())
    res = {}
    for n in ("full", "repeat1", "repeat4", "extend16", "cycle16"):
        rel, cs, cu, gm = [], [], [], []
        for ids, s in [t for t in toks if t[0].numel() - t[1] - 1 > 0][:n_docs]:
            _, tr = ad._run(ids[None].to(dev), cf[n], trace=True)
            xs = [t[0][0, s:].float() for t in tr]
            r, c, u, gg = [], [], [], []
            for j in range(1, len(xs)):
                d = xs[j] - xs[j - 1]
                r.append(float((d.norm(dim=-1) / xs[j - 1].norm(dim=-1).clamp_min(1e-6)).mean()))
                c.append(float(F.cosine_similarity(xs[j], xs[j - 1], dim=-1).mean()))
                if j >= 2:
                    dp = xs[j - 1] - xs[j - 2]
                    u.append(float(F.cosine_similarity(d, dp, dim=-1).mean()))
                gt = tr[j][1]
                gg.append(float(gt[0, s:].float().mean()) if gt is not None else float("nan"))
            rel.append(r); cs.append(c); cu.append(u); gm.append(gg)
        res[n] = {"rel_update": np.mean(rel, 0).tolist(), "state_cos": np.mean(cs, 0).tolist(),
                  "update_cos": np.mean(cu, 0).tolist(), "gate_mean": np.mean(gm, 0).tolist()}
        print(n, {k: [round(x, 3) for x in v] for k, v in res[n].items()}, flush=True)
    (OUT / "dynamics.json").write_text(json.dumps(res, indent=1))


@torch.no_grad()
def gate_nll_part(ad, toks):
    """Full schedule: readout after every application (= prefix(j)); per-token NLL change vs. gate and update."""
    from scipy.stats import spearmanr
    cf = dict(ad.configs())
    G, U, DN, DIFF = [[] for _ in range(8)], [[] for _ in range(8)], [[] for _ in range(8)], []
    for ids, s in toks:
        if ids.numel() - s - 1 <= 0:
            continue
        x_ids = ids[None].to(dev)
        _, tr = ad._run(x_ids, cf["full"], trace=True)
        y = x_ids[0, s + 1:]
        nlls = []
        for j in range(len(tr)):
            lg = ad.readout(tr[j][0])[0, s:-1]
            nlls.append(F.cross_entropy(lg, y, reduction="none"))
        full_nll = nlls[-1]
        for j in range(1, 9):
            xj, xp = tr[j][0][0, s:-1].float(), tr[j - 1][0][0, s:-1].float()
            U[j - 1].append(((xj - xp).norm(dim=-1) / xp.norm(dim=-1).clamp_min(1e-6)).cpu())
            G[j - 1].append(tr[j][1][0, s:-1].float().reshape(-1).cpu())
            DN[j - 1].append((nlls[j - 1] - nlls[j]).float().cpu())
        DIFF.append(full_nll.float().cpu())
    diff = torch.cat(DIFF).numpy()
    res = []
    for j in range(8):
        g, u, dn = (torch.cat(v[j]).numpy() for v in (G, U, DN))
        res.append({"application": j + 1, "mean_gate": float(g.mean()), "mean_update": float(u.mean()),
                    "mean_dnll": float(dn.mean()),
                    "rho_gate_dnll": float(spearmanr(g, dn).correlation),
                    "rho_update_dnll": float(spearmanr(u, dn).correlation),
                    "rho_gate_difficulty": float(spearmanr(g, diff).correlation),
                    "rho_dnll_difficulty": float(spearmanr(dn, diff).correlation)})
        print(res[-1], flush=True)
    (OUT / "gate_nll.json").write_text(json.dumps(res, indent=1))


# ----------------------------------------------------------------------------- calibration-set size
def calsize_part(ad, toks_eval, toks_cal):
    cf = dict(ad.configs())
    W = ad.m.output.weight.float()
    head = ad.head
    data = {}
    for n in ("prefix1", "full"):
        Ze, _, Ye, _ = collect(ad, toks_eval, cf[n], want_h=False)
        Zc, _, Yc, Dc = collect(ad, toks_cal, cf[n], want_h=False)
        data[n] = (Ze, Ye, Zc, Yc, Dc)
    raw = {n: float(nll_of(W, data[n][0], data[n][1]).mean()) for n in data}
    gap = raw["prefix1"] - raw["full"]
    rng = np.random.default_rng(0)
    res = []
    for ndoc in (5, 10, 25, 50, 100, 200):
        reps = 1 if ndoc == 200 else 5
        for r in range(reps):
            docs = rng.choice(200, ndoc, replace=False) if ndoc < 200 else np.arange(200)
            rec = {"n_docs": int(ndoc), "rep": r}
            nll_T, nll_A = {}, {}
            for n in data:
                Ze, Ye, Zc, Yc, Dc = data[n]
                m = torch.isin(Dc, torch.tensor(docs))
                Zs, Ys = Zc[m].to(dev), Yc[m].to(dev)
                T = DP.fit_T(head, Zs, Ys)
                rec[f"T_{n}"] = T
                rec[f"tokens"] = int(m.sum())
                nll_T[n] = float(nll_of(W, Ze, Ye, T=T).mean())
                if ndoc >= 25:
                    a, b, lam = DP.fit_affine(head, Zs[:12000], Ys[:12000])
                    nll_A[n] = float(nll_of(W, Ze, Ye, adapter=lambda z, d: z * a + b).mean())
            rec["share_T"] = (gap - (nll_T["prefix1"] - nll_T["full"])) / gap
            if nll_A:
                rec["share_affine"] = (gap - (nll_A["prefix1"] - nll_A["full"])) / gap
            res.append(rec)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
    (OUT / "calsize.json").write_text(json.dumps({"gap": gap, "raw": raw, "runs": res}, indent=1))


# ----------------------------------------------------------------------------- lightweight readouts
class Adapter(nn.Module):
    """z' = a_k * z + b_k + V(U z + e_k): per-feature affine plus a rank-r residual. per_depth: separate
    parameters per depth; depth-conditioned: shared U, V with a learned depth embedding e_k and per-depth a_k, b_k
    disabled (shared a, b)."""

    def __init__(self, d, depths, rank=64, per_depth=True):
        super().__init__()
        self.depths = list(depths)
        self.per_depth = per_depth
        nd = len(self.depths) if per_depth else 1
        self.a = nn.Parameter(torch.ones(nd, d))
        self.b = nn.Parameter(torch.zeros(nd, d))
        self.U = nn.Parameter(torch.randn(nd, rank, d) * d ** -0.5)
        self.V = nn.Parameter(torch.zeros(nd, d, rank))
        self.e = nn.Parameter(torch.zeros(len(self.depths), rank))

    def forward(self, z, depth):
        k = self.depths.index(depth)
        i = k if self.per_depth else 0
        r = z @ self.U[i].T
        if not self.per_depth:
            r = r + self.e[k]
        return z * self.a[i] + self.b[i] + r @ self.V[i].T


def readouts_part(ad, toks_eval, toks_cal, epochs=4):
    cf = dict(ad.configs())
    W = ad.m.output.weight.float().detach()
    depths = ["prefix1", "prefix2", "prefix4", "full"]
    D = {n: (collect(ad, toks_cal, cf[n], want_h=False), collect(ad, toks_eval, cf[n], want_h=False)) for n in depths}
    raw = {n: float(nll_of(W, D[n][1][0], D[n][1][2]).mean()) for n in depths}
    res = {"raw": raw}
    torch.manual_seed(0)

    def train(model, names):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        for ep in range(epochs):
            order = []
            for n in names:
                Zc, _, Yc, _ = D[n][0]
                perm = torch.randperm(Yc.numel())
                order += [(n, perm[i:i + 2048]) for i in range(0, Yc.numel(), 2048)]
            for j in torch.randperm(len(order)).tolist():
                n, idx = order[j]
                Zc, _, Yc, _ = D[n][0]
                z = Zc[idx].to(dev).float()
                loss = F.cross_entropy(F.linear(model(z, n), W), Yc[idx].to(dev))
                opt.zero_grad(); loss.backward(); opt.step()
        return model

    per = {}
    for n in depths:
        mdl = train(Adapter(W.shape[1], [n], per_depth=True).to(dev), [n])
        per[n] = float(nll_of(W, D[n][1][0], D[n][1][2], adapter=mdl, depth=n).mean())
        print("per-depth", n, raw[n], "->", per[n], flush=True)
    cond = train(Adapter(W.shape[1], depths, per_depth=False).to(dev), depths)
    dc = {n: float(nll_of(W, D[n][1][0], D[n][1][2], adapter=cond, depth=n).mean()) for n in depths}
    print("depth-conditioned", dc, flush=True)
    gap = {n: raw[n] - raw["full"] for n in depths}
    res.update({"per_depth": per, "depth_conditioned": dc,
                "recovered_share_per_depth": {n: (raw[n] - per[n] - (raw["full"] - per["full"])) / gap[n] for n in depths[:-1]},
                "recovered_share_depth_conditioned": {n: (raw[n] - dc[n] - (raw["full"] - dc["full"])) / gap[n] for n in depths[:-1]}})
    (OUT / "readouts.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="geometry,dynamics,gate_nll,calsize,readouts")
    ap.add_argument("--ckpt", default="<checkpoint>.pt")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    tok, m, step = S.load(args.ckpt, dev)
    ad = S.SonaAdapter(m, tok)
    te, tc = load_tokens(tok, "eval"), load_tokens(tok, "cal")
    t0 = time.time()
    for part in args.parts.split(","):
        print(f"=== {part} [{time.time() - t0:.0f}s]", flush=True)
        {"geometry": lambda: geometry_part(ad, te),
         "dynamics": lambda: dynamics_part(ad, te),
         "gate_nll": lambda: gate_nll_part(ad, te),
         "calsize": lambda: calsize_part(ad, te, tc),
         "readouts": lambda: readouts_part(ad, te, tc)}[part]()
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
