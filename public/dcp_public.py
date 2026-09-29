#!/usr/bin/env python3
"""
dcp_public.py — Depth Control Protocol on public causal LMs (no Sona weights needed).

Addresses the reviewer requests that can be answered on public models:
  * calibration and evaluation documents are DISJOINT (math_cal vs math_eval, web_cal vs web_eval),
    and every calibrated NLL is measured on exactly the same token set as the raw NLL;
  * calibration metrics beyond NLL (ECE, Brier, entropy, logit norm, top-1 confidence);
  * richer recalibration than a scalar T: a per-feature affine map on the readout input
    (z' = a*z + b, i.e. refitting the final-norm gain/bias = a shallow depth-specific readout);
  * cross-domain transfer of the depth-specific temperature (fit on math, apply to web and vice versa);
  * in-sample vs held-out temperature (optimism of fitting T on the evaluation set);
  * direct measurement of the residual stream reaching the readout (norms, cosine / CKA to full
    depth, effective rank, anisotropy, Frechet distance);
  * per-application update norms (dynamics of repeated layers);
  * depth extrapolation beyond the training depth (Huginn, r > 32);
  * per-token outputs saved to disk so that bootstrap intervals, difficulty and token-position
    analyses are computed offline (analyze.py) without re-running models.

Configurations
  dense  : prefix k / suffix k / repeat k over the L decoder layers (repeat = first k layers, then
           layer k again until L applications), k in round(frac * L).
  huginn : num_steps r (prefix == repeat for a single weight-tied core).
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

HERE = Path(__file__).parent
DOMAINS = ("math", "web")


# ----------------------------------------------------------------------------- data
def load_split(domain, split):
    return [json.loads(l) for l in open(HERE / "data" / f"{domain}_{split}.jsonl")]


def tokenize(tok, docs, max_len, bos):
    """Returns list of (ids LongTensor, logit index predicting the first scored token).
    math: only solution tokens are scored; web: every token after the first."""
    out = []
    for d in docs:
        pre = bos + (tok(d["prompt"], add_special_tokens=False).input_ids if d["prompt"] else [])
        t = tok(d["text"], add_special_tokens=False).input_ids
        ids = (pre + t)[:max_len]
        start = len(pre) - 1 if d["prompt"] else 0
        out.append((torch.tensor(ids), start))
    return out


# ----------------------------------------------------------------------------- model adapters
def install_legacy_tied_weights_shim():
    """Huginn's remote code declares _tied_weights_keys as a list; transformers>=5 wants a dict."""
    import transformers.modeling_utils as MU
    orig = MU.PreTrainedModel.get_expanded_tied_weights_keys

    def patched(self, all_submodels=False):
        tw = getattr(self, "_tied_weights_keys", None)
        if isinstance(tw, (list, tuple)):
            src = None
            emb = self.get_input_embeddings()
            for n, p in self.named_parameters(remove_duplicate=False):
                if emb is not None and p is emb.weight:
                    src = n
                    break
            self._tied_weights_keys = {k: (src or k) for k in tw}
        return orig(self, all_submodels=all_submodels)

    MU.PreTrainedModel.get_expanded_tied_weights_keys = patched


def get_attr(obj, dotted):
    for a in dotted.split("."):
        obj = getattr(obj, a, None)
        if obj is None:
            return None
    return obj


class Adapter:
    """Runs one model under a named execution configuration and captures the readout input."""

    def __init__(self, model, kind):
        self.m, self.kind = model, kind
        self.head = model.get_output_embeddings()
        self.cap = {}
        self.head.register_forward_pre_hook(self._cap_z)
        # module whose INPUT is the pre-norm residual stream
        self.prenorm = None
        for name in ("transformer.ln_f", "model.norm", "gpt_neox.final_layer_norm",
                     "model.decoder.final_layer_norm", "model.decoder.project_out"):
            mod = get_attr(model, name)
            if isinstance(mod, nn.Module):
                self.prenorm = mod
                break
        if self.prenorm is not None:
            self.prenorm.register_forward_pre_hook(self._cap_h)
        self.dyn = None  # list of per-application relative updates, when enabled
        if kind == "dense":
            for name in ("model.layers", "gpt_neox.layers", "model.decoder.layers"):
                par, attr = name.rsplit(".", 1)
                if isinstance(get_attr(model, name), nn.ModuleList):
                    self.parent, self.attr = get_attr(model, par), attr
                    break
            self.orig = list(getattr(self.parent, self.attr))
            self.L = len(self.orig)
            for layer in self.orig:
                layer.register_forward_pre_hook(self._dyn_pre, with_kwargs=True)
                layer.register_forward_hook(self._dyn_post, with_kwargs=True)

    # hooks -----------------------------------------------------------------
    def _cap_z(self, mod, args):
        self.cap["z"] = args[0]

    def _cap_h(self, mod, args):
        self.cap["h"] = args[0]

    @staticmethod
    def _hidden(args, kwargs):
        x = args[0] if args else kwargs.get("hidden_states")
        return x[0] if isinstance(x, (tuple, list)) else x

    def _dyn_pre(self, mod, args, kwargs):
        if self.dyn is not None:
            self._last_in = self._hidden(args, kwargs).detach()

    def _dyn_post(self, mod, args, kwargs, out):
        if self.dyn is not None:
            o = out[0] if isinstance(out, (tuple, list)) else out
            x = self._last_in.float()
            rel = ((o.float() - x).norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-6)).mean().item()
            self.dyn.append(rel)

    # configurations --------------------------------------------------------
    def set_plan(self, plan):
        setattr(self.parent, self.attr, nn.ModuleList([self.orig[i] for i in plan]))
        self.m.config.num_hidden_layers = len(plan)
        if hasattr(self.m.config, "get_text_config"):
            self.m.config.get_text_config().num_hidden_layers = len(plan)

    def configs(self, fracs, huginn_steps):
        if self.kind == "huginn":
            return [(f"r{r}", {"steps": r}) for r in huginn_steps]
        ks = sorted({max(1, round(f * self.L)) for f in fracs})
        idx = list(range(self.L))
        cf = [("full", {"plan": idx})]
        for k in ks:
            if k == self.L:
                continue
            cf.append((f"prefix{k}", {"plan": idx[:k]}))
            cf.append((f"suffix{k}", {"plan": idx[self.L - k:]}))
            cf.append((f"repeat{k}", {"plan": idx[:k] + [k - 1] * (self.L - k)}))
        return cf

    @torch.no_grad()
    def run(self, ids, cfg):
        self.cap.clear()
        if self.kind == "huginn":
            out = self.m(input_ids=ids, num_steps=int(cfg["steps"]))
        else:
            out = self.m(input_ids=ids, use_cache=False)
        z = self.cap["z"][0]
        h = self.cap.get("h", self.cap["z"])[0]
        return out.logits[0], z, h


# ----------------------------------------------------------------------------- calibration
_WF = {}


def head_logits(head, z, a=None, b=None):
    if a is not None:
        if id(head) not in _WF:
            _WF[id(head)] = (head.weight.float(), None if head.bias is None else head.bias.float())
        W, bias = _WF[id(head)]
        return F.linear(z.float() * a + b, W, bias)
    return head(z.to(head.weight.dtype)).float()


def fit_T(head, Z, Y, chunk=2048, max_iter=100):
    logT = torch.zeros((), device=Z.device, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.5, max_iter=max_iter, tolerance_grad=1e-9,
                            tolerance_change=1e-12, line_search_fn="strong_wolfe")
    n = Y.numel()

    def closure():
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for i in range(0, n, chunk):
            with torch.no_grad():
                lg = head_logits(head, Z[i:i + chunk])
            loss = F.cross_entropy(lg / logT.exp(), Y[i:i + chunk], reduction="sum") / n
            loss.backward()
            tot += loss.item()
        return torch.tensor(tot)

    opt.step(closure)
    return float(min(logT.detach().exp(), 1e3))  # T -> inf means "uniform is best"; cap for reporting


def _fit_affine_once(head, Z, Y, lam, chunk=2048, max_iter=20):
    d = Z.shape[1]
    a = torch.ones(d, device=Z.device, requires_grad=True)
    b = torch.zeros(d, device=Z.device, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=1.0, max_iter=max_iter, history_size=20,
                            line_search_fn="strong_wolfe")
    n = Y.numel()

    def closure():
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for i in range(0, n, chunk):
            loss = F.cross_entropy(head_logits(head, Z[i:i + chunk], a, b), Y[i:i + chunk],
                                   reduction="sum") / n
            loss.backward()
            tot += loss.item()
        reg = lam * ((a - 1).pow(2).sum() + b.pow(2).sum())
        reg.backward()
        return torch.tensor(tot + reg.item())

    opt.step(closure)
    return a.detach(), b.detach()


def fit_affine(head, Z, Y, lams=(1e-1, 1e-2, 1e-3)):
    """Ridge-regularised (towards identity) per-feature affine map on the readout input.
    lambda is selected on a 20% split of the calibration tokens, then refit on all of them."""
    n = Y.numel()
    cut = int(0.8 * n)
    best, best_lam = None, lams[0]
    for lam in lams:
        a, b = _fit_affine_once(head, Z[:cut], Y[:cut], lam)
        with torch.no_grad():
            v = sum(F.cross_entropy(head_logits(head, Z[i:i + 1024], a, b), Y[i:i + 1024],
                                    reduction="sum").item() for i in range(cut, n, 1024))
        if best is None or v < best:
            best, best_lam = v, lam
    a, b = _fit_affine_once(head, Z, Y, best_lam)
    return a, b, best_lam


@torch.no_grad()
def token_stats(head, Z, Y, T=1.0, a=None, b=None, full=False, chunk=1024):
    res = {k: [] for k in ("nll", "conf", "correct", "entropy", "brier", "lnorm")}
    for i in range(0, Y.numel(), chunk):
        lg = head_logits(head, Z[i:i + chunk], a, b)
        y = Y[i:i + chunk]
        if full:
            res["lnorm"].append(lg.norm(dim=-1).cpu())
        lg = lg / T
        lp = lg.log_softmax(-1)
        res["nll"].append((-lp.gather(1, y[:, None])[:, 0]).cpu())
        if full:
            p = lp.exp()
            conf, am = p.max(-1)
            res["conf"].append(conf.cpu())
            res["correct"].append((am == y).cpu())
            res["entropy"].append((-(p * lp).sum(-1)).cpu())
            res["brier"].append((1 - 2 * p.gather(1, y[:, None])[:, 0] + (p * p).sum(-1)).cpu())
    return {k: torch.cat(v).numpy() for k, v in res.items() if v}


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


# ----------------------------------------------------------------------------- residual geometry
def geometry(H, Href, Z, Zref):
    """H/Z: [n, d] features at this configuration; *ref: same tokens at full depth."""
    H, Href, Z, Zref = (x.float() for x in (H, Href, Z, Zref))

    def eff_rank(X):
        X = X - X.mean(0)
        ev = torch.linalg.eigvalsh(X.T @ X / X.shape[0]).clamp_min(0)
        return float(ev.sum() ** 2 / (ev ** 2).sum())

    def cka(X, Y):
        X, Y = X - X.mean(0), Y - Y.mean(0)
        return float((X.T @ Y).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm()))

    def aniso(X, n=2000):
        g = torch.Generator().manual_seed(0)
        i = torch.randint(0, X.shape[0], (n,), generator=g)
        j = torch.randint(0, X.shape[0], (n,), generator=g)
        return float(F.cosine_similarity(X[i], X[j], dim=-1).mean())

    def frechet(X, R, k=64):
        mu = R.mean(0)
        U, S, Vh = torch.linalg.svd(R - mu, full_matrices=False)
        P = Vh[:k].T
        x, r = (X - mu) @ P, (R - mu) @ P
        m1, m2 = x.mean(0), r.mean(0)
        C1, C2 = torch.cov(x.T), torch.cov(r.T)
        e1, V1 = torch.linalg.eigh(C1)
        s1 = V1 @ torch.diag(e1.clamp_min(0).sqrt()) @ V1.T
        M = s1 @ C2 @ s1
        tr_sqrt = torch.linalg.eigvalsh(M).clamp_min(0).sqrt().sum()
        fd = (m1 - m2).pow(2).sum() + torch.trace(C1) + torch.trace(C2) - 2 * tr_sqrt
        return float(fd / torch.trace(C2))  # normalised by reference total variance

    return {
        "h_norm": float(H.norm(dim=-1).mean()), "h_norm_ref": float(Href.norm(dim=-1).mean()),
        "z_norm": float(Z.norm(dim=-1).mean()), "z_norm_ref": float(Zref.norm(dim=-1).mean()),
        "h_cos_to_full": float(F.cosine_similarity(H, Href, dim=-1).mean()),
        "z_cos_to_full": float(F.cosine_similarity(Z, Zref, dim=-1).mean()),
        "h_cka_to_full": cka(H, Href), "z_cka_to_full": cka(Z, Zref),
        "h_eff_rank": eff_rank(H), "h_eff_rank_ref": eff_rank(Href),
        "h_anisotropy": aniso(H), "h_anisotropy_ref": aniso(Href),
        "h_frechet_norm": frechet(H, Href), "z_frechet_norm": frechet(Z, Zref),
    }


# ----------------------------------------------------------------------------- main loop
def collect(ad, cfg, toks, device, store_h_idx=None):
    """Forward all docs; return readout inputs Z (bf16, cpu), targets Y, doc ids, positions, H subsample."""
    Zs, Ys, D, P, Hs = [], [], [], [], []
    for di, (ids, start) in enumerate(toks):
        x = ids[None].to(device)
        logits, z, h = ad.run(x, cfg)
        z, h = z[start:-1], h[start:-1]
        y = ids[start + 1:]
        Zs.append(z.to(torch.bfloat16).cpu()); Ys.append(y)
        Hs.append(h.to(torch.bfloat16).cpu())
        D.append(torch.full((y.numel(),), di, dtype=torch.int32))
        P.append(torch.arange(start + 1, start + 1 + y.numel(), dtype=torch.int32))
        if di == 0:  # readout-capture self-test: lm_head(z) must reproduce the model's logits
            ref = logits[start:-1].float()
            mine = head_logits(ad.head, z)
            ad.selftest = max(getattr(ad, "selftest", 0.0), float((ref - mine).abs().max().detach()))
    Z, Y = torch.cat(Zs), torch.cat(Ys)
    H = torch.cat(Hs)
    Hsub = H[store_h_idx] if store_h_idx is not None else None
    return Z, Y, torch.cat(D), torch.cat(P), Hsub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", default="dense", choices=["dense", "huginn"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fracs", default="0.25,0.5,0.75,0.86,0.93,1.0")
    ap.add_argument("--steps", default="1,2,4,8,16,32,48,64")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--n_docs", type=int, default=200)
    ap.add_argument("--cal_tokens", type=int, default=50000)
    ap.add_argument("--aff_tokens", type=int, default=12000)
    ap.add_argument("--geo_tokens", type=int, default=4096)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True  # fp32 matmuls of the affine readout fits
    out = HERE / "out" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.kind == "huginn":
        install_legacy_tied_weights_shim()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
                                                 dtype=getattr(torch, args.dtype)).to(device).eval()
    ad = Adapter(model, args.kind)
    # prepend BOS exactly when the tokenizer does so by default
    first = tok("a").input_ids[0]
    bos = [first] if tok.bos_token_id is not None and first == tok.bos_token_id else []
    data = {(d, s): tokenize(tok, load_split(d, s)[:args.n_docs], args.max_len, bos)
            for d in DOMAINS for s in ("eval", "cal")}
    ntok = {k: sum(len(i) - s - 1 for i, s in v) for k, v in data.items()}
    print(f"{args.model} kind={args.kind} tokens={ntok}", flush=True)

    cfgs = ad.configs([float(f) for f in args.fracs.split(",")],
                      [int(s) for s in args.steps.split(",")])
    # reference configuration for geometry: full depth (dense) / training mean depth r=32 (huginn)
    ref_name = "full" if args.kind == "dense" else "r32"
    cfgs = sorted(cfgs, key=lambda c: c[0] != ref_name)
    rng = np.random.default_rng(0)
    geo_idx = {d: torch.tensor(np.sort(rng.choice(ntok[(d, "eval")],
                                                  min(args.geo_tokens, ntok[(d, "eval")]),
                                                  replace=False))) for d in DOMAINS}
    ref = {}
    summary = {"model": args.model, "kind": args.kind, "tokens": {f"{a}_{b}": v for (a, b), v in ntok.items()},
               "configs": {}}
    if (out / "summary.json").exists():  # resume: keep finished configurations
        summary["configs"] = json.loads((out / "summary.json").read_text()).get("configs", {})

    for name, cfg in cfgs:
        if name != ref_name and name in summary["configs"] and (out / f"{name}.npz").exists():
            print(f"[{name:>9}] already done, skipping", flush=True)
            continue
        if args.kind == "dense":
            ad.set_plan(cfg["plan"])
        tc = time.time()
        # dynamics on 20 math eval docs
        ad.dyn = [] if args.kind == "dense" else None
        if ad.dyn is not None:
            per_doc = []
            for ids, s in data[("math", "eval")][:20]:
                ad.dyn = []
                ad.run(ids[None].to(device), cfg)
                per_doc.append(ad.dyn)
            dyn = np.mean(np.array(per_doc), 0).tolist()
        else:
            dyn = None
        ad.dyn = None

        feats = {}
        for d in DOMAINS:
            for s in ("eval", "cal"):
                Z, Y, D, P, Hs = collect(ad, cfg, data[(d, s)], device,
                                         geo_idx[d] if s == "eval" else None)
                feats[(d, s)] = (Z, Y, D, P, Hs)
        # calibrators (fit on disjoint cal docs of each domain; plus in-sample on eval)
        head = ad.head
        cal = {}
        for d in DOMAINS:
            Z, Y = feats[(d, "cal")][:2]
            n = min(args.cal_tokens, Y.numel())
            Zc, Yc = Z[:n].to(device), Y[:n].to(device)
            cal[f"T_{d}"] = fit_T(head, Zc, Yc)
            na = min(args.aff_tokens, n)
            a_, b_, lam_ = fit_affine(head, Zc[:na], Yc[:na])
            cal[f"aff_{d}"] = (a_, b_)
            cal[f"lam_{d}"] = lam_
            Ze, Ye = feats[(d, "eval")][:2]
            cal[f"Tin_{d}"] = fit_T(head, Ze.to(device), Ye.to(device))
            del Zc, Yc
        rec = {"n_applications": len(cfg.get("plan", [])) or None,
               "distinct": len(set(cfg.get("plan", []))) or None,
               "T": {d: cal[f"T_{d}"] for d in DOMAINS},
               "T_insample": {d: cal[f"Tin_{d}"] for d in DOMAINS},
               "affine_lambda": {d: cal[f"lam_{d}"] for d in DOMAINS},
               "update_norms": dyn, "domains": {}}
        arrays = {}
        for d in DOMAINS:
            other = "web" if d == "math" else "math"
            Z, Y, D, P, Hs = feats[(d, "eval")]
            Zg, Yg = Z.to(device), Y.to(device)
            raw = token_stats(head, Zg, Yg, full=True)
            tcal = token_stats(head, Zg, Yg, T=cal[f"T_{d}"], full=True)
            tx = token_stats(head, Zg, Yg, T=cal[f"T_{other}"])
            tin = token_stats(head, Zg, Yg, T=cal[f"Tin_{d}"])
            a, b = cal[f"aff_{d}"]
            aff = token_stats(head, Zg, Yg, a=a, b=b, full=True)
            del Zg, Yg
            r = {"nll": float(raw["nll"].mean()), "nll_T": float(tcal["nll"].mean()),
                 "nll_T_cross": float(tx["nll"].mean()), "nll_T_insample": float(tin["nll"].mean()),
                 "nll_affine": float(aff["nll"].mean()),
                 "acc": float(raw["correct"].mean()),
                 "ece": ece(raw["conf"], raw["correct"]), "ece_T": ece(tcal["conf"], tcal["correct"]),
                 "ece_affine": ece(aff["conf"], aff["correct"]),
                 "brier": float(raw["brier"].mean()), "brier_T": float(tcal["brier"].mean()),
                 "entropy": float(raw["entropy"].mean()), "conf": float(raw["conf"].mean()),
                 "logit_norm": float(raw["lnorm"].mean()), "n_tokens": int(Y.numel())}
            zsub = Z[geo_idx[d]]
            if name == ref_name:
                ref[d] = (Hs, zsub)
            r["geometry"] = geometry(Hs, ref[d][0], zsub, ref[d][1])
            rec["domains"][d] = r
            for k, v in (("raw", raw), ("T", tcal), ("Tx", tx), ("Tin", tin), ("aff", aff)):
                for kk, vv in v.items():
                    arrays[f"{d}_{k}_{kk}"] = vv.astype(np.float32 if kk == "nll" else np.float16)
            arrays[f"{d}_doc"] = D.numpy()
            arrays[f"{d}_pos"] = P.numpy()
        np.savez_compressed(out / f"{name}.npz", **arrays)
        summary["configs"][name] = rec
        summary["selftest_max_abs_logit_diff"] = getattr(ad, "selftest", None)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        m = rec["domains"]["math"]
        w = rec["domains"]["web"]
        print(f"[{name:>9}] math nll {m['nll']:.4f} T {rec['T']['math']:.3f} nll_T {m['nll_T']:.4f} "
              f"aff {m['nll_affine']:.4f} ece {m['ece']:.3f} | web nll {w['nll']:.4f} "
              f"T {rec['T']['web']:.3f} | zcos {m['geometry']['z_cos_to_full']:.3f} "
              f"({time.time() - tc:.0f}s, total {time.time() - t0:.0f}s)", flush=True)
    print("selftest max|logit diff| =", getattr(ad, "selftest", None))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
