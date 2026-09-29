#!/usr/bin/env python3
"""
calibration.py — exact temperature fitting, shared by every depth experiment.

Replaces the coarse log-space grid used in the first round of measurements.
That grid had 41 points over log T in [-1, 1], i.e. a spacing of 0.05, so a
reported "temperature drift of 0.049" was exactly one grid step and could not
be distinguished from zero. Claims about how flat a temperature curve is need
a fit whose resolution is far below the effect being reported.

Method: minimise token-level NLL over a single scalar log T by L-BFGS with
strong-Wolfe line search (the standard temperature-scaling procedure, Guo et
al. 2017). Converges to ~1e-6 in log T, three orders of magnitude finer than
the previous grid.

The optimisation is chunked so that a 150k-vocabulary model does not
materialise the whole logit tensor in fp32 at once.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def _nll_at(L, Y, T, chunk):
    tot, n = 0.0, 0
    for i in range(0, L.shape[0], chunk):
        lg = L[i:i + chunk].float() / T
        tot += F.cross_entropy(lg, Y[i:i + chunk], reduction="sum").item()
        n += lg.shape[0]
    return tot / max(n, 1)


def _fit_logT(L, Y, chunk, max_iter, idx=None):
    """L-BFGS fit of log T on rows `idx` (all rows if None). Returns T."""
    dev = L.device
    n_tok = int(Y.numel() if idx is None else idx.numel())
    log_T = torch.zeros((), device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.5, max_iter=max_iter,
                            tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")
    rows = torch.arange(L.shape[0], device=dev) if idx is None else idx

    def closure():
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for i in range(0, rows.numel(), chunk):
            r = rows[i:i + chunk]
            lg = L[r].float() / log_T.exp()
            loss = F.cross_entropy(lg, Y[r], reduction="sum") / n_tok
            loss.backward()
            total += float(loss.detach())
        return torch.tensor(total, device=dev)

    opt.step(closure)
    return float(log_T.detach().exp())


def fit_temperature(cal_logits, cal_tgts, device="cuda", chunk=256,
                    max_iter=200, n_boot=0, seed=0):
    """Fit a single logit temperature by L-BFGS.

    cal_logits : list of [t_i, V] tensors (any dtype, any device)
    cal_tgts   : list of [t_i] integer tensors
    returns dict with the fitted temperature, calibrated NLL, uncalibrated
    NLL on the same subset, and diagnostics for reproducibility.
    """
    if not cal_tgts:
        return {}
    dev = device if torch.cuda.is_available() else "cpu"
    # keep the store in bf16; each chunk is cast to fp32 inside the closure
    L = torch.cat([l.to(torch.bfloat16) for l in cal_logits]).to(dev)
    Y = torch.cat([t.long() for t in cal_tgts]).to(dev)
    n_tok = int(Y.numel())

    T = _fit_logT(L, Y, chunk, max_iter)
    out = {
        "cal_temperature": T,
        "cal_nll": _nll_at(L, Y, T, chunk),
        "cal_subset_raw_nll": _nll_at(L, Y, 1.0, chunk),
        "cal_subset_tokens": n_tok,
        "cal_method": "lbfgs-strong-wolfe on log T",
    }
    # Bootstrap the temperature itself. The four-model comparison in the paper
    # contrasts temperature RANGES across models, so that quantity needs an
    # interval of its own rather than a bare point estimate.
    if n_boot:
        g = torch.Generator(device="cpu").manual_seed(seed)
        n = L.shape[0]
        Ts = []
        for _ in range(n_boot):
            idx = torch.randint(0, n, (n,), generator=g).to(dev)
            Ts.append(_fit_logT(L, Y, chunk, max_iter, idx=idx))
        Ts = torch.tensor(Ts)
        lo, hi = torch.quantile(Ts, torch.tensor([0.025, 0.975]))
        out.update({"cal_T_ci_lo": float(lo), "cal_T_ci_hi": float(hi),
                    "cal_T_boot_sd": float(Ts.std()), "cal_T_n_boot": n_boot})
    del L, Y
    if dev != "cpu":
        torch.cuda.empty_cache()
    return out


def bootstrap_nll(per_doc_nll, per_doc_tok, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for a token-weighted NLL, resampling DOCUMENTS.

    Every NLL in this paper is a ratio of summed losses to summed tokens over
    a document sample, so the relevant uncertainty is over which documents
    were drawn. Resampling documents (not tokens) respects that both are
    correlated within a document.
    """
    import numpy as _np
    a = _np.asarray(per_doc_nll, dtype=_np.float64)
    w = _np.asarray(per_doc_tok, dtype=_np.float64)
    n = len(a)
    if n < 2:
        return {}
    rng = _np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = a[idx].sum(axis=1) / w[idx].sum(axis=1)
    lo, hi = _np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"nll_ci_lo": float(lo), "nll_ci_hi": float(hi),
            "nll_boot_se": float(boots.std()), "n_docs": int(n),
            "n_boot": int(n_boot)}


if __name__ == "__main__":
    torch.manual_seed(0)
    V, N = 4096, 4000
    z = torch.randn(N, V) * 3.0

    # (1) RECOVERY. If labels are drawn from softmax(z / T_true), the NLL-
    #     optimal temperature for z is T_true. Recovering it validates the
    #     objective, not just the optimiser.
    print("recovery (labels ~ softmax(z / T_true)):")
    for T_true in (0.6, 1.0, 1.7, 2.5):
        y = torch.multinomial(torch.softmax(z / T_true, dim=-1), 1).squeeze(-1)
        got = fit_temperature([z], [y], device="cpu")
        err = abs(got["cal_temperature"] - T_true)
        print(f"  T_true={T_true:4.2f}  fitted={got['cal_temperature']:.4f}  "
              f"err={err:.3f}  {'ok' if err < 0.08 else 'FAIL'}")

    # (2) INVARIANCE. Fitting on pre-scaled logits z/c must return T*/c
    #     exactly. This is a deterministic check of optimiser precision, with
    #     no sampling noise.
    print("\ninvariance (fit(z/c) * c == fit(z)):")
    y = torch.multinomial(torch.softmax(z, dim=-1), 1).squeeze(-1)
    base = fit_temperature([z], [y], device="cpu")["cal_temperature"]
    for c in (0.5, 2.0, 4.0):
        got = fit_temperature([z / c], [y], device="cpu")["cal_temperature"]
        err = abs(got * c - base)
        print(f"  c={c:3.1f}  fit(z/c)*c={got*c:.6f}  fit(z)={base:.6f}  "
              f"err={err:.2e}  {'ok' if err < 1e-3 else 'FAIL'}")
    print(f"\nprevious grid resolution in T near 1.0: ~0.05  "
          f"| this fit: <1e-3")
