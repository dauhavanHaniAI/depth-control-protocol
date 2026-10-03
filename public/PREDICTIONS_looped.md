# Predictions for externally trained looped models (written before any measurement)

Written 2026-10-03, before any of the models below was downloaded or run. The file is hashed
(SHA-256 recorded in `PREDICTIONS_looped.sha256`) and committed before the first measurement.

## Models and why they test the depth-schedule account

| Family | Training depth schedule | Readout supervised at |
|---|---|---|
| IFM LoopedLM-P2 `huginn-{s,m,l}-fixed-r5` | fixed R = 5 | R = 5 only |
| IFM LoopedLM-P2 `huginn-{s,m,l}-fixed-pln5` | R ~ capped Poisson-lognormal, mean 5 | every sampled R |
| IFM LoopedLM-P2 `huginn-{s,m}-learned-entropy0p01` | learned depth prior, initialized at PLN-5 | every sampled R (exploratory) |
| ByteDance Ouro-1.4B, Ouro-2.6B | 4 loops; loss on every loop's readout, weighted by a learned exit distribution | loops 1-4 |

Within each IFM scale, `fixed-r5` and `fixed-pln5` share architecture, data, token budget and
optimizer; they differ in the depth schedule (and the truncated-backpropagation depth, 5 vs 10).
This is the matched comparison of our from-scratch grid (Branch A vs Branch B), trained by an
independent group at three scales.

## Protocol (fixed before measurement)

- Data: the same disjoint splits as Appendix C (`data/math_{eval,cal}.jsonl`, `data/web_{eval,cal}.jsonl`),
  200 evaluation and 200 calibration documents per domain, max 1,024 tokens, math scores solution tokens only.
- Depths: IFM r in {1,2,3,4,5,6,7,8,10}; Ouro r in {1,...,8}. Reference depth: the training depth
  (IFM: 5; Ouro: 4).
- Temperature: one scalar T_r per depth, fitted on calibration documents (L-BFGS on log T), applied
  to evaluation documents. Affine readout-input map as in Appendix C.
- Temperature-correctable share at r: [gap_raw(r) - gap_cal(r)] / gap_raw(r), gap(r) = Q(r) - Q(ref),
  on mathematical text; paired document bootstrap, 2,000 replicates.
- Drift amplitude: max_r T_r - min_r T_r over r in {1,...,ref}.
- Threshold for "component present": share >= 5% at r = 1 (the threshold fixed for the grid).

## Predictions

P1. IFM fixed-r5, at each scale: temperature-correctable share at r = 1 is >= 5%; |T_r - 1| > 0.1 for
    some r < 5; |T_5 - 1| < 0.05.
P2. IFM fixed-pln5, at each scale: share at r = 1 is < 5%, and its drift amplitude over r = 1..5 is
    smaller than that of fixed-r5 at the same scale.
P3. Ouro-1.4B and Ouro-2.6B: share at r = 1 (relative to r = 4) is < 5%; T_r in [0.9, 1.1] for r = 1..4.
P4. Outside the trained support: for IFM fixed-r5, |T_r - 1| at r in {6,7,8} exceeds |T_5 - 1|.
    For Ouro, max |T_r - 1| over r in {5,...,8} exceeds max |T_r - 1| over r in {1,...,4}.
    (Alternative that would also explain a failure of P4: the recurrence has converged to a
    fixed point, so states beyond the support barely change.)

No prediction is made for the learned-depth models, the web domain, or the affine map; those are
reported as exploratory. Whatever the outcome, every prediction above is reported with its result.
