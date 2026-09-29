# Depth Control Protocol (DCP)

Code for **"Beyond Depth Truncation: Controlled Evaluation of Depth Utilization in Recursive Language Models"**
([arXiv:2609.19934](https://arxiv.org/abs/2609.19934)).

Truncating a recurrent language model's depth at inference and reading off the slope of quality versus depth is the
standard way to ask whether the model uses its depth. The slope mixes several things: how many block applications
write into the residual stream, how many *distinct* iterations run, and whether the readout, trained at full depth,
can decode a truncated residual stream. DCP separates these with controlled contrasts:

| Control | What it changes |
|---|---|
| **prefix($k$)** | the first $k$ applications (naive truncation) |
| **repeat($k$)** | the first $k$ applications, then application $k$ again until the full budget: fixed application count, $k$ distinct iterations |
| **suffix($k$)** | the last $k$ applications: which iterations run, at a fixed count |
| **temperature / affine recalibration** | how much of the gap is correctable at the readout, fitted on calibration documents disjoint from evaluation documents |

## Repository layout

```
public/      DCP on public models (Appendix C of the paper) -- fully runnable
  build_data.py      disjoint calibration / evaluation splits (MATH solutions, FineWeb); the splits used are in data/
  dcp_public.py      prefix / suffix / repeat schedules (dense) or recurrence depths (Huginn-0125); temperature
                     (same-domain, cross-domain, in-sample) and affine readout fits; ECE, Brier, residual geometry,
                     per-application update norms; per-token outputs
  run_step2.sh       run several models on one GPU, with resume
  analyze.py         paired document bootstrap for every component -> out/analysis.json, out/tables.tex
  make_appendix.py   LaTeX tables of the appendix
  out/               analysis.json and per-model summary.json for all reported runs
recurrent/   DCP scripts for the 542.8M recurrent model and the 90M calibration grid (Sections 4-8)
```

## Quick start (public models)

```bash
pip install -r requirements.txt
cd public
python build_data.py                     # optional: data/ already contains the exact splits used
python dcp_public.py --model Qwen/Qwen2.5-Math-1.5B --kind dense --tag Qwen2.5-Math-1.5B
python dcp_public.py --model tomg-group-umd/huginn-0125 --kind huginn --tag huginn-0125
python analyze.py && python make_appendix.py
```

A dense 1.5B model takes about one hour on one RTX 3090 (16 configurations x 800 documents); Huginn-0125 up to
$r = 64$ takes about two hours. For Huginn under transformers 5.x, `dcp_public.py` includes a compatibility shim; if
flex attention fails to compile, add the `nvidia/cu13/lib` directory of your environment to `LD_LIBRARY_PATH`
(see `run_step2.sh`).

Every calibration map is fitted on calibration documents disjoint from the evaluation documents, and raw and
calibrated NLL are always compared on exactly the same tokens.

## Main public-model results (mathematical text)

Share of the naive truncation gap that is correctable at the readout (95% paired document-bootstrap CI in
`public/out/analysis.json`):

| Model | Truncation | Gap (nats) | Scalar temperature | Affine readout |
|---|---|---|---|---|
| Huginn-0125 (recurrent, sampled depth) | $r = 1$ vs. 32 | 1.717 | 0.5% | 5.8% |
| Qwen2.5-Math-1.5B | 26/28 layers | 0.786 | 18.7% | 50.9% |
| Qwen3-1.7B | 26/28 | 0.557 | 39.2% | 60.1% |
| Qwen2.5-1.5B / -Instruct | 26/28 | 0.844 / 0.864 | 27.4% / 26.3% | 56.0% / 56.9% |
| SmolLM2-1.7B | 22/24 | 2.307 | 14.0% | 75.4% |
| Pythia-1.4B | 22/24 | 0.714 | 0.0% | 61.6% |
| OPT-350m | 22/24 | 0.985 | 0.9% | 47.1% |

## Recurrent-model scripts

The scripts in `recurrent/` document the exact execution schedules (`build_plan` in `exp_depth_ablation.py`, which
keeps the original global iteration index for repeated applications) and the measurement procedures used for the
542.8M model, its continued-training interventions and the 90M calibration grid. They import the model definition
(`model/vera_psi.py`) and training code, which are not included in this repository.

## Citation

```bibtex
@article{dau2026beyond,
  title   = {Beyond Depth Truncation: Controlled Evaluation of Depth Utilization in Recursive Language Models},
  author  = {Dau, Ha Van and Khuat, Thanh Tung and Nguyen, Thanh Dung},
  journal = {arXiv preprint arXiv:2609.19934},
  year    = {2026}
}
```

## License

MIT (see `LICENSE`).
