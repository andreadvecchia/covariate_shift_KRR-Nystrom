# covshift-correct

Minimal, academic-friendly repo that reproduces Kernel Ridge Regression (KRR) baselines under **covariate shift**, comparing:

- **KRR** (vanilla)
- **IW–KRR** (importance-weighted)
- **IW–Nyström KRR** (importance-weighted with Nyström compression via BLESS)

> Extracted from a notebook and distilled into a tiny, well-structured package.

## Install

```bash
pip install -e .
# (optional for 3D plot) 
pip install -e .[plot]
```

## Quick run

Use the console entry-point to run a short comparison with cross-validated hyperparameters and 5 random seeds:

```bash
covshift-compare
```

or directly:

```bash
python scripts/compare_models.py
```

## Package layout

```
src/covshift/
  __init__.py
  core.py      # config, sampling, models, CV, training, CLI
  plot.py      # optional 3D visualization (requires matplotlib)
  bless.py     # lightweight BLESS routine (vendored)
scripts/
  compare_models.py  # calls into covshift.core
```

