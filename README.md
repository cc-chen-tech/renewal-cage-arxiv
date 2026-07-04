# Delayed Renewal Cage Model

[![CI](https://github.com/cc-chen-tech/renewal-cage-arxiv/actions/workflows/ci.yml/badge.svg)](https://github.com/cc-chen-tech/renewal-cage-arxiv/actions/workflows/ci.yml)

This repository contains a reproducible theoretical note on a minimal delayed
renewal cage model for non-Gaussian dynamics near glass transition.

The model combines:

- local Ornstein-Uhlenbeck-like cage variance,
- delayed cage-renewal events,
- Gaussian cage-center jumps,
- closed-form MSD, NGP, van Hove distribution, peak asymptotics, and
  delay-exponent diagnostics.

The core one-dimensional and three-dimensional NGP result is

```text
MSD(t) = L(t) + q R(t)

alpha_2(t) = q^2 R(t) / [L(t) + q R(t)]^2
```

where

```text
L(t) = A[1-exp(-t/tau_c)]

R(t) = lambda [t - 2 tau_d(1-exp(-t/tau_d))
               + (tau_d/2)(1-exp(-2t/tau_d))]
```

The square delay is the minimal integer choice in the generalized family
`r_m(t)=lambda[1-exp(-t/tau_d)]^m` that gives a regular zero-origin NGP.

## Repository Layout

```text
src/renewal_cage.py                         closed-form model functions
tests/test_renewal_cage.py                  unit tests
scripts/generate_renewal_cage_results.py    reproducible data/figure generator
scripts/build_arxiv_package.py              arXiv PDF figure and source builder
scripts/compile_latex.sh                    LaTeX compile helper
data/                                      generated CSV outputs
figures/                                   generated SVG figures
docs/                                      derivation and literature positioning notes
docs/arxiv-readiness-checklist.md          submission-readiness checklist
manuscript/renewal-cage-arxiv-draft.md      prose draft
paper/main.tex                             arXiv-style LaTeX manuscript
paper/references.bib                       bibliography
dist/renewal-cage-arxiv-source.zip          arXiv source package
```

## Reproduce

```bash
python3 -m unittest discover -s tests -v
python3 scripts/generate_renewal_cage_results.py
python3 scripts/build_arxiv_package.py
```

If a TeX distribution is installed, compile the manuscript with:

```bash
bash scripts/compile_latex.sh
```

The GitHub Actions workflow runs the tests, regenerates data and figures, builds the
arXiv source package, installs TeX Live, and compiles `paper/main.tex`.

Expected outputs include:

```text
figures/renewal_cage_results.svg
figures/renewal_cage_dimensionless.svg
data/renewal_cage_main.csv
data/renewal_cage_sweeps.csv
data/renewal_cage_dimensionless.csv
data/renewal_cage_diagnostics.csv
data/renewal_cage_van_hove.csv
data/renewal_cage_tail_ratios.csv
paper/figures/renewal_cage_results.pdf
paper/figures/renewal_cage_dimensionless.pdf
dist/renewal-cage-arxiv-source.zip
```

## Current Status

This is a research draft intended to become a short arXiv note. The model,
figures, arXiv source package, and LaTeX manuscript build are reproducible in
CI. The remaining submission-level checks are tracked in
`docs/arxiv-readiness-checklist.md`.
