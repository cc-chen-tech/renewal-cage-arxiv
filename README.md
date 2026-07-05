# Delayed Renewal Cage Model

[![CI](https://github.com/cc-chen-tech/renewal-cage-arxiv/actions/workflows/ci.yml/badge.svg)](https://github.com/cc-chen-tech/renewal-cage-arxiv/actions/workflows/ci.yml)

This repository contains a reproducible theoretical note on a minimal delayed
renewal cage model for dynamical glass signatures. It is an effective
microdynamic diagnostic theory with an explicit thermodynamic entropy closure,
not a first-principles derivation of the thermodynamic glass transition.

The model combines:

- local Ornstein-Uhlenbeck-like cage variance,
- delayed cage-renewal events,
- Gaussian cage-center jumps,
- closed-form MSD, NGP, self van Hove distribution, self-intermediate
  scattering function, temperature-dependent alpha relaxation, peak
  asymptotics, Stokes-Einstein decoupling, activated-barrier diagnostics,
  fractional Stokes-Einstein exponents, apparent alpha-activation/fragility
  diagnostics, alpha-shape time-temperature-superposition collapse residuals,
  renewal-count susceptibility, renewal-domain chi4/cooperative-size
  diagnostics, spatial facilitation chi4-length closure,
  thermodynamic entropy/Adam-Gibbs closure,
  effective MCT beta-window closure,
  NGP peak/alpha-relaxation coupling, finite-exchange
  heterogeneity diagnostics, a temperature-dependent facilitated-exchange law,
  a persistence/exchange renewal-clock decoupling diagnostic, a static-gamma
  mobility-disorder null model, late-time mechanism-selection diagnostics, a
  glass-dynamics phenomenon audit, barrier-facilitation signature phase
  diagrams, closed barrier-threshold requirements, finite-time consistency
  diagnostics, joint persistence/exchange inversion protocols, and observable
  inversion/falsifiability criteria.

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
The finite-time consistency diagnostic inverts the plateau relation
`alpha=beta*y/(1+y)^2`; the late-time check uses the `y>1` branch and requires
the observed late NGP to lie below the peak bound `beta/4`.
The scattering-transport inversion uses the measured Debye-Waller plateau,
`D`, `tau_alpha`, and `tau_d` to infer `A`, `q`, and `lambda`; it requires the
existence margin `D tau_d F(tau_alpha/tau_d) k^2 / [-log h]` to exceed one.
A fuller observable inversion uses `f_k`, `D`, NGP peak time, and NGP peak
height to infer `A`, `q`, `lambda`, and `tau_d`, leaving `tau_alpha` as a
held-out consistency check.
The persistence/exchange inversion uses `D` to fix `tau_x=q/(2D)`, solves
`G(a_k,tau_alpha;tau_p,tau_x)=h` for `tau_p`, and leaves late NGP recovery as a
held-out falsification observable.
The joint persistence/exchange protocol uses one anchor alpha time plus `D` to
infer the clocks, then holds out multi-`k` alpha times, late NGP recovery, and a
single-domain `chi4` proxy.
The uncertainty-weighted persistence/exchange protocol converts those held-out
residuals into `z` scores using measurement-error estimates.
The finite-exchange protocol adds a second data-level residual: late NGP gives
`c_NGP=R_l alpha_2(t_l)-1`, the alpha slope gives `c_alpha` through
`log(1+Gamma_k c_alpha)/c_alpha`, and `log(c_alpha/c_NGP)` tests whether both
observables share one exchange scale. With measurement uncertainties, the same
closed derivatives propagate this residual into a `z` score. A multi-`k`
extension requires all `c_alpha(k)` values from `F_s(k,t)` to collapse to the
same `c_NGP`.
A static-gamma null model gives `Var N=R+R^2/kappa0`,
`alpha_2(t)->1/kappa0`, and `-log Phi_alpha/R->0`; it can broaden alpha
relaxation but fails long-time Gaussian recovery.
The SOTA benchmark consistency table turns selected literature-level claims
into checkable rows, currently covering Debye-Waller cage localization,
Kob-Andersen beta-window visibility and MCT exponent-parameter consistency,
Gaussian-recovery mechanism selection against the static-disorder null,
cooling-induced NGP peak shift with late Gaussian recovery, fractional
Stokes-Einstein decoupling, dynamic-heterogeneity chi4 growth, alpha-shape
time-temperature-superposition breakdown, spatial facilitation-front
growth-law inversion, KWW stretched-alpha window fitting, persistence/exchange
transport inversion with held-out late-NGP validation, joint multi-k/chi4
persistence-exchange falsification, a van Hove tail/recovery consistency
check, an Angell/Adam-Gibbs fragility-growth scope check, and a
thermodynamic-transition no-go/scope-boundary row.
The alpha-shape TTS diagnostic rescales the cage-normalized alpha relaxation by
`tau_alpha`:

```text
Y_k(u;T) = -log Phi_alpha(k,u tau_alpha;T) / [-log h]
C_k(T) = Gamma_k(T) lambda(T) tau_d(T)
```

Within the minimal model, the whole scaled shape is controlled by `C_k`; exact
time-temperature superposition requires `C_k` to be temperature independent.
The NGP peak and alpha relaxation are also linked by renewal counts:
`R_peak=A/q` and
`R_alpha=-log(h)/[1-exp(-k^2 q/2)]`, so their time ratio is fixed by the same
delayed-renewal inverse.

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
figures/renewal_cage_scattering.svg
figures/renewal_cage_temperature.svg
figures/renewal_cage_alpha_shape.svg
figures/renewal_cage_facilitated_exchange.svg
figures/renewal_cage_persistence_exchange.svg
figures/renewal_cage_persistence_exchange_protocol.svg
figures/renewal_cage_persistence_exchange_joint_protocol.svg
figures/renewal_cage_persistence_exchange_uncertainty_protocol.svg
figures/renewal_cage_glass_audit.svg
figures/renewal_cage_glass_phase_diagram.svg
figures/renewal_cage_spatial_chi4.svg
figures/renewal_cage_thermodynamic_closure.svg
figures/renewal_cage_mct_beta_closure.svg
figures/renewal_cage_sota_benchmark_consistency.svg
figures/renewal_cage_literature_inversion_readiness.svg
figures/renewal_cage_observable_falsification_matrix.svg
figures/renewal_cage_benchmark_fusion_readiness.svg
figures/renewal_cage_barrier_requirements.svg
figures/renewal_cage_barrier.svg
figures/renewal_cage_heterogeneity.svg
figures/renewal_cage_heterogeneity_map.svg
figures/renewal_cage_static_null.svg
figures/renewal_cage_mechanism_selection.svg
figures/renewal_cage_inversion.svg
data/renewal_cage_main.csv
data/renewal_cage_sweeps.csv
data/renewal_cage_dimensionless.csv
data/renewal_cage_diagnostics.csv
data/renewal_cage_consistency.csv
data/renewal_cage_sota_comparison.csv
data/renewal_cage_van_hove.csv
data/renewal_cage_tail_ratios.csv
data/renewal_cage_scattering.csv
data/renewal_cage_peak_relaxation.csv
data/renewal_cage_temperature.csv
data/renewal_cage_alpha_shape.csv
data/renewal_cage_kww_alpha.csv
data/renewal_cage_facilitated_exchange.csv
data/renewal_cage_persistence_exchange.csv
data/renewal_cage_persistence_exchange_protocol.csv
data/renewal_cage_persistence_exchange_joint_protocol.csv
data/renewal_cage_persistence_exchange_uncertainty_protocol.csv
data/renewal_cage_glass_audit.csv
data/renewal_cage_glass_phase_diagram.csv
data/renewal_cage_spatial_chi4.csv
data/renewal_cage_spatial_facilitation_inversion.csv
data/renewal_cage_thermodynamic_closure.csv
data/renewal_cage_mct_beta_closure.csv
data/renewal_cage_sota_benchmark_consistency.csv
data/renewal_cage_literature_inversion_readiness.csv
data/renewal_cage_observable_falsification_matrix.csv
data/renewal_cage_benchmark_fusion_readiness.csv
data/renewal_cage_barrier_requirements.csv
data/renewal_cage_susceptibility.csv
data/renewal_cage_chi4.csv
data/renewal_cage_barrier.csv
data/renewal_cage_heterogeneity.csv
data/renewal_cage_heterogeneity_diagnostics.csv
data/renewal_cage_heterogeneity_map.csv
data/renewal_cage_heterogeneity_protocol.csv
data/renewal_cage_heterogeneity_multik.csv
data/renewal_cage_static_null.csv
data/renewal_cage_mechanism_selection.csv
data/renewal_cage_inversion.csv
data/renewal_cage_full_inference.csv
paper/figures/renewal_cage_results.pdf
paper/figures/renewal_cage_dimensionless.pdf
paper/figures/renewal_cage_scattering.pdf
paper/figures/renewal_cage_temperature.pdf
paper/figures/renewal_cage_alpha_shape.pdf
paper/figures/renewal_cage_facilitated_exchange.pdf
paper/figures/renewal_cage_persistence_exchange.pdf
paper/figures/renewal_cage_persistence_exchange_protocol.pdf
paper/figures/renewal_cage_persistence_exchange_joint_protocol.pdf
paper/figures/renewal_cage_persistence_exchange_uncertainty_protocol.pdf
paper/figures/renewal_cage_glass_audit.pdf
paper/figures/renewal_cage_glass_phase_diagram.pdf
paper/figures/renewal_cage_spatial_chi4.pdf
paper/figures/renewal_cage_thermodynamic_closure.pdf
paper/figures/renewal_cage_mct_beta_closure.pdf
paper/figures/renewal_cage_sota_benchmark_consistency.pdf
paper/figures/renewal_cage_literature_inversion_readiness.pdf
paper/figures/renewal_cage_observable_falsification_matrix.pdf
paper/figures/renewal_cage_benchmark_fusion_readiness.pdf
paper/figures/renewal_cage_barrier_requirements.pdf
paper/figures/renewal_cage_barrier.pdf
paper/figures/renewal_cage_heterogeneity.pdf
paper/figures/renewal_cage_heterogeneity_map.pdf
paper/figures/renewal_cage_static_null.pdf
paper/figures/renewal_cage_mechanism_selection.pdf
paper/figures/renewal_cage_inversion.pdf
dist/renewal-cage-arxiv-source.zip
```

## Current Status

This is a research draft intended to become a short arXiv note. The model,
figures, arXiv source package, and LaTeX manuscript build are reproducible in
CI. The remaining submission-level checks are tracked in
`docs/arxiv-readiness-checklist.md`.
