# Microscopic Langevin Force-Response Checkpoint

## Scope

This checkpoint asks how far the renewal-cage dynamics can be connected to the
standard many-particle Kob-Andersen (KA) Langevin equation without assuming a
jump clock.  It records a dynamical coarse-graining result, not a derivation of
configurational entropy, a Kauzmann temperature, or an ideal-glass transition.
Every exported result therefore keeps `thermodynamic_claim_allowed = 0`.

The microscopic starting point is

```text
m dv_i/dt = F_i(R) - gamma v_i + sqrt(2 gamma T) eta_i,
dR_i/dt = v_i,
```

where `F_i(R)` is recomputed from the full KA configuration.  Matched-noise
symmetric perturbations isolate the linear response while retaining all 4096
particles and the explicit Langevin bath.

## Reproducible evidence

The raw force-balance diagnostic separates the conservative pair force from
the recorded total force and checks the residual against the Langevin noise
scale:

- script: `scripts/analyze_ka_raw_langevin_force_balance.py`
- output: `data/renewal_cage_ka_raw_langevin_force_balance_T058_summary.csv`
- result: residual FDT variance ratio `0.904 +/- 0.005` and lag-one
  correlation `0.0607 +/- 0.0031` for four clones and eight targets.

The symmetric-response calculation verifies a linear perturbative regime:

- scripts: `scripts/analyze_ka_symmetric_impulse_response.py` and
  `scripts/analyze_ka_decomposed_symmetric_response.py`
- outputs: `data/renewal_cage_ka_symmetric_impulse_response_T058_summary.csv`
  and `data/renewal_cage_ka_decomposed_symmetric_response_T058_summary.csv`
- result: at `0.2 tau`, the scalar pair-force response has relative errors
  `3.95e-4`, `9.15e-4`, and `1.02e-3` for perturbations `0.001`, `0.002`, and
  `0.004`.

The four-member ensemble extends the linearity test and then performs a strict
temporal holdout:

- script: `scripts/analyze_ka_decomposed_symmetric_response_ensemble.py`
- output:
  `data/renewal_cage_ka_decomposed_symmetric_response_ensemble_T058_summary.csv`
- result: the `0.002` response relative to `0.001` remains linear through
  approximately `1 tau` (position-response relative L2 errors `8.14e-5`,
  `8.60e-4`, and `2.80e-3` at `0.2`, `0.5`, and `1 tau`) but not at `2 tau`
  (`0.606`).
- falsification: a scalar finite-support kernel inferred only from the early
  response has heldout relative error approximately one.  A stable
  `(position, velocity, pair force)` Markov embedding trained through
  `0.05 tau` improves the early holdout but still has relative errors `0.653`,
  `0.846`, and `0.879` at `0.2`, `0.5`, and `1 tau`.

The independent trajectory diagnostic reaches the same narrow conclusion:

- script: `scripts/analyze_ka_pair_force_auxiliary_markov.py`
- output: `data/renewal_cage_ka_pair_force_auxiliary_markov_T058_summary.csv`
- result: the exact instantaneous pair force is a strong resolved state
  coordinate (`heldout R^2 = 0.986`) but the two-variable closure misses
  diffusion by `75%`, so it is not a sufficient long-time reduction.

## Exact generator interface

For the force observable, the first generator-generated coordinate is computed
directly from the same KA potential,

```text
G_i = L F_i = -sum_j H_ij v_j
    = -sum_(j != i) K_ij (v_i - v_j).
```

The conditional Langevin covariance is

```text
Cov[dG_i | R,V] / dt = 2 gamma T sum_j H_ij H_ij^T.
```

`ka_lj_force_generator_observables` exposes both quantities.  Unit tests check
the force directional derivative and the Hessian-noise identity on an analytic
pair configuration.  This gives a parameter-free entry point for the next
Krylov/Mori-Zwanzig closure test.

## Current boundary

The evidence supports this statement:

> The exact KA pair force and its generator descendants are necessary resolved
> coordinates for short-time cage response, but the tested low-dimensional
> instantaneous or finite-support closures do not yet predict the long-time
> cage-to-cage dynamics.

It does not yet establish a microscopic renewal hazard, exponential precursor
readiness, a Kramers-conditioned escape law, or a thermodynamic glass
transition.  Those remain explicit validation targets rather than conclusions.
