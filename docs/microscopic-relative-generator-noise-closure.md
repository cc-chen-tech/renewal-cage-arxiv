# Microscopic Relative Generator Noise Closure

## Question

The generator-augmented Mori checkpoint fixed a held-out deterministic memory
equation for the relative cage state

```text
g = [u-u0, p, Lp],
g_(n+1) = sum_(ell=0)^40 Omega_ell g_(n-ell) + xi_(n+1).
```

Its remaining operational gap was the noise: `xi` could be reconstructed from
a microscopic trajectory, but it could not be generated without replaying that
trajectory. This checkpoint asks whether a training-only empirical noise law
can autonomously propagate the relative matrix Mori model on independent
many-particle Langevin clones.

## Finite-Memory Innovation

The four discovery clones determine the particle bias, coordinate scaling,
and order-40 Mori operators. For every training source, the sliding residual is

```text
xi_(n+1) = g_(n+1) - sum_(ell=0)^40 Omega_ell g_(n-ell).
```

Autonomous trajectories start from measured 41-frame training histories. At
each block boundary, the simulator draws a training source and time origin,
then replays `B` consecutive residual vectors in their measured temporal order.
The source and origin are redrawn for the next block. Thus `B=1` is an iid
empirical-innovation null, whereas `B>1` preserves finite colored-noise history
without using a held trajectory during propagation.

This is a nonparametric closure of the projected innovation process. It retains
the measured joint distribution across `[u-u0,p,Lp]` and the within-block time
ordering. It does not identify the residual with the original thermostat Wiener
increment and does not derive a thermal second fluctuation-dissipation theorem.

## Discovery

Whole-clone leave-one-out discovery scanned block lengths
`[1,4,16,40,100,200,400]` with the Mori order fixed at 40. Each fold used 2,000
autonomous simulations through 800 frames. The preregistered gate jointly tests
the stationary covariance, relative-position/velocity correlation, marginal
excess kurtosis, and numerical stability.

| Block length | worst covariance RMSE | worst covariance max error | minimum terminal variance ratio | gate |
|---:|---:|---:|---:|:---:|
| 1 | 0.46205 | 0.79545 | 0.27484 | fail |
| 4 | 0.20155 | 0.42090 | 0.63000 | fail |
| 16 | 0.09052 | 0.25448 | 0.89507 | fail |
| 40 | 0.06023 | 0.23928 | 0.94526 | pass |
| 100 | 0.04937 | 0.23892 | 0.89816 | pass |
| 200 | 0.04252 | 0.24298 | 0.90470 | pass |
| 400 | 0.04386 | 0.24672 | 0.91558 | pass |

The iid null reproduces the normalized two-point propagation reasonably well
but loses most stationary variance. Therefore the deterministic Mori operators
alone do not make the finite-memory residual white. The shortest all-gate
colored block is 40 frames, or `0.40 tau`, and it was fixed before independent
validation.

## Independent Validation

The fixed order-40/operator-40-block model was tested on the two independent
10-tau validation clones generated for the previous Mori checkpoint. Their
states, covariances, and residuals were not used in fitting or block selection.
Each reported validation run used 10,000 autonomous simulations. A second run
used disjoint Monte Carlo seeds to audit simulation uncertainty; no operator,
block length, metric, or gate was changed between runs.

| Validation run | worst covariance RMSE | worst covariance max error | worst target-correlation max error | worst kurtosis error | minimum terminal variance ratio |
|---|---:|---:|---:|---:|---:|
| seeds `20260716/20261716` | 0.05376 | 0.20661 | 0.14747 | 0.14555 | 0.98218 |
| seeds `20270716/20271716` | 0.05451 | 0.20338 | 0.14617 | 0.14873 | 0.94138 |

Both runs pass the fixed limits: covariance RMSE `<=0.08`, covariance maximum
error `<=0.25`, target-correlation RMSE `<=0.08`, target-correlation maximum
error `<=0.25`, kurtosis error `<=0.35`, and maximum absolute state `<=20`.

An initial 2,000-simulation validation estimate put one covariance maximum
error at `0.26202`, just outside the fixed `0.25` gate. Increasing only the
Monte Carlo sample count to 10,000 moved both disjoint-seed estimates below the
gate. Reporting both repeats makes this a convergence audit rather than an
unreported reselection of the favorable run.

## Physical Meaning And Boundary

This result closes a concrete intermediate layer:

```text
4096-particle Langevin trajectories
  -> smooth relative cage coordinate and exact generator image
  -> finite matrix Mori memory
  -> measured colored innovation blocks
  -> autonomous relative-coordinate trajectories.
```

It establishes that the unresolved many-body bath acting on the chosen relative
state has essential temporal color over about the same 40-frame scale as the
resolved Mori memory. It also supplies an autonomous simulator for the local
relative process that reproduces held-out second moments and a one-point
non-Gaussian statistic.

It does not yet provide a microscopic thermal-noise model. The empirical blocks
are sampled from many-particle trajectories, not generated from temperature,
friction, or a derived bath spectral density. It also omits the cage-center
process, cage jumps, event waiting times, long-time diffusion, NGP, and
`F_s(k,t)`. Consequently it is not yet the desired complete single-particle
Langevin dynamics.

The claim boundary is

```text
selected_innovation_block_length = 40
iid_innovation_noise_allowed = 0
colored_orthogonal_noise_required = 1
empirical_block_noise_generation_closed = 1
autonomous_relative_matrix_mori_simulation_allowed = 1
thermal_fdt_adjoint_audit_pass = 0
microscopic_thermal_noise_model_closed = 0
autonomous_single_particle_gle_allowed = 0
complete_event_clock_closure_allowed = 0
kramers_escape_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

## Reproduction

```bash
python scripts/analyze_ka_relative_generator_noise_closure.py \
  --training-drift-cache-directory tmp/decomposed_cage_drift_reduced_T058 \
  --output-prefix data/renewal_cage_ka_relative_generator_noise_discovery_T058

python scripts/analyze_ka_relative_generator_noise_closure.py \
  --training-drift-cache-directory tmp/decomposed_cage_drift_reduced_T058 \
  --validation-drift-cache-directory tmp/relative_generator_mori_validation_drift_T058 \
  --fixed-innovation-block-length 40 \
  --simulation-count 10000 \
  --output-prefix data/renewal_cage_ka_relative_generator_noise_validation_highstat_T058

python scripts/analyze_ka_relative_generator_noise_closure.py \
  --training-drift-cache-directory tmp/decomposed_cage_drift_reduced_T058 \
  --validation-drift-cache-directory tmp/relative_generator_mori_validation_drift_T058 \
  --fixed-innovation-block-length 40 \
  --simulation-count 10000 \
  --seed 20270716 \
  --output-prefix data/renewal_cage_ka_relative_generator_noise_validation_highstat_seedB_T058
```
