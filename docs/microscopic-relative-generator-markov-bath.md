# Microscopic Relative Generator Markov Bath

## Question

The empirical block-noise checkpoint autonomously propagated the relative Mori
state, but its 40-frame residual blocks were still replayed directly from the
many-particle trajectories. This checkpoint asks whether that colored bath can
be replaced by a finite-dimensional Markov state driven by temporally white
noise:

```text
g_(n+1) = sum_(ell=0)^40 Omega_ell g_(n-ell) + xi_(n+1),

xi_(n+1) - mu
  = sum_(j=1)^P A_j [xi_(n+1-j)-mu] + epsilon_(n+1).
```

Here `g=[u-u0,p,Lp]`, and both the Mori operators and the residual `xi` come
from the microscopic 4096-particle Langevin trajectories. No diffusion,
scattering, event-clock, or other macroscopic observable enters the fit.

## Finite Markov Embedding

Define the bath state

```text
h_n = [xi_n, xi_(n-1), ..., xi_(n-P+1)].
```

The VAR recurrence is a first-order Markov chain in `h_n`, with a companion
matrix assembled from `[A_1,...,A_P]`. The coefficients are obtained from the
pooled multivariate Yule-Walker equations. A positive residual covariance and
companion spectral radius below one are required before simulation.

This is the discrete analogue of augmenting a generalized Langevin equation
with auxiliary bath variables. Linear colored-noise Langevin constructions are
well established, for example in Ceriotti, Bussi, and Parrinello,
[Phys. Rev. Lett. 102, 020601 (2009)](https://doi.org/10.1103/PhysRevLett.102.020601),
and data-derived GLE parameterizations were developed by Lei, Baker, and Li,
[PNAS 113, 14183 (2016)](https://doi.org/10.1073/pnas.1609587113). The new
question tested here is narrower: whether the bath inferred from this specific
microscopic cage-generator projection can be made white-driven without losing
its held-out glassy statistics.

## Leakage Controls

Each discovery fold uses three complete clones for the cage bias, normalization,
Mori operators, VAR coefficients, and white-residual distribution. The held
clone is used only for scoring. Every available training source is used: 576
sources in each three-clone discovery fit and 768 sources in the four-clone
validation fit. Therefore an apparent order improvement cannot come from
changing the fitted particle/component subset.

Residual whiteness is checked through 40 frames, not only at lag one, on both
training and held clones. Model selection scans `P=[4,8,16,40]` and chooses the
shortest all-gate order. The same covariance, target-correlation,
excess-kurtosis, and stability gates used for the block closure are retained.

## Discovery: Gaussian Null

The fitted VAR dynamics is stable and nearly whitens the residual. At `P=16`,
the worst companion spectral radius is `0.86722`, and the worst residual
correlation over lags 1-40 is `0.01161` in training and `0.01725` on held
clones. Gaussian white driving also passes the second-moment limits:

| Gaussian `P=16` worst fold | value | limit |
|---|---:|---:|
| stationary covariance RMSE | 0.03588 | <= 0.08 |
| stationary covariance maximum error | 0.15708 | <= 0.25 |
| target correlation RMSE | 0.06201 | <= 0.08 |
| target correlation maximum error | 0.19704 | <= 0.25 |
| marginal excess-kurtosis error | 0.42796 | <= 0.35 |

Thus the Gaussian null fails only the non-Gaussian marginal gate. Increasing
the bath order to 40 does not repair it: the worst kurtosis error remains
`0.42474`. This is a useful negative result. A linear Gaussian auxiliary bath
can reproduce the relative two-point dynamics, but it cannot reproduce the
held non-Gaussian one-point statistics.

## Discovery: Empirical White Driving

The alternative keeps the fitted VAR coefficients fixed but draws each
`epsilon_n` independently from the training residual pool. It preserves the
measured one-step vector distribution while discarding its time order. At
`P=16`, the largest absolute white-residual excess kurtosis is `2.80920`, so
this is decisively non-Gaussian white driving.

| Empirical driving | `P=8` worst fold | `P=16` worst fold | limit |
|---|---:|---:|---:|
| residual correlation, lags 1-40 | 0.01656 | 0.01161 | <= 0.05 |
| stationary covariance RMSE | 0.06390 | 0.03604 | <= 0.08 |
| stationary covariance maximum error | 0.27979 | 0.16672 | <= 0.25 |
| target correlation maximum error | 0.20562 | 0.22033 | <= 0.25 |
| marginal excess-kurtosis error | 0.29791 | 0.27282 | <= 0.35 |
| gate | fail | pass | all limits |

Order 16 is therefore the shortest passing finite Markov bath. The improvement
over Gaussian driving is not a memory change; it comes solely from retaining
the non-Gaussian distribution of the now-white unresolved kicks.

## Independent Validation

After fixing `P=16` and empirical iid driving, the model was tested on the two
independent 10-tau Langevin clones. Each validation run used 10,000 autonomous
paths. All 768 training sources enter both fits; the second run uses fully
disjoint Monte Carlo simulation seeds.

| Validation worst metric | seed set A | seed set B | limit |
|---|---:|---:|---:|
| held residual correlation, lags 1-40 | 0.01318 | 0.01318 | <= 0.05 |
| stationary covariance RMSE | 0.02299 | 0.02259 | <= 0.08 |
| stationary covariance maximum error | 0.09603 | 0.10730 | <= 0.25 |
| target correlation RMSE | 0.04853 | 0.04799 | <= 0.08 |
| target correlation maximum error | 0.15516 | 0.15391 | <= 0.25 |
| marginal excess-kurtosis error | 0.32873 | 0.31293 | <= 0.35 |
| minimum terminal variance ratio | 0.98941 | 0.95953 | diagnostic |

Both repeats pass. This replaces direct 40-frame innovation replay by a
16-lag, three-component bath state driven by iid kicks while preserving held
relative-state second moments and marginal non-Gaussianity.

## What This Means Physically

The unresolved local many-body environment has two separable ingredients:

```text
finite linear bath memory
  + temporally white but strongly non-Gaussian microscopic kicks.
```

The first ingredient can be represented by a finite Markov companion state.
The second cannot be replaced by standard Gaussian thermal noise without losing
the held non-Gaussian signature. This points to missing nonlinear or
state-dependent microscopic bath coordinates: after those variables are
resolved, their underlying Gaussian thermostat noise may become sufficient.

This checkpoint is not yet a continuous-time Langevin derivation. A discrete
stable companion matrix is not automatically embeddable as a real continuous
OU drift with a positive diffusion matrix. More importantly, the iid empirical
kicks are non-Gaussian and are sampled from many-particle data rather than
derived from `T` and `gamma`. The cage center, jumps, event clock, diffusion,
NGP, and `F_s(k,t)` also remain outside this relative-state simulator.

The claim boundary is

```text
selected_bath_order = 16
finite_dimensional_discrete_markov_bath_supported = 1
empirical_non_gaussian_white_driving_supported = 1
empirical_white_markov_bath_generation_closed = 1
autonomous_relative_matrix_mori_simulation_allowed = 1
gaussian_markov_bath_generation_closed = 0
continuous_time_ou_embedding_audited = 0
thermal_fdt_adjoint_audit_pass = 0
microscopic_thermal_noise_model_closed = 0
autonomous_single_particle_gle_allowed = 0
complete_event_clock_closure_allowed = 0
kramers_escape_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

## Reproduction

```bash
python scripts/analyze_ka_relative_generator_markov_bath.py \
  --training-drift-cache-directory tmp/decomposed_cage_drift_reduced_T058 \
  --bath-orders 4 8 16 40 \
  --fit-source-count 8192 \
  --white-driving-distribution gaussian \
  --output-prefix data/renewal_cage_ka_relative_generator_markov_bath_discovery_T058

python scripts/analyze_ka_relative_generator_markov_bath.py \
  --training-drift-cache-directory tmp/decomposed_cage_drift_reduced_T058 \
  --bath-orders 4 8 16 40 \
  --fit-source-count 8192 \
  --white-driving-distribution empirical \
  --output-prefix data/renewal_cage_ka_relative_generator_markov_bath_empirical_discovery_T058

python scripts/analyze_ka_relative_generator_markov_bath.py \
  --training-drift-cache-directory tmp/decomposed_cage_drift_reduced_T058 \
  --validation-drift-cache-directory tmp/relative_generator_mori_validation_drift_T058 \
  --fixed-bath-order 16 \
  --simulation-count 10000 \
  --fit-source-count 8192 \
  --white-driving-distribution empirical \
  --output-prefix data/renewal_cage_ka_relative_generator_markov_bath_empirical_validation_T058

python scripts/analyze_ka_relative_generator_markov_bath.py \
  --training-drift-cache-directory tmp/decomposed_cage_drift_reduced_T058 \
  --validation-drift-cache-directory tmp/relative_generator_mori_validation_drift_T058 \
  --fixed-bath-order 16 \
  --simulation-count 10000 \
  --fit-source-count 8192 \
  --white-driving-distribution empirical \
  --seed 20360717 \
  --output-prefix data/renewal_cage_ka_relative_generator_markov_bath_empirical_validation_seedB_T058
```
