# Microscopic Relative Harmonic Volterra And FDT Test

## Question

The relative PMF is harmonic but the Markov OU fails. This experiment asks
whether a scalar memory kernel closes the remaining dynamics and, separately,
whether that kernel satisfies the generalized fluctuation-dissipation theorem
required of a physical Mori GLE.

## Microscopic Bias Coordinate

About 29 to 32% of the raw relative-coordinate variance is a persistent
particle-specific offset in an asymmetric cage. For each held clone, define
the offset using only the other three isoconfigurational clones:

```text
u0_i = mean over training clone and time axes of u_i,
delta u_i = u_i - u0_i.
```

The training estimate correlates `0.790-0.835` with the held clone's own
time-averaged offset and removes `17.1-20.7%` of held variance without using
the held trajectory. This identifies `u0_i` as an initial-configuration
dependent microscopic cage-asymmetry variable.

## Harmonic Volterra Equation

With PMF stiffness fixed by FDT/state counting, the candidate scalar GLE is

```text
d delta_u / dt = p,
dp/dt = -kappa delta_u - gamma p
        - integral_0^t K(s) p(t-s) ds + eta(t).
```

Orthogonality to the initial resolved state gives

```text
d C_pp(t)/dt = -kappa C_up(t) - gamma C_pp(t)
               - integral_0^t K(s) C_pp(t-s) ds.
```

A causal trapezoid recursion fixes a `1 tau` kernel from training
`C_pp,C_up`. No held correlation, force, event, or macro observable enters the
kernel.

## Held Correlation Prediction

| Metric | Four-fold result |
|---|---:|
| minimum training-bias/held-bias correlation | `0.79011` |
| minimum held variance fraction removed | `0.17126` |
| maximum fit-window correlation RMSE | `0.06054` |
| maximum `1-8 tau` extrapolation RMSE | `0.06382` |
| maximum extrapolation correlation error | `0.18326` |
| mean kernel tail/peak at `1 tau` | `0.24979` |

The kernel passes the preregistered two-point correlation gate in every held
clone. It predicts normalized `C_uu`, `C_pp`, and `C_up` for seven times its
fitted support with no held refit.

## Generalized-FDT Failure

For a physical scalar Mori GLE, the reconstructed orthogonal force must obey

```text
<eta(t) eta(0)> = sigma_p^2 K(t).
```

It does not:

| FDT diagnostic | Four-fold result |
|---|---:|
| random-force variance ratio | `0.76728` |
| force/kernel shape correlation | `0.27149` |
| normalized FDT RMSE | `0.94666` |
| minimum kernel Toeplitz eigenvalue | `-3.8590e4` |

The strongly negative Toeplitz eigenvalue proves that the inferred scalar
kernel cannot itself be a stationary random-force covariance. Reproducing a
two-point correlation by triangular inversion is therefore not enough to
establish a physical GLE.

## Physical Verdict

Two genuine microscopic structures have been isolated:

1. a particle-specific isoconfigurational cage bias `u0_i`;
2. a reproducible non-Markovian two-point memory representation for
   fluctuations around that bias.

But the scalar kernel fails generalized FDT and cannot generate the correct
orthogonal force. The next reduction must retain a richer orthogonal state,
such as a matrix kernel coupling cage bias/environment modes, or compute the
Mori random force through an explicit orthogonal projection. It is not valid
to promote this Volterra fit to autonomous stochastic dynamics.

The claim boundary is

```text
isoconfigurational_cage_bias_supported = 1
relative_correlation_volterra_allowed = 1
relative_mori_fdt_closure_allowed = 0
physical_scalar_relative_gle_allowed = 0
relative_orthogonal_force_closure_required = 1
autonomous_single_particle_gle_allowed = 0
complete_event_clock_closure_allowed = 0
kramers_escape_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

## Reproduction

```bash
python scripts/analyze_ka_relative_harmonic_volterra_fdt.py \
  --drift-cache-directory tmp/decomposed_cage_drift_reduced_T058 \
  --covariance-cache-directory tmp/projected_ito_covariance_T058 \
  --output-prefix data/renewal_cage_ka_relative_harmonic_volterra_fdt_T058
```
