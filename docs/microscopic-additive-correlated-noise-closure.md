# Microscopic Additive Correlated-Noise Closure

## Question

The exact projected SDE has a configuration-dependent `6 x 6` thermostat
covariance `Q(R)`. This experiment tests whether that multiplicative noise is
needed at the `0.01 tau` saved-frame scale, or whether a constant covariance
learned on other trajectories is sufficient.

## Held-Clone Models

For each held clone, covariance parameters are estimated only from the other
three clones:

```text
exact:              Q_n(R)
constant-full:      mean_train Q
constant-isotropic: [[a I, c I], [c I, d I]]
uncorrelated-null:  [[a I, 0],   [0, d I]]
single-null:        q I_6
```

The primary `constant-isotropic` model is the rotational group average of the
training covariance. It has only three scalar coefficients and remains
positive definite. Innovations use the past-adapted AB2 drift quadrature from
the projected-Ito audit.

## Configuration Dependence

The exact covariance is not pointwise constant. Its relative Frobenius RMS
around the global mean is `0.17633`, and its trace coefficient of variation is
`0.07659`. The closure question is whether those fluctuations have measurable
consequences for the finite-frame innovation law.

## Results

| Covariance model | trace ratio | Mahalanobis / 6 | covariance error | max lag | max state corr. | every fold pass |
|---|---:|---:|---:|---:|---:|---:|
| exact `Q(R)` | 0.93279 | 0.93805 | 0.06832 | 0.03043 | 0.03238 | yes |
| constant full | 0.93280 | 0.93708 | 0.06860 | 0.03032 | 0.03095 | yes |
| constant isotropic | 0.93280 | 0.93707 | 0.06938 | 0.03032 | 0.03068 | yes |
| uncorrelated null | 0.93280 | 0.93319 | 0.82362 | 0.02994 | 0.04063 | no |
| single-scalar null | 0.93280 | 0.93280 | 0.72951 | 0.02994 | 0.04063 | no |

The maximum difference between the constant-isotropic and exact pooled
diagnostics is only `0.00170`. Every held clone passes separately; the largest
held-fold covariance error is `0.07470`.

The fitted rotationally invariant rate matrix is

```text
a =  2.53667
c = -0.90789
d =  0.43911
```

Its center-relative correlation coefficient is

```text
c / sqrt(a d) = -0.86023.
```

The corresponding scalar `2 x 2` block has eigenvalues `0.10073` and
`2.87505`, so the constant covariance is positive definite. Removing the
cross term preserves total variance but destroys the observed tensor
covariance, increasing whitening error to `0.82362`.

## Physical Verdict

At this resolution, the configuration dependence of `Q(R)` is not required
to close the local projected innovation law. The multiplicative thermostat
can be replaced by additive Gaussian noise with a constant three-parameter
rotationally invariant covariance learned on independent trajectories.

The center and relative noises cannot be sampled independently. Their strong
anticorrelation is the coarse-grained signature that both coordinates are
driven by the same microscopic particle thermostats while satisfying
`v=w+p`. This cross noise is fixed by the coordinate projection, not a
phenomenological event-clock parameter.

This closes the stochastic coefficient but not the deterministic one. The
exact drifts `b_C(R,V)` and `b_u(R,V)` still require unresolved many-particle
state, and the earlier linear drift-history baths failed autonomous transport.
The next bottleneck is therefore a local many-body drift closure, not renewal
noise or configuration-dependent diffusion.

The claim boundary is

```text
constant_correlated_projected_noise_allowed = 1
configuration_dependent_noise_required = 0
center_relative_cross_noise_required = 1
autonomous_drift_closure_allowed = 0
autonomous_single_particle_gle_allowed = 0
complete_event_clock_closure_allowed = 0
kramers_escape_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

## Reproduction

```bash
python scripts/analyze_ka_additive_correlated_noise_closure.py \
  --drift-cache-directory tmp/decomposed_cage_drift_reduced_T058 \
  --covariance-cache-directory tmp/projected_ito_covariance_T058 \
  --output-prefix data/renewal_cage_ka_additive_correlated_noise_closure_T058
```
