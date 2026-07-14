# Projected Ito Innovation Audit Design

## Objective

Test the exact local stochastic equation for the smooth cage coordinates
before attempting another autonomous closure. The question is whether observed
increments of `(w,p)` agree with the microscopic drifts `(b_C,b_u)` and their
configuration-dependent joint thermostat covariance.

## Finite-Frame Estimator

For `Y=(w,p)`, `b=(b_C,b_u)`, and joint covariance rate `Q`, the primary
left-point Ito residual is

```text
e_n = Y_(n+1) - Y_n - dt b_n,
Sigma_n = dt Q_n.
```

The centered trapezoid estimator is retained only as a discretization
sensitivity. Nonoverlapping blocks of 1, 2, 4, and 8 saved intervals sum both
`e_n` and `Sigma_n` before whitening.

## Diagnostics And Gates

Pool four independent clones only after retaining clone and particle axes for
lag diagnostics. Report whitened mean, covariance error, Mahalanobis energy,
component excess kurtosis, lag-one correlation, and maximum correlation with
the starting physical state `(w,p,u)`.

The local projected-SDE gate uses left-point stride 1 and requires:

- trace variance ratio and mean Mahalanobis per dimension in `[0.8,1.2]`;
- maximum absolute whitened mean at most `0.05`;
- maximum covariance error at most `0.10`;
- maximum lag-one correlation at most `0.05`;
- maximum state correlation at most `0.10`.

Gaussian kurtosis is reported but not promoted to a hard gate because a
state-dependent stochastic integral need not be exactly Gaussian over a
finite saved interval.

If the left estimator and trapezoid sensitivity disagree, a past-adapted
Adams-Bashforth-2 drift quadrature may be added as a post-primary diagnostic.
It must not replace or retroactively alter the preregistered left-point gate.

Passing this audit validates the local projected SDE, not an autonomous
single-particle model. The latter still requires closing `b_C`, `b_u`, and `Q`
from retained single-particle state.
