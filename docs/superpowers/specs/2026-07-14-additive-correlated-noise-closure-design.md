# Additive Correlated Projected-Noise Closure Design

## Objective

Determine whether the exact configuration-dependent projected thermostat
covariance `Q(R)` is needed at the saved-frame scale, or whether training-only
constant covariance closes held-out cage-center/relative innovations.

This formal comparison is pilot-informed by the projected-Ito audit and is an
exploratory closure test, not an independent confirmatory dataset.

## Models

For every held clone, fit covariance only on the other three clones:

```text
exact:              Q_n(R), diagnostic reference only
constant-full:      mean_train Q
constant-isotropic: [[a I, c I], [c I, d I]]
uncorrelated-null:  [[a I, 0],   [0, d I]]
single-null:        q I_6
```

The rotationally invariant coefficients are traces of the corresponding
`3 x 3` training blocks. No held-clone, event, or macro information enters
them.

## Gate

Use the past-adapted Adams-Bashforth-2 stride-1 innovations established by the
finite-frame audit. The constant-isotropic model must pass the same variance,
mean, covariance, lag, and state-correlation gates on every held clone and in
the pooled result. Its pooled metrics must remain within `0.02` of exact `Q(R)`
for trace ratio, Mahalanobis per dimension, covariance error, lag, and state
correlation. The uncorrelated null must fail the covariance-error threshold.

Passing permits only an additive correlated-noise closure for the projected
local SDE. It does not close the deterministic drifts or autonomous transport.
