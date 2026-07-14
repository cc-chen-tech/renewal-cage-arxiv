# Bilinear State-Dependent Memory Design

## Objective

Test whether the strongly colored orthogonal residual left by the best
stationary exact-force Hankel bath is caused by heterogeneous dissipation that
depends continuously on the resolved bath state.

## Resolved State And Invariants

Use the rank-16, 64-frame exact-force Hankel state

```text
s_n = [v_n, z_(1,n), ..., z_(16,n)].
```

Every `z` is a training-only linear combination of the conservative force
history.  Define two rotationally invariant, autonomous features:

```text
E_n = mean_m |z_(m,n) / sigma_m|^2,
P_n = (v_n / sigma_v) . (z_(1,n) / sigma_1) / 3.
```

Their training means and scales are frozen before held-clone evaluation.

## State-Dependent Transition

The teacher-forced diagnostic is

```text
s_(n+1) = mean(s) + A0 ds_n + A_E Ehat_n ds_n
                       + A_P Phat_n ds_n + epsilon_(n+1).
```

This is the lowest-order continuous state-dependent memory coupling.  It is
the discrete bilinear analogue of feature-coupled state-dependent GLE memory,
not a scalar switch or an event-conditioned noise resampling rule.  Compare:

1. covariance-contracted stationary rank-16 bath;
2. energy-only bilinear bath;
3. energy-plus-power bilinear bath.

All matrices are fitted by one fixed ridge rule from three training clones.
No macro observable or event label enters the fit.

## Promotion Gate

On four leave-one-clone-out folds, the full bilinear model is promoted to an
autonomous stochastic simulation only if:

- mean held velocity `R2` is at least `0.97`;
- maximum held residual-state correlation is at most `0.20`;
- maximum held residual lag correlation is at most `0.80` times the stationary
  rank-16 baseline and at most `0.70` absolutely;
- every fold improves residual lag correlation over the stationary baseline.

Failure means that these two interpretable force-bath invariants do not supply
the missing state.  The next branch must learn multiple continuous cage
features or use a cage-relative coordinate; it cannot promote the bilinear
model to a Langevin closure.

The following remain false in this diagnostic phase:

```text
autonomous_state_dependent_gle_allowed
complete_event_clock_closure_allowed
kramers_escape_claim_allowed
thermodynamic_claim_allowed
```

