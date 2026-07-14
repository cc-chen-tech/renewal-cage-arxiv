# Relative Generator Markov Bath Design

## Objective

Replace direct colored-residual block replay by a finite-dimensional Markov
bath driven by temporally iid noise, while preserving the held-out relative
generator Mori statistics.

## Model

Fit a pooled stationary vector autoregression to the order-40 Mori residual:

```text
xi_(n+1)-mu = sum_(j=1)^P A_j [xi_(n+1-j)-mu] + epsilon_(n+1).
```

Use multivariate Yule-Walker equations, require a stable companion matrix and a
positive residual covariance, and audit residual correlations over lags 1-40.
Compare Gaussian white `epsilon` against iid samples from the fitted training
residual distribution.

## Protocol

1. Use whole-clone leave-one-out discovery with bath orders `[4,8,16,40]`.
2. Use every available training source in each fold: 576 in discovery and 768
   in validation.
3. Keep Mori order 40 and all observable gates fixed.
4. Select the shortest all-gate bath only from discovery folds.
5. Validate the frozen bath on two independent clones with 10,000 autonomous
   paths and repeat with disjoint simulation seeds.

## Claim Boundary

A pass permits a finite discrete Markov bath and autonomous relative-state
simulation. Empirical non-Gaussian iid driving does not permit a Gaussian
thermal-bath, continuous-time OU, thermal-FDT, event-clock, Kramers,
thermodynamic, or complete single-particle Langevin claim.
