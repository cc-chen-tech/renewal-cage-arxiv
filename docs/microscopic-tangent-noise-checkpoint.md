# Microscopic tangent-noise checkpoint

## Scope

This checkpoint asks whether the matched common-noise response of the full
4096-particle Kob-Andersen Langevin system obeys the stochastic tangent
equation implied by the microscopic pair forces. It is a dynamical result
only. Every generated table preserves `thermodynamic_claim_allowed = 0`.

## Exact tangent equation

For the tagged-particle force `F_i(X)` under underdamped Langevin dynamics,

```text
dX = V dt
dV = (F - gamma V) dt + sqrt(2 gamma T) dW
```

the common-noise central response obeys

```text
d(delta X)  = delta V dt
d(delta V)  = (delta F - gamma delta V) dt
d(delta F)  = delta(LF) dt
d(delta LF) = delta(L2F) dt - sqrt(2 gamma T) delta H dW.
```

Here `delta H` is the central response of the force Jacobian. The final
martingale term is essential: a deterministic 12-state closure for
`(delta X, delta V, delta F, delta LF)` omits microscopic noise that survives
the common-noise subtraction because the two trajectories sample different
force Jacobians.

For pair Hessian responses `delta H_ij`, the parameter-free instantaneous
covariance rate is

```text
Q_delta = 2 gamma T [
    (sum_j delta H_ij)(sum_j delta H_ij)^T
    + sum_j delta H_ij delta H_ij^T
].
```

The two terms are respectively the tagged-particle noise contribution and the
independent neighbor-noise contributions.

## Numerical protocol

- Full 4096-particle KA trajectories at `T = 0.58`, `gamma = 1`.
- Eight independently seeded matched common-noise members.
- Central perturbations `epsilon = 0.001` and `0.002`.
- Integration and saved-frame spacing `0.001 tau`; diagnostics use stride 5.
- LAMMPS dumps use `%.17g`; lower text precision was found to contaminate the
  finite-difference identities at the smallest frame spacing.
- No macroscopic observable is fitted in the covariance test.

The analysis is implemented by
`scripts/analyze_ka_generator_response_resolution.py`,
`scripts/analyze_ka_generator_response_cutoff.py`, and
`scripts/analyze_ka_tangent_noise_covariance.py`.

## Hard-cutoff boundary

The standard KA `lj/cut` force is not differentiable when a plus/minus pair
straddles the cutoff. More importantly, a first active-set mismatch changes
the subsequent microscopic paths. Removing only the interval containing the
crossing is therefore invalid. The diagnostics now retain only intervals
whose endpoint precedes the first mismatch and right-censor the complete
remainder of that member.

All eight members cross early. At stride 5, rigorous censoring retains only
35 of 320 candidate intervals for `epsilon = 0.001` and 33 of 320 for
`epsilon = 0.002`. Individual-member estimates are intentionally emitted as
`nan` when fewer than four pre-crossing intervals remain.

## Parameter-free covariance check

On the pooled pre-crossing intervals:

| epsilon | retained | trace variance ratio | mean squared Mahalanobis | energy correlation | whitened lag-1 |
|---:|---:|---:|---:|---:|---:|
| 0.001 | 35 | 0.977 | 2.653 | 0.473 | 0.040 |
| 0.002 | 33 | 1.001 | 2.779 | 0.461 | 0.052 |

The reference value of the mean squared Mahalanobis norm is 3 for a
three-component Gaussian increment. The predicted covariance itself is also
stable between the two perturbation amplitudes: memberwise cross-epsilon
relative differences are below `2.3e-4`, with correlations above
`0.99999998` on the common pre-crossing intervals.

These numbers support the microscopic multiplicative-noise term and reject
the deterministic tangent closure on the differentiable pre-crossing window.
They do not establish a long-time stochastic closure because the hard cutoff
removes most of the available horizon.

## Next falsification test

The next control is a many-particle Langevin run with a pair potential smooth
through the derivative order required by `L2F`. It must use the same seeds,
perturbations, precision, and covariance diagnostic. A successful control
should retain the complete horizon and reproduce the covariance calibration;
failure would identify missing microscopic terms rather than a cutoff
artifact.
