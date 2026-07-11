# Stationary Finite-Flight Renewal Design

## Purpose and scope

The existing `PersistenceExchangeParams` model is a useful nonstationary delayed-first null: it assigns an independent exponential first wait with mean `tau_p` and exponential later waits with mean `tau_x`. It remains unchanged for reproducibility. This extension adds an equilibrium stationary renewal closure, finite-duration cage flights, and event-space jump correlations needed to confront continuous-time Kob-Andersen trajectories.

The result is an effective diagnostic theory for dynamical glass signatures. It does not derive configurational entropy, a Kauzmann transition, or any thermodynamic glass transition; every new evidence row therefore retains `thermodynamic_claim_allowed=0`.

## Public interfaces

- `StationaryRenewalParams(exchange_mean, exchange_cv2)` validates positive mean and nonnegative squared coefficient of variation. It derives `persistence_mean = 0.5 * exchange_mean * (1 + exchange_cv2)` and `persistence_exchange_ratio = 0.5 * (1 + exchange_cv2)` from the equilibrium residual-life identity.
- `stationary_gamma_count_moments(times, params)` uses a gamma exchange law with shape `1 / exchange_cv2` and its equilibrium residual distribution. It returns numerical mean and variance of the stationary count while enforcing the exact first moment `E[N(t)] = t / exchange_mean`.
- `finite_flight_weight_integral(time, duration, moment_order)` returns the integrated mark weight
  `t - duration + 2*duration/(moment_order + 1)` for `t >= duration`, and
  `duration*(t/duration)**moment_order*(1 - (t/duration)*(moment_order - 1)/(moment_order + 1))` otherwise. Zero duration gives the instantaneous limit `time`.
- `finite_flight_moments_1d(...)`, `finite_flight_ngp_1d(...)`, and `finite_flight_self_intermediate_scattering(...)` accept stationary count cumulants and measured mark moments or a discrete mark distribution. Cage motion remains Gaussian and independent.
- `event_space_correlated_diffusion(event_rate, jump_squared_mean, jump_dot_correlations, dimension=3)` evaluates
  `D = event_rate * (jump_squared_mean + 2*sum(jump_dot_correlations)) / (2*dimension)` and rejects a nonpositive Green-Kubo bracket.

## Compatibility and data policy

No existing signature or numerical output changes. The legacy persistence/exchange functions are explicitly described as nonstationary delayed-first diagnostics. Raw Zenodo trajectory files are not committed. The repository stores only source metadata, fixed extraction settings, derived observables, uncertainty/robustness columns, model predictions, and pass/fail verdicts for `T=0.58` and `T=0.45`.

## Acceptance tests

- The stationary clock satisfies `E[N(t)] = t/exchange_mean` and the residual-life ratio exactly.
- The finite-flight kernel reaches the instantaneous limit as duration tends to zero and reproduces its analytic piecewise values.
- Finite-flight NGP tends to zero at long time and improves the fixed real-data transient canary relative to instantaneous jumps.
- The event-space Green-Kubo correction reproduces the held-out `T=0.45` diffusion coefficient within 2%, while the uncorrected event estimate records its failure.
- Derived data record cooling-induced diffusion slowdown, alpha-time growth, SE-product growth, NGP growth/shift, and overlap-susceptibility growth without promoting the overlap proxy to a full spatial four-point theory.
- Unit tests, package tests, artifact generation, SVG/PDF rendering checks, and the arXiv package manifest pass.

