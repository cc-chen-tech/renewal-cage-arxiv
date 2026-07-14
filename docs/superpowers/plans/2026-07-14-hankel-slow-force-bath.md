# Hankel Slow-Force Bath Implementation Plan

1. Add focused tests for force Hankel assembly, training-only temporal PCA,
   covariance-contracted transition fitting, covariance-consistent residual
   recoloring, and a synthetic two-timescale bath recovery case.
2. Implement the reusable slow-force bath routines in
   `src/ka_slow_force_bath.py`.
3. Add a CLI that validates the four-clone manifest, recomputes exact KA pair
   forces, performs leave-one-clone-out fits, propagates autonomous paths, and
   writes detail/summary curves without fitting macro observables.
4. Run the fixed rank sweep on the existing `10 tau` trajectories and inspect
   the preregistered rank-8 verdict.
5. Add a claim-limited report and package regression gate.
6. Run focused and full tests, `py_compile`, and `git diff --check`; commit only
   the explicit design, implementation, tracked summaries, and report.

