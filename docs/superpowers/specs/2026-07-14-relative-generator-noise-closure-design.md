# Relative Generator Noise Closure Design

## Objective

Close the stochastic generation step of the validated order-40 relative
generator Mori representation without using validation trajectories in fitting
or propagation.

## Protocol

1. Fit bias, scaling, and Mori operators only on discovery clones.
2. Reconstruct the finite-memory innovation
   `xi_(n+1)=g_(n+1)-sum_(ell=0)^40 Omega_ell g_(n-ell)`.
3. Generate autonomous paths by moving-block resampling of consecutive
   innovation vectors from one measured source at a time.
4. Scan block lengths `[1,4,16,40,100,200,400]` by whole-clone leave-one-out
   discovery and select the shortest all-gate block.
5. Freeze the block length and validate on two new many-particle Langevin
   clones with 10,000 simulations and two disjoint Monte Carlo seed sets.

The gate requires covariance RMSE `<=0.08`, covariance maximum error `<=0.25`,
target-correlation RMSE `<=0.08`, target-correlation maximum error `<=0.25`,
marginal excess-kurtosis error `<=0.35`, and maximum absolute state `<=20`.

## Claim Boundary

A held-out pass permits an empirical colored-noise generator and autonomous
simulation of the relative matrix Mori state. It does not permit a thermal-FDT,
microscopic thermal-noise, cage-center, event-clock, Kramers, thermodynamic, or
complete single-particle GLE claim.
