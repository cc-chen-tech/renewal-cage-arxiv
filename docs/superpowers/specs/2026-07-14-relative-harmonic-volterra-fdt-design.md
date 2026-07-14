# Relative Harmonic Volterra And FDT Design

## Objective

Test whether the non-Markovian relative dynamics can be represented by a
scalar harmonic GLE after removing a microscopic isoconfigurational cage bias.
Separate held correlation prediction from the stronger generalized-FDT claim.

## Coordinate And Kernel

Estimate the particle-specific bias from training clones only,

```text
u0_i = mean over training clones and time of u_i,
delta u_i = u_i - u0_i.
```

For fixed PMF stiffness `kappa`, bare friction `gamma`, and scalar kernel `K`,

```text
d delta_u / dt = p,
dp/dt = -kappa delta_u - gamma p
        - integral K(s) p(t-s) ds + eta.
```

Orthogonality to the initial resolved state gives a triangular Volterra
equation for training `C_pp(t)` and `C_up(t)=<delta_u(t)p(0)>`. Use causal
trapezoid quadrature, a fixed `1 tau` kernel, and predict held correlations to
`8 tau`.

## Distinct Gates

The correlation gate requires every-fold extrapolation RMSE below `0.08` and
maximum correlation error below `0.20`. The physical Mori/FDT gate additionally
requires held reconstructed random-force covariance to match
`<eta(t)eta(0)>=sigma_p^2 K(t)` with variance ratio in `[0.8,1.2]`, normalized
RMSE below `0.20`, and shape correlation above `0.8`.

Passing the correlation gate alone permits only a predictive Volterra
representation. It must not be promoted to a physical scalar GLE.
