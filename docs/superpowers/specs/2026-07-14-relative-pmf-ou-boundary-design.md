# Relative PMF And Markov-OU Boundary Design

## Objective

Test whether the smooth cage-interior coordinate has a microscopic equilibrium
mean-force closure and whether that static closure is sufficient for Markovian
OU dynamics.

## Equilibrium Identity

For isotropic constant relative covariance `Q_u=d I`,

```text
du = p dt,
dp = [a(u) - gamma p] dt + sqrt(d) dW.
```

Stationarity gives

```text
sigma_p^2 = d / (2 gamma),
a(u) = sigma_p^2 grad_u log rho(u).
```

If microscopic state counting gives an isotropic Gaussian
`rho(u) proportional exp[-|u|^2/(2 sigma_u^2)]`, then

```text
a(u) = -kappa u,
kappa = sigma_p^2 / sigma_u^2.
```

This acceleration prefactor differs from the naive `T/sigma_u^2` because `p`
has an effective projected mass `M=T/sigma_p^2`.

## Held-Clone Tests

Fit `sigma_u^2`, `sigma_p^2`, and the radial density on three clones. On the
held fourth clone require:

- FDT variance error below `0.03`;
- radial log-density quadratic coefficient within `0.03` of `-1/2`;
- binned conditional mean-force correlation above `0.98`;
- normalized binned RMSE below `0.10`;
- every clone passes.

The temperature-naive acceleration is an explicit null. Separately compare
the parameter-free Markov OU correlation matrix at lags from 1 to 100 frames.
PMF closure does not promote Markov dynamics if its maximum correlation error
exceeds `0.20` or the mean-force residual lag exceeds `0.20`.
