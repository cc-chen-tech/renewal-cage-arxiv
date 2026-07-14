# Relative Generator Mori Design

## Objective

Test whether an exact generator coordinate turns the bias-centered relative
cage dynamics into a held-out matrix Mori representation. Keep correlation
prediction, discrete generalized-FDT identity, thermal FDT, and autonomous
noise generation as separate claims.

## Microscopic Basis

Use the training-clone cage bias and the exact projected Ito drift,

```text
delta u = u - u0
g_phase = [delta u, p]
g_generator = [delta u, p, Lp]
Lp = J F + Hess(u):(V tensor V) - gamma p.
```

No event label or macroscopic observable enters these coordinates.

## Discrete Mori Test

Infer `Omega_k` from training correlation matrices by triangular recursion.
For every held time origin reconstruct

```text
W_k|i = g_(i+k+1) - sum_(ell=0)^k Omega_ell g_(i+k-ell)
```

and test both `W_k` orthogonality to the initial resolved state and the
discrete generalized-FDT identity

```text
Omega_k = -<W_k W_0^T> C(-Delta)^-1.
```

The identity is not promoted to a thermal FDT until the adjoint and
time-reversal-parity structure of the thermostatted process is audited.

## Discovery And Confirmation

Scan memory orders `[1,4,16,32,40]` by leave-one-clone-out on four discovery
clones. Fix the first all-gate candidate before generating two new 10-tau
clones with independent velocity and Langevin seeds. Fit all parameters on
the four discovery clones and score the two new clones once.

The representation gate requires maximum held orthogonality below `0.10`,
GFD normalized RMSE below `0.20`, GFD shape correlation above `0.80`, target
correlation RMSE below `0.08`, and target maximum error below `0.20`.
