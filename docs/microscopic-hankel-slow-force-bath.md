# Microscopic Hankel Slow-Force Bath

## Question

Can the missing cage-scale memory be represented by a finite stable bath made
only from the exact conservative force history of a tagged particle?  This is
the direct alternative to adding more local generator derivatives after the
`(x,v,F,LF,L2F)` closure failed.

## Derivation

The full 4096-particle KA Langevin equation is

```text
d r_i = v_i dt
d v_i = [F_i(R) - gamma v_i] dt + sqrt(2 gamma T) dW_i.
```

Mori-Zwanzig elimination of the other particles produces memory and an
orthogonal force.  A rational memory approximation can be written as an
extended Markov system.  Here its auxiliary coordinates are constrained to be
measurable functions of the exact pair force.  On the saved grid,

```text
h_n = [F_n, F_(n-1), ..., F_(n-p+1)],
z_n = U_r^T [h_n - mean(h)],
s_n = [v_n, z_n].
```

`U_r` is obtained by temporal PCA of training-clone force histories only.  No
MSD, NGP, scattering, diffusion coefficient, or event label enters either the
basis or the dynamics fit.

For the pooled training covariance `C=<s_n s_n^T>` and lag covariance
`C10=<s_(n+1)s_n^T>`, the propagator is fitted in covariance-whitened
coordinates.  Singular values above `0.999` are clipped before transforming
back.  Consequently,

```text
Q = C - A C A^T >= 0
```

and autonomous noise is obtained by whitening full empirical residual blocks
and recoloring them with `Q`.  This preserves measured non-Gaussian block
shape while satisfying the stationary covariance identity.  It is not a
generalized-FDT claim.

This construction follows the physical logic of rational GLE embeddings
([Lei, Baker, and Li, 2016](https://arxiv.org/abs/1606.02596)), but imposes an
additional held-out requirement absent from a pure correlation fit.  The
alternative of nonlinear feature-coupled memory is motivated by
[Ge, Zhang, and Lei, 2023](https://arxiv.org/abs/2310.18582).

## Protocol

Four independent `10 tau` KA Langevin clones at `T=0.58` are saved every
`0.01 tau`.  Conservative force is recomputed from every full configuration
for 64 fixed A particles.  Every fold learns its temporal basis and transition
from three clones and holds out the fourth.

The preregistered primary model uses 64 force frames (`0.63 tau`) and eight
Hankel modes.  Ranks 2, 4, and 16 and the raw order-2 force delay are controls.
An exploratory extension tests ranks 24, 32, 48, and 64.  Autonomous paths are
propagated to `4 tau`.  The fixed event definition is raw-particle `p_hop`,
half-window `0.4 tau`, threshold `0.08`, with non-recrossing collapse.

## Primary Result

The rank-8 model resolves `0.87341` of force-history variance.  Its numerical
construction is stable in every fold: the maximum spectral radius is `0.99491`
and the stationary covariance relative error is below `3.5e-15`.  Held
one-step velocity `R2` is `0.96192`.

The apparent instantaneous orthogonality is misleading.  Maximum held
residual-state correlation is only `0.12436`, but maximum held residual lag
correlation over 1--16 frames is `0.93739`.  The unresolved force is therefore
not white after projection onto these slow modes.

| model | force variance | diffusion error | max `F_s` error | max NGP error | event-rate error |
|---|---:|---:|---:|---:|---:|
| raw delay 2 | 1.00000 | 2.5058 | 0.7584 | 0.2675 | 0.9550 |
| rank 2 | 0.34565 | 52.0939 | 1.0139 | 0.2414 | 20.0218 |
| rank 4 | 0.60766 | 44.0377 | 1.0092 | 0.2543 | 20.3055 |
| rank 8 | 0.87341 | 21.2371 | 1.0037 | 0.2477 | 17.1130 |
| rank 16 | 0.98965 | 3.0262 | 0.8214 | 0.2444 | 0.5090 |

The low-rank models create a slowly persistent velocity component and grossly
overpredict displacement and events.  Increasing rank suppresses this error,
but rank 16 still fails every physical gate and its residual correlations are
larger.

## Rank Extension

The failure is not removed by approaching the full 64-frame delay space.
Ranks 24, 32, 48, and 64 resolve `0.99913`, `0.99987`, `0.99998`, and `1.0` of
force-history variance, but their diffusion errors are `2.732`, `3.229`,
`4.139`, and `4.358`.  Their event-rate errors also increase from `1.184` to
`4.059`.  Thus prediction does not converge with explained force variance.
The rank sweep cannot be used to select a post-hoc closure.

## Physical Verdict

A stationary linear auxiliary bath can be made stable and can predict one
saved step extremely well, but it does not close cage-scale transport.  The
decisive diagnostic is the strongly colored held orthogonal residual.  The
missing object is not another linear combination of the same finite force
history; it is a conditional orthogonal process whose evolution changes with
the cage/bath state.

The next admissible equation is therefore nonlinear state-dependent memory,
schematically

```text
dv(t) = F_mean(a(t)) dt
        - integral Phi(a(t))^T K(t-s) Phi(a(s)) v(s) ds dt
        + Phi(a(t))^T dR(t),
```

where `a(t)` must be a continuous microscopic cage coordinate and the feature
couplings and residual law must pass held-clone tests.  A scalar switching
label or an independently sampled event-age noise law is already ruled out by
earlier experiments.

```text
hankel_slow_force_bath_allowed = 0
state_dependent_memory_allowed = 0
complete_event_clock_closure_allowed = 0
kramers_escape_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

This result does not reject a single-particle generalized Langevin reduction.
It rejects the homogeneous finite linear bath as that reduction.
