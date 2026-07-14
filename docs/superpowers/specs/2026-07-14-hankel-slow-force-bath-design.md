# Hankel Slow-Force Bath Design

## Objective

Test whether a stable auxiliary bath extracted only from the exact
many-particle KA pair-force history can extend the microscopic tagged-particle
closure from the verified `0.32 tau` window to `4 tau`, while also reproducing
the fixed `p_hop` event rate.

This is the next branch after the finite local generator chain

```text
(x, v, F, L F, L2 F)
```

failed held-out response.  It tests explicit slow memory, not another local
time derivative and not a phenomenological renewal variable.

## Microscopic Starting Point

For a tagged A particle in the full C3-compatible KA Langevin system,

```text
d r_i = v_i dt
d v_i = [F_i(R) - gamma v_i] dt + sqrt(2 gamma T) dW_i.
```

Eliminating the other particles by Mori-Zwanzig gives a generalized Langevin
equation with a memory term and an orthogonal force.  A rational approximation
to the memory is equivalent to an extended Markov system with auxiliary bath
variables.  Here those variables are not free latent parameters: they are
linear combinations of the measured conservative pair-force history.

For saved-frame spacing `Delta t`, define the force Hankel vector

```text
h_n = [F_n, F_(n-1), ..., F_(n-p+1)].
```

The training-only covariance eigensystem gives temporal modes `U_r`, and

```text
z_n = U_r^T [h_n - mean(h)].
```

The reduced state is

```text
s_n = [v_n, z_(1,n), ..., z_(r,n)].
```

Thus every auxiliary coordinate has a deterministic observable map back to
the exact force exerted by the other 4095 particles.

## Stable Covariance Closure

Pool training samples over particles, Cartesian components, and training
clones.  Let

```text
C = <s_n s_n^T>,
C10 = <s_(n+1) s_n^T>.
```

In covariance-whitened coordinates the least-squares propagator is

```text
A_w = C^(-1/2) C10 C^(-1/2).
```

Finite data may violate stationarity.  Clip only singular values above
`rho_max=0.999`, then transform back to obtain `A`.  This enforces

```text
Q = C - A C A^T >= 0.
```

The training residuals are whitened by their empirical covariance and
recolored by `Q`.  Sampling full `(mode, Cartesian)` residual blocks preserves
their measured non-Gaussian and cross-component shape while satisfying the
stationary covariance identity.  This is a covariance-consistent orthogonal
noise model, not a claim that the generalized FDT has been proved.

## Data Protocol

Use the existing four independent `10 tau` isoconfigurational KA Langevin
clones at `T=0.58`, saved every `0.01 tau`, with conservative forces recomputed
from all 4096 coordinates.

- choose 64 fixed A particles once with a pinned seed;
- use leave-one-clone-out fitting;
- fit no MSD, NGP, scattering, event, or diffusion observable;
- use a fixed Hankel support of 64 frames (`0.64 tau`);
- primary slow-bath rank: 8;
- diagnostic ranks: 2, 4, and 16;
- control: covariance-contracted raw force delay of order 2;
- propagate autonomous trajectories to `4 tau`.

The support is fixed from the independently measured constrained-force kernel
range and is not selected from the macro prediction score.

## Observables And Gates

For each held clone report:

- held one-step state `R2`;
- transition spectral radius;
- stationary covariance identity error;
- maximum held residual-state correlation;
- terminal MSD/diffusion relative error at `4 tau`;
- maximum NGP absolute error;
- maximum relative error of `F_s(k,t)` for `k=1,3,7.25`;
- raw-particle `p_hop` event-rate relative error using fixed half-window
  `0.4 tau`, threshold `0.08`, and non-recrossing collapse.

The primary rank-8 model passes only if all four held clones satisfy numerical
integrity and the aggregate errors obey

```text
diffusion_relative_error <= 0.20
multi_k_fs_max_relative_error <= 0.20
ngp_max_absolute_error <= 0.10
event_rate_relative_error <= 0.20
maximum_held_residual_state_correlation <= 0.20
```

The covariance identity error must be at most `1e-8` and the spectral radius
at most `1 + 1e-8`.

## Claim Boundary

Passing would establish a long-window microscopic slow-bath embedding for the
tested raw tagged-particle observables.  It would not yet derive persistence
versus exchange decoupling, SE violation across temperatures, Kramers escape,
or thermodynamics.  The following remain false in this phase:

```text
state_dependent_memory_allowed
complete_event_clock_closure_allowed
kramers_escape_claim_allowed
thermodynamic_claim_allowed
```

If the rank-8 slow bath fails while covariance stability and linear-state
identification pass, the next model must introduce nonlinear state-dependent
feature coupling rather than more stationary linear modes.

