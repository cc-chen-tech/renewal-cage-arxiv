# Same-Label Microscopic Soft-Mode Precursor Test

## Question

The radial residual test showed that isotropic A/B shell packing does not
close the smooth-cage escape state. This experiment tests a genuinely
different instantaneous hypothesis:

> Does directional participation in exact local KA Hessian modes predict
> smooth-center escape across unseen parent configurations?

The answer is **no**. Both the Hessian-only state and its combination with the
exact smooth-cage geometry degrade held-parent prediction.

## Local Hessian state

For every target A particle, all particles within radius `1.5` form an active
cluster. The exact pair-potential Hessian is assembled for active coordinates,

```text
H_ab = d2 U_KA / (d r_a d r_b).
```

Interactions with particles outside the cluster remain in the diagonal
blocks. The exterior is therefore pinned, not discarded. After diagonalizing
`H`, the first and fourth eigenmodes above `1e-6` define

```text
L_q = log(1/lambda_q),
P_q = log[(sum_alpha |e_q(i,alpha)|^2)/lambda_q],
q in {0,3}.
```

The state also includes log cluster size and the number of eigenvalues no
larger than `1e-6`. These six coordinates measure directional softness,
target participation, environment size, and instantaneous local instability.
They are deterministic functions of the full frame-zero configuration and
are exactly clone-invariant.

No event outcome selected the cutoff, ranks, or eigenvalue floor. These values
were inherited from prior local-Hessian work.

## Why this test was still needed

Related tests already existed:

- a frozen-minimum committor soft-mode state had held-parent Brier skill
  `-0.0382`;
- an adiabatic Hessian Schur-complement state had skill `-0.0140` on its valid
  subset;
- a one-trajectory instantaneous soft-mode hazard canary had skill near zero;
- inherent harmonic modes and harmonic-seeded cubic barrier proxies failed to
  predict constrained-force memory labels.

None used the current five-parent, 320-configuration smooth-center escape
labels. The present experiment supplies that missing like-for-like comparison.

## Fixed data and validation

```text
temperature                  = 0.58
parents                      = 5 independent restarts
clones per parent            = 8
fixed A targets per parent   = 64
clone survival rows          = 2560
parent-target propensity rows = 320
events / censors             = 1731 / 829
horizon                      = 20 tau
p_hop threshold              = 0.08
half-window                  = 8 frames
regularization               = L2 = 1
validation                   = leave one complete parent out
```

The compared models are `geometry=(u,G)`, `softmode`, and
`geometry_softmode`. The geometry metrics reproduce the previous checkpoint
with maximum absolute error `0.0`; clone positions differ by maximum `0.0`.

## Result

### Clone-level censored first escape

| State | Held-parent Brier skill | Likelihood gain / observation | Minimum parent likelihood gain | Maximum survival error |
|---|---:|---:|---:|---:|
| geometry `(u,G)` | `0.00836` | `0.00646` | `1.2947` | `0.06743` |
| softmode | `-0.00799` | `-0.00346` | `-5.6009` | `0.06741` |
| geometry + softmode | `-0.00087` | `0.00199` | `-3.7375` | `0.06596` |

The combined held-parent rows are:

| Parent | Events / 512 | Brier skill | Likelihood gain | Survival error |
|---:|---:|---:|---:|---:|
| 1 | `341` | `0.00487` | `1.7662` | `0.02600` |
| 2 | `346` | `-0.01211` | `-3.7375` | `0.04736` |
| 3 | `364` | `-0.01909` | `-2.4768` | `0.06596` |
| 4 | `328` | `0.00554` | `3.3896` | `0.06513` |
| 5 | `352` | `0.01643` | `6.1475` | `0.04650` |

The same fixed descriptor changes sign across parents. It is not a
transferable readiness state.

### Configuration-level propensity

| State | Held-parent binomial Brier skill | Likelihood gain / clone trial |
|---|---:|---:|
| geometry `(u,G)` | `0.00848` | `0.00442` |
| softmode | `-0.00774` | `-0.00428` |
| geometry + softmode | `-0.00096` | `-0.00064` |

The aggregated result removes repeated-clone weighting as an explanation. The
directional local Hessian state still worsens prediction.

## Physical conclusion

The Hessian is a microscopic object, but microscopic exactness does not imply
that a few eigenvalue-derived scalars form a sufficient reduced state. The
many-particle Langevin bath contains mode amplitudes, phases, nonlinear mode
couplings, and history. An instantaneous spectrum discards them.

Taken together, the following local instantaneous routes now fail held-out
tests:

```text
smooth geometry (u,G)
species-resolved radial packing
instantaneous pinned-cluster Hessian modes
frozen-minimum soft modes
adiabatic Hessian feedback
harmonic-seeded cubic barriers
```

The disciplined response is to **stop adding scalar static local descriptors**.
This does not mean structure is irrelevant. It means the relevant reduced
state is dynamical or multivariate and cannot be represented by one frozen
local landscape number.

## Dynamic-memory route

For the tagged smooth-cage coordinate, eliminating the remaining particles by
Mori-Zwanzig gives the exact form

```text
du/dt = p,

m dp/dt = F_PMF(u)
          - integral_0^t K[u(t),s] p(t-s) ds
          + R(t),
```

where the orthogonal force `R` and memory kernel `K` retain the eliminated
bath. At equilibrium their conditional covariance must satisfy the
fluctuation-dissipation relation. A Markov single-coordinate Langevin equation
is only the special case in which `K` collapses to a delta function and `R` is
white Gaussian noise; the current evidence does not support that limit.

The next experiment must identify slow auxiliary bath coordinates from
microscopic time correlations, not from escape labels. A finite-dimensional
embedding has the form

```text
m dp/dt = F_PMF(u) + sum_a c_a z_a + eta_0,
dz_a/dt = -z_a/tau_a + d_a p + eta_a,
```

which generates a sum-of-exponentials memory kernel after the `z_a` variables
are eliminated. The number and timescales of `z_a` must be selected from
held-parent orthogonal-force correlation spectra. These are memory states of
one tagged-particle equation, not additional phenomenological cage particles.

Existing tests show that a short pair-force delay closes observables only to
`0.32 tau` and fails at multi-tau lags. The next advance therefore requires a
state-dependent or nonlinear slow bath embedding, followed by held-parent
tests of diffusion, NGP, multi-k scattering, and the event clock.

## Claim boundary

```text
instantaneous_local_softmode_precursor_allowed = 0
event_clock_claim_allowed = 0
autonomous_single_particle_gle_claim_allowed = 0
kramers_escape_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

This result closes the tested instantaneous local-scalar branch. It does not
yet provide the required dynamic-memory single-particle Langevin closure.
