# Microscopic Radial Precursor Residual Test

## Question

The exact smooth-cage projection `(u,p,G,b)` did not form a sufficient
initial escape state. This follow-up tested the simplest missing-state
hypothesis without changing the event labels:

> Is species-resolved local packing a transferable microscopic precursor that
> explains escape information absent from the smooth-cage geometry `(u,G)`?

The answer is **no** for the fixed local radial representation. Radial shell
density carries a small signal, but it duplicates rather than extends the
geometry signal on held-parent predictions.

## Microscopic coordinate

For target `i`, species `a` in `{A,B}`, and shell center `rho_n`, the structural
coordinate is

```text
R_i(a,n) = sum_{j != i, type(j)=a}
             f_c(r_ij)
             exp[-(r_ij-rho_n)^2/(2 sigma_r^2)],

f_c(r) = [1 + cos(pi r/r_c)]/2,  r < r_c,
         0,                       r >= r_c.
```

Minimum-image distances use the complete 4096-particle configuration. The
feature grid was inherited from the earlier radial committor test and frozen
before this run:

```text
rho_n   = [0.8, 1.05, 1.3, 1.55, 1.8, 2.05, 2.3]
sigma_r = 0.12
r_c     = 2.5
```

The 14 A/B shell densities are deterministic functions of the initial
many-particle configuration. They are exactly clone-invariant and contain no
velocity, fitted waiting time, or macro-observable input.

## Fixed data and diagnostics

The experiment reuses the exact smooth-center first-passage cache:

```text
temperature                  = 0.58
independent parents          = 5
Langevin clones per parent   = 8
fixed A targets per parent   = 64
clone-level observations     = 2560
parent-target configurations = 320
observed first escapes       = 1731
right-censored observations  = 829
horizon                      = 20 tau
p_hop threshold              = 0.08
half-window                  = 8 frames
regularization               = L2 = 1
validation                   = leave one complete parent out
```

Two tests use the same underlying labels:

1. A censored exponential model predicts each clone's first escape time.
2. A grouped binomial model predicts the number of escaping clones for each
   parent-target configuration.

The compared states are `geometry=(u,G)`, `radial=R`, and
`geometry_radial=(u,G,R)`. The geometry-only implementation reproduces every
metric from the previous checkpoint with maximum absolute error `0.0`.

## Result

### Clone-level censored escape

| State | Held-parent Brier skill | Likelihood gain / observation | Minimum parent likelihood gain | Maximum survival error |
|---|---:|---:|---:|---:|
| geometry `(u,G)` | `0.00836` | `0.00646` | `1.2947` | `0.06743` |
| radial `R` | `0.00825` | `0.00655` | `0.6379` | `0.06437` |
| combined `(u,G,R)` | `0.00798` | `0.00659` | `1.1976` | `0.06100` |

The combined state has positive likelihood transfer in every parent and a
slightly smaller survival error. It nevertheless performs worse than both
components in Brier skill and remains far below the fixed structural reference
`0.026964`. It therefore fails the residual-information gate.

The combined held-parent rows are:

| Parent | Events / 512 | Brier skill | Likelihood gain | Survival error |
|---:|---:|---:|---:|---:|
| 1 | `341` | `0.00832` | `2.4566` | `0.02369` |
| 2 | `346` | `0.01591` | `3.2284` | `0.04741` |
| 3 | `364` | `0.00502` | `1.1976` | `0.06017` |
| 4 | `328` | `0.00120` | `5.8438` | `0.05316` |
| 5 | `352` | `0.00942` | `4.1422` | `0.06100` |

### Configuration-level propensity

| State | Held-parent binomial Brier skill | Likelihood gain / clone trial |
|---|---:|---:|
| geometry `(u,G)` | `0.00848` | `0.00442` |
| radial `R` | `0.00886` | `0.00435` |
| combined `(u,G,R)` | `0.00744` | `0.00370` |

Aggregation removes any ambiguity about repeated structural vectors across
the eight clones. The combined state again gets worse, so local radial packing
does not supply a new isoconfigurational propensity coordinate.

## Physical interpretation

The small positive likelihood gains show that both `(u,G)` and `R` correlate
weakly with escape exposure. The radial coordinates can adjust coarse survival
calibration. They do not improve which particle escapes, which is the necessary
property of a readiness precursor.

This distinction matters. A flexible survival family can improve likelihood
by shifting rates slightly while leaving particle ordering unchanged or worse.
The Brier ablation and the 320-row propensity test directly reject the claim
that radial shell density closes the missing microscopic state.

The result rules out only the fixed local isotropic representation. It does
not rule out configuration dependence generally. The omitted state may be:

1. directional and collective, such as participation in a localized
   low-frequency Hessian mode or an anharmonic escape direction; or
2. historical, such as an orthogonal-force memory variable generated when the
   remaining many-particle coordinates are eliminated.

Adding more radial centers after seeing this result would be post-selection,
not a microscopic derivation.

## Next microscopic branch

The next static test should use a preregistered local-cluster Hessian state on
the same five parents and unchanged escape labels. A radial density records
how many neighbors occupy each shell but discards the directional curvature
matrix and collective mode participation. A Hessian mode is therefore a
genuinely different physical hypothesis, not another radial fit.

If a held-parent soft-mode state explains residual escape propensity, its
amplitude can become a candidate precursor coordinate whose temporal
first-passage law is tested next. If it also fails, the evidence will favor an
explicit non-Markovian reduction. The appropriate single-particle equation is
then a generalized Langevin equation,

```text
m d2u/dt2 = -dW(u)/du
             - integral_0^t K(u(t-s),s) du/dt(t-s) ds
             + eta(t),

<eta(t) eta(t') | u> = T K(u, |t-t'|),
```

with a position-dependent memory kernel estimated from the many-particle
trajectory. A finite-dimensional Markov embedding would introduce auxiliary
bath coordinates only after the measured kernel spectrum determines how many
are required.

## Claim boundary

The machine-readable verdict is

```text
static_radial_precursor_allowed = 0
event_clock_claim_allowed = 0
autonomous_single_particle_gle_claim_allowed = 0
kramers_escape_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

This experiment narrows the missing-state mechanism. It does not yet derive a
renewal clock, Kramers barrier, autonomous single-particle GLE, or thermodynamic
glass transition.
