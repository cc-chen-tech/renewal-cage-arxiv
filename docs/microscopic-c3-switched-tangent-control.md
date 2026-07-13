# Microscopic C3-Switched KA Tangent Control

## Question

This control asks whether the tagged-particle tangent noise used in the
microscopic bridge follows from the full 4096-particle Langevin dynamics, or
whether the earlier long-horizon failure was caused by the non-differentiable
`lj/cut` boundary. It does not fit an effective stochastic process to MSD or
NGP. The simulated state remains the complete Kob-Andersen mixture.

The answer is positive for the smooth control over the tested `0.2 tau`
horizon: the parameter-free Hessian/FDT covariance passes on every saved
interval. This is a microscopic tangent-dynamics result, not a derivation of
the renewal hazard.

## Smooth Microscopic Potential

For every KA pair, the Lennard-Jones energy is unchanged below
`r_on = 2.0 sigma`. In the interval from `r_on` to `r_c = 2.5 sigma`, the
protocol `ka_lj_c3_switch` uses

```text
U(r) = V_LJ(r) S(x)
x = (r-r_on)/(r_c-r_on)
S(x) = 1 - 35 x^4 + 84 x^5 - 70 x^6 + 20 x^7.
```

At `r_on`, the switch equals one and its first three derivatives vanish. At
`r_c`, the switch and its first three derivatives vanish. Therefore `U`,
`U'`, `U''`, and `U'''` are continuous. This derivative order is required
because the force uses `U'`, the Hessian uses `U''`, and `L2F` uses `U'''`.

The Python radial derivatives pass endpoint and finite-difference tests. A
pinned Lepton-enabled LAMMPS build evaluates the same analytic energy. Direct
AA, AB, and BB dimer checks at 15 separations gave a maximum absolute
LAMMPS/Python force error of `5.914e-15`; see
[`force parity`](../data/renewal_cage_ka_c3_lepton_force_parity.csv).

## Parent Preparation

The existing hard-cutoff `T=0.58` restart supplied only the starting
configuration. After replacing the pair style, the C3 system was run for
`10 tau` with `gamma=1` and `dt=0.001`. All 102 thermodynamic samples were
finite. Over the final half, the temperature was `0.58333 +/- 0.00746`, its
slope was `-1.13e-3/tau`, and the total-energy slope was `-4.67e-3/tau`; see
[`parent summary`](../data/renewal_cage_ka_c3_parent_T058_tau10_summary.csv).

This is sufficient parent conditioning for a short tangent control. It is not
evidence that the C3 system is an equilibrated glass, and it is not used to
make a thermodynamic claim.

## Microscopic Tangent Equation

The simulated many-particle equation is

```text
d r_i = v_i dt
d v_i = (F_i - gamma v_i) dt + sqrt(2 gamma T) dW_i.
```

For pair Hessian blocks `H_ij`, the first force generator is

```text
LF_i = -sum_j H_ij (v_i-v_j).
```

Under a common-noise displacement experiment, its tangent evolves as

```text
d(delta LF_i)
  = delta(L2F_i) dt
    - sqrt(2 gamma T) sum_j delta(H_ij) (dW_i-dW_j).
```

Thus `delta(L2F)` is only the deterministic drift. The fourth derivative
diagnostic, called `generator_second` in the tables, cannot be required to
equal the observed path derivative by itself. Its missing increment is the
multiplicative tangent noise whose covariance is predicted directly from the
pair Hessians. The original design phrase "four deterministic identities"
was therefore mathematically too strong. The retained summary marks that
literal gate unevaluable instead of silently redefining it after seeing the
data.

## Full-Horizon Result

Eight independent common-noise seed pairs were run at displacement amplitudes
`epsilon=0.001` and `0.002`, with positive and negative paths. All 32 paths
contain 201 frames. Every path hash, protocol marker, `L2F`, and full
target-pair Hessian array passed validation.

At stride 5, all `320/320` intervals per epsilon remain eligible. There are
55 and 76 pair-support crossing intervals, but a C3 support crossing is not a
singularity and causes no right censoring. The principal aggregate results are:

| saved stride | frame time | trace ratio | mean squared Mahalanobis | whitened lag-1 |
|---:|---:|---:|---:|---:|
| 1 | 0.001 | 0.5334 | 1.576 | 0.505 |
| 2 | 0.002 | 0.8092 | 2.368 | 0.172 |
| 5 | 0.005 | 1.0095 | 2.813 | 0.0625 |

The stride-5 trace ratios lie in the preregistered `[0.8,1.2]` interval, the
three-dimensional Mahalanobis values lie in `[2.4,3.6]`, and the absolute
lag-1 correlations are below `0.1`. The two epsilon estimates are also
effectively identical: the maximum memberwise covariance relative difference
is `2.24e-4`, and the minimum correlation is `0.99999997`.

The result is resolution-dependent in a physically informative way. At one
integration step the residual is temporally correlated, so an instantaneous
white-noise approximation is premature. After coarse integration over
`0.005 tau`, the predicted microscopic covariance closes without a fitted
amplitude. Full tables are in
[`stride 1`](../data/renewal_cage_ka_c3_tangent_noise_covariance_stride1_T058.csv),
[`stride 2`](../data/renewal_cage_ka_c3_tangent_noise_covariance_stride2_T058.csv),
[`stride 5`](../data/renewal_cage_ka_c3_tangent_noise_covariance_stride5_T058.csv),
and the [`control summary`](../data/renewal_cage_ka_c3_tangent_control_T058_summary.csv).

## Short-Time Physical Fidelity

A same-seed hard-cutoff audit path and the C3 audit path were compared using
all 4096 particles. The scaled pair-distance histogram has total variation
`0.01564`; the mean force norm changes by `-1.28%`; the mean isotropic cage
curvature changes by `-1.15%`. The A-particle MSD changes by `-0.33%` at
`0.01 tau` and `-3.84%` at `0.2 tau`; see the
[`fidelity summary`](../data/renewal_cage_ka_c3_physical_fidelity_T058_summary.csv)
and [`curves`](../data/renewal_cage_ka_c3_physical_fidelity_T058_curves.csv).

These are single-audit-path, particle-distribution comparisons. They support
short-time control fidelity but do not prove equality of long-time diffusion,
relaxation, NGP, or event statistics. Longer re-equilibration and independent
parents are required before using the switched system for those claims.

## What Is and Is Not Derived

The experiment validates this microscopic link:

```text
many-particle Langevin dynamics
-> exact force/Hessian/L2F tangent drift
-> parameter-free multiplicative tangent-noise covariance.
```

It does not yet derive:

```text
microscopic configuration
-> precursor variables
-> first-passage readiness
-> Kramers-conditioned cage escape
-> renewal hazard and event-clock observables.
```

The next defensible reduction is to project the validated many-particle
tangent response onto a measured cage coordinate and candidate precursor
variables, then test memory, first passage, and escape on held-out paths. A
single-particle Langevin representation can be claimed microscopic only after
its drift, noise, memory, and event statistics survive that projection.

`thermodynamic_claim_allowed = 0`. No result here implies configurational
entropy, a Kauzmann temperature, an ideal-glass transition, or a heat-capacity
singularity.
