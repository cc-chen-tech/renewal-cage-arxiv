# Smooth Cage Microscopic Event-Clock Design

## Status and scope

This experiment implements the event-clock gate already preregistered in the
smooth force-support cage projection design. It does not change the cage
weight, fit a jump threshold, or promote the instantaneous projected SDE to an
autonomous single-particle model.

The question is narrower and falsifiable:

> Does the instantaneous microscopic state `(u,p,G,b)` of the smooth cage
> coordinate predict the first nonrecrossing cage-center escape in a parent
> configuration that was absent from model fitting?

A negative result means that the coordinate remains an exact instantaneous
projection but is not a sufficient event-clock state.

## Candidate routes and selection

Three routes were compared.

1. **Five-parent isoconfigurational first passage (selected).** Reuse five
   independent parent configurations and eight `20 tau` Langevin noise clones
   per parent. Reconstruct each clone's initial full state from its recorded
   velocity seed and parent restart. This supplies parent-level holdouts at low
   additional MD cost.
2. **One long physical trajectory with a time split.** This supplies many
   source frames but has stronger temporal dependence and can leak one local
   environment across the split.
3. **A new five-parent C3 ensemble.** This is the cleanest differentiable
   parent protocol but repeats about 640,000 many-particle integration steps
   before the event coordinate has shown any held-out skill.

Route 1 is the decisive low-cost gate. A positive result must later be
repeated under route 3 before an exact C3 microscopic event-clock claim.

## Microscopic data

Use the first eight completed clones from each directory:

```text
tmp/isoconfigurational_canary_T058
tmp/isoconfigurational_replicate_01_T058
tmp/isoconfigurational_replicate_02_T058
tmp/isoconfigurational_replicate_03_T058
tmp/isoconfigurational_replicate_04_T058
```

Each trajectory is a `4096`-particle standard KA system at `T=0.58`, generated
by NVE plus a Langevin thermostat with damping `gamma=1`. It contains 401
frames separated by `0.05 tau`, for a common `20 tau` horizon. The five parent
restart hashes must be distinct. Within each parent, all clones share initial
positions and differ in randomized Maxwell velocities and bath noise.

Select 64 A-particle indices once with seed `20260714`. The same indices must
be A particles in every parent and must be used for every clone. Target count,
event threshold, feature definitions, and regularization are fixed before
examining held-out scores.

## Initial full-state reconstruction

The long trajectory dumps contain positions but not velocities or forces.
Their manifests record the parent restart, temperature, velocity seed, and
Langevin seed. For every selected clone, rerun only the initialization:

```text
read_restart parent.restart
reset_timestep 0
velocity all create 0.58 velocity_seed mom yes rot no dist gaussian
fix integrator all nve
fix bath all langevin 0.58 0.58 1.0 langevin_seed
dump initial all custom ... x y z ix iy iz vx vy vz fx fy fz
run 0
```

The reconstructed frame is accepted only if particle ids/types are aligned
and wrapped/unwrapped positions agree with the original clone's frame zero to
the dump precision. Raw reconstruction dumps are deleted after reduction.

## Smooth cage event coordinate

At every saved frame and selected particle, evaluate the already validated
force-support coordinate

```text
C_i(R) = r_i + sum_j w_ij d_ij / sum_j w_ij,
w_ij = wendland_c4(r_ij / (2.5 sigma_ij)).
```

Use the unwrapped tagged position plus the minimum-image weighted offset so
`C_i(t)` is an unwrapped cage-center path. Apply the existing Candelier event
extractor without modification:

```text
p_hop threshold       = 0.08
half window           = 8 frames = 0.4 tau
recrossing radius     = sqrt(0.08)
event reduction       = contiguous peak plus recursive adjacent A-B-A removal
first-passage horizon = 20 tau
```

No label from the final eight-frame window can be invented; particles without
a retained event are administratively right-censored at `20 tau`.

## Projected microscopic state

For every clone and target at frame zero, calculate the exact instantaneous
projected variables

```text
u = r_i - C_i,
p = J_i V,
G = J_i J_i^T,
b = J_i F + Hess(u_i):(V tensor V) - gamma p.
```

Use the force dumped by LAMMPS in `J_i F`; use the existing centered
directional derivative for the geometric term. Let

```text
g1 <= g2 <= g3
```

be the eigenvalues of `G`. The full rotationally invariant feature vector is

```text
z_full = [
  log(|u|^2),
  log(|p|^2),
  (u dot p)/(|u||p|),
  log(g1), log(g2), log(g3),
  log(|b|^2),
  (u dot b)/(|u||b|),
  (p dot b)/(|p||b|)
].
```

Machine-scale positive floors are fixed in code only to make logs and cosine
denominators finite; they are not fitted parameters.

Preregistered ablations are:

```text
geometry:   [u,G]
kinematic:  [u,p,G]
full:       [u,p,G,b]
null:       one constant escape rate fitted on training parents
```

## Censored exponential reaction-coordinate model

For observation `n`, let `t_n=min(T_n,20)` and `delta_n=1` for an observed
first event and zero for censoring. On training parents, fit

```text
lambda_n = exp(beta_0 + beta dot z_n)
```

by minimizing

```text
L(beta) = sum_n [lambda_n t_n - delta_n log(lambda_n)]
          + (1/2) beta^T P beta,
```

where the intercept is unpenalized, standardized feature coefficients use
fixed `L2=1`, and no horizon/event-rate hyperparameter is selected on held-out
data. The analytic gradient and Hessian are

```text
grad L = X^T(lambda t - delta) + P beta,
Hess L = X^T diag(lambda t) X + P.
```

Refit after withholding each parent. For the held parent, report censored
log-likelihood, finite-horizon Brier score, rate distribution, and survival
calibration at `1,2,4,8,12,16,20 tau`.

## Gates and nulls

The event extraction is usable only if:

- all 40 clone trajectories share their declared grid and pass reconstruction;
- all five parent hashes are distinct;
- every held parent contains both escaped and censored observations;
- at least 64 retained first events occur in total;
- the pooled constant-rate survival maximum absolute error is reported.

The full-state microscopic sufficiency gate passes only if all are true:

```text
mean leave-one-parent-out Brier skill versus constant rate > 0.027
mean held-out censored log-likelihood gain per observation > 0
full-state Brier skill exceeds geometry-only skill by >= 0.01
full-state Brier skill exceeds kinematic skill by >= 0
maximum held-parent survival calibration error <= 0.10
all five held-parent log-likelihood gains are nonnegative
```

The `0.027` threshold is fixed from the best prior relative-neighbor structural
committor result, `0.026964`, before this smooth-state test. It is a difficulty
benchmark, not a fit target.

## Claim boundary and next decision

Passing this experiment permits only:

```text
microscopic_initial_escape_state_allowed = 1
```

It does not yet permit:

```text
event_clock_claim_allowed = 0
autonomous_single_particle_gle_claim_allowed = 0
kramers_escape_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

If the gate passes, the next experiment must evaluate the same state on a
physical-time path and test a time-varying conditional hazard, followed by a
C3 five-parent replication. If it fails, preserve the ablation result and add
missing state variables only when a microscopic residual identifies them;
do not tune the event threshold or add a phenomenological delayed hazard.
