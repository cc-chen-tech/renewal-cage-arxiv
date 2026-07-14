# Decomposed Cage-Drift Bath Design

## Objective

Replace the raw tagged-force history by the exact microscopic drift split
implied by `x_i=C_i+u_i`, then test whether separate cage-center and relative
histories remove the colored residual and recover autonomous observables.

## Exact Drift And Noise Split

For the smooth cage coordinate `u_i(R)` with Jacobian `J_i`, let

```text
p_i = J_i V,
w_i = v_i - p_i.
```

The deterministic drifts are

```text
b_u = J_i F + Hess(u_i):(V tensor V) - gamma p_i,
b_C = F_i - gamma v_i - b_u.
```

Writing `A_i` for the tagged-particle block of `J_i` and `G_i=J_i J_i^T`,
the relative, center, and center-relative noise covariance rates are

```text
Q_u  = 2 gamma T G_i,
Q_C  = 2 gamma T [I - A_i - A_i^T + G_i],
Q_Cu = 2 gamma T [A_i^T - G_i].
```

They must form a positive-semidefinite `6x6` block covariance and obey

```text
Q_C + Q_u + Q_Cu + Q_Cu^T = 2 gamma T I.
```

The geometric drift is evaluated by a fixed centered directional derivative
of `J(R)V`; convergence is checked on a held set of frames with steps
`5e-6`, `1e-5`, and `2e-5`.

## Resolved Models

Use the same four 10 tau clones and 64 fixed A particles. Compare:

```text
raw-H16 = [v, z_total(16)],
split-8 = [w, p, u, z_C(8), z_u(8)],
split-16 = [w, p, u, z_C(16), z_u(16)].
```

Every `z` is a training-only temporal PCA mode of a 64-frame deterministic
drift history. The raw model is the existing exact-force baseline. The split
model reconstructs tagged velocity as `v=w+p` and integrates that sum in
autonomous simulation.

## Validation

Report leave-one-clone-out:

- tagged-velocity one-step `R2`;
- mode-resolved residual-state and lag correlations;
- diffusion, NGP, multi-k `F_s`, and nonrecrossing event rate;
- spectral radius and stationary covariance identity;
- drift-step sensitivity and exact noise reconstruction error.

Promotion requires the same residual and macro gates as the smooth-cage
Hankel experiment. No event labels or macro observables enter PCA, transition,
or innovation fits.

The following remain false unless all relevant gates pass:

```text
decomposed_cage_drift_bath_allowed
autonomous_single_particle_gle_allowed
complete_event_clock_closure_allowed
kramers_escape_claim_allowed
thermodynamic_claim_allowed
```
