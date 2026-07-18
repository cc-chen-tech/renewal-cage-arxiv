# Deterministic L2p Conditional-Diffusion Jacobian Design

## Status and scope

This design revises only the numerical estimator used by the approved
`L2p` conditional-diffusion gate. It does not change the trajectories,
targets, held-out split, physical covariance models, score tolerances, or
claim boundaries frozen in
`2026-07-19-l2p-conditional-diffusion-gate-design.md`.

The real T=0.58 numerical canary used the exact frozen four-clone data and
matched drift/second-generator caches. On clone 1, two frames, 64 targets,
and nested probe counts 4/8/16/32, the 16-to-32 relative Frobenius errors were

```text
median = 0.25341121305104475
p95    = 0.48285826425022704
```

against frozen limits 0.10 and 0.25. Therefore the stochastic 32-probe
Hutchinson estimator fails before any held closure score is inspected. This
is a numerical-estimator result, not evidence for or against a microscopic
mechanism. Probe counts and tolerances must not be retuned after seeing it.

## Microscopic object

Let `R,V in R^(3N)` obey the underdamped many-particle Langevin equation

```text
dR = V dt
dV = [F(R) - gamma V] dt + sqrt(2 gamma T) dW,
```

and let `u(R) in R^3` be the fixed smooth cage-relative coordinate already
implemented by `smooth_force_support_cage`. Define

```text
J = grad_R u,
p = J V,
H = grad_R^2 u,
T3 = grad_R^3 u.
```

For the generator

```text
L = V.grad_R + [F-gamma V].grad_V + gamma T Delta_V,
```

direct differentiation gives

```text
L p = J F - gamma J V + H[V,V]

L^2 p = J DF[V]
      + T3[V,V,V]
      + 3 H[V,F]
      - 3 gamma H[V,V]
      - gamma J F
      + gamma^2 J V
      + 2 gamma T Delta_R u.
```

The Ito term is configuration-dependent but velocity-independent. Hence for
an arbitrary velocity direction `eta`,

```text
D_V(L^2 p)[eta]
  = J DF[eta]
  + 3 T3[V,V,eta]
  + 3 H[eta,F]
  - 6 gamma H[V,eta]
  + gamma^2 J eta.
```

This response is linear in `eta`. Its full matrix is

```text
A_c(R,V) = grad_V(L^2 p),
```

and the exact conditional quadratic-variation rate inherited from the
microscopic velocity noise is

```text
Q_c(R,V) = 2 gamma T A_c A_c^T.
```

No random velocity probes enter this expression.

## Deterministic construction

The cage coordinate already supplies an analytic `J(R)`. Only derivatives of
that analytic Jacobian are differenced. With a configuration step `h`,

```text
D_R J[F] ~= [J(R+hF) - J(R-hF)] / (2h)
D_R J[V] ~= [J(R+hV) - J(R-hV)] / (2h)
D_R^2 J[V,V] ~= [J(R+hV) - 2J(R) + J(R-hV)] / h^2.
```

The resulting matrix is

```text
A_c ~= J DF
     + 3 D_R J[F]
     - 6 gamma D_R J[V]
     + gamma^2 J
     + 3 D_R^2 J[V,V].
```

`J DF` is not finite-differenced. For each unordered active KA pair `(i,j)`
with pair-potential Hessian `K_ij`, its action is

```text
(J_j - J_i) K_ij (eta_i - eta_j).
```

Accumulating this expression directly constructs all particle blocks of
`J DF` while preserving the actual many-body pair geometry and the selected
`ka_lj_cut` or `ka_lj_c3_switch` force protocol.

## Numerical validation gates

The deterministic implementation remains a numerical approximation until
the `J` derivative step is shown to converge. Validation is ordered and
fail-closed:

1. **Analytic Jacobian assembly.** A batched full `J` must match the existing
   scalar cage Jacobian for every target and must reproduce projected vector
   responses and Jacobian Gram matrices.
2. **Pair-Hessian contraction.** Applying the constructed `J DF` matrix to an
   arbitrary `eta` must match projection of the existing microscopic
   `ka_lj_*force_generator*` response, for both force protocols.
3. **Directional identity.** Applying `A_c` to fixed random directions must
   match `smooth_cage_l2p_velocity_directional_derivative_batch` on synthetic
   configurations. This comparison is independent of held covariance scores.
4. **Step convergence.** A predeclared step ladder is evaluated on fixed real
   frames. The primary step is accepted only if adjacent-step relative errors
   pass frozen median and p95 limits documented before the held run.
5. **PSD and provenance.** Every `Q_c` must be finite, symmetric, positive
   semidefinite to numerical tolerance, and tied to the exact trajectory hash,
   target list, potential protocol, and derivative step.
6. **Performance canary.** A real-frame timing run must show that completing
   all four 200-frame caches is operationally feasible before launching it.

If any numerical gate fails, held closure classification is not run and no
physical interpretation is assigned.

### Frozen real-frame canary

The following controls were fixed after the synthetic directional-identity
test and before evaluating any deterministic tensor on a real trajectory
frame:

```text
configuration-step ladder = {1e-4, 3e-5, 1e-5, 3e-6}
primary step              = 1e-5
reference step            = 3e-6
direction seeds           = 20260721, 20260722, 20260723, 20260724
directional velocity step = 2e-5
directional position step = 1e-5
directional phase step    = 3e-6
```

For the fixed first real frame and all 64 frozen targets:

```text
A(primary) vs A(reference): median <= 0.02 and p95 <= 0.10
Q(primary) vs Q(reference): median <= 0.02 and p95 <= 0.10
```

The per-target relative response error of `A(primary) eta` against the
existing full `L2p` directional derivative, pooled over the four fixed
directions, must also satisfy median <= 0.02 and p95 <= 0.10. The median
`A` and `Q` errors relative to the reference must not increase when moving
from `3e-5` to `1e-5`. Every primary `Q` must satisfy

```text
lambda_min(Q) >= -1e-10 max(trace(Q), 1).
```

These are numerical-only limits. Failure stops the held closure run and does
not authorize changing a physical claim flag or retuning a tolerance.

### Post-run real-frame outcome

The frozen canary was then run on clone 1, timestep 0, all 64 fixed targets,
and trajectory SHA256
`a77d5177fa9d632ef97a680d4f953885d10ed6e175ae76487b24861160dcce18`.
No held closure score was evaluated. The mechanically cached outcome was

```text
A primary/reference median = 5.498628244528567e-7
A primary/reference p95    = 9.403464208072587e-7
Q primary/reference median = 5.156498949931173e-7
Q primary/reference p95    = 9.91161939117552e-7
directional median         = 6.393326966820003e-4
directional p95            = 2.402920962324348e-3
directional maximum        = 5.912976339950533e-3
minimum eigenvalue(Q)      = 1.0416182709402591e5
```

Both median step errors decreased from `3e-5` to `1e-5`. All frozen
numerical gates passed. A primary frame required about seven seconds in the
real-frame timing run; the first sensitivity frame additionally paid for the
reference/coarse matrices and four independent directional checks. The
checkpoint cache was 19 KiB and a completed one-frame rerun exited without
recomputing the frame.

## Claim boundary

This estimator tests whether state-dependent microscopic diffusion inherited
from `L^2 p` helps whiten the held residual. It does not by itself prove an
autonomous one-particle Langevin equation, a unique environment coordinate,
Kramers escape, a complete event clock, or any thermodynamic glass transition.

The following remain fixed at zero:

```text
microscopic_environment_coordinate_z_allowed = 0
continuous_gaussian_langevin_bath_allowed = 0
autonomous_single_particle_gle_allowed = 0
complete_event_clock_closure_allowed = 0
kramers_escape_claim_allowed = 0
thermodynamic_claim_allowed = 0
```
