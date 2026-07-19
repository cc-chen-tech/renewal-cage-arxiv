# Deterministic microscopic L2p conditional diffusion

## Question

This gate asks whether the state-dependent diffusion inherited directly from
the many-particle KA Langevin generator is sufficient to whiten the held
single-particle `L2p` innovation. It replaces the numerically unresolved
32-probe Hutchinson estimator by the full deterministic velocity Jacobian. The
held covariance families, four-clone split, score tolerances, and claim
boundaries are unchanged.

## Microscopic construction

Start from the underdamped many-particle Langevin dynamics

```text
dR = V dt
dV = [F(R) - gamma V] dt + sqrt(2 gamma T) dW.
```

For a smooth relative cage coordinate `u(R)`, let `J = grad_R u` and
`p = J V`. Direct application of the phase-space generator gives

```text
L p = J F - gamma J V + H[V,V]

L^2 p = J DF[V] + T3[V,V,V] + 3 H[V,F]
        - 3 gamma H[V,V] - gamma JF + gamma^2 JV
        + 2 gamma T Delta_R u.
```

The full velocity Jacobian is therefore

```text
A_c = D_V(L^2 p)
    = J DF + 3 D_R J[F] - 6 gamma D_R J[V]
      + gamma^2 J + 3 D_R^2 J[V,V].
```

`J DF` is evaluated by exact contraction of the cage Jacobian with unordered
KA pair Hessians. The remaining derivatives use the frozen centered
configuration-step ladder. No random velocity probes remain. The thermostat
then fixes the conditional diffusion tensor

```text
Q_c(R,V) = 2 gamma T A_c A_c^T,
```

which is symmetric and positive semidefinite by construction. This is an exact
microscopic noise projection for the selected smooth cage coordinate, not an
assumption of an autonomous one-particle bath.

## Frozen production run

The remote run used four 200-frame `T=0.58` KA trajectories, 64 fixed targets,
`memory_order=40`, `bath_order=16`, and the frozen Jacobian controls

```text
primary step   = 1e-5
reference step = 3e-6
coarse step    = 3e-5
direction seeds = 20260721, 20260722, 20260723, 20260724.
```

Every clone completed `200/200` frames and independently reported
`deterministic_jacobian_numerically_resolved`. Across the four first-frame
canaries, primary/reference median relative errors were `4.32e-7` to
`5.61e-7` for `Q_c`; directional-response median errors were `6.31e-4` to
`6.91e-4`. All step, monotonicity, directional-identity, and PSD gates passed.
The full remote job exited with status zero and no job swaps.

## Held result

The estimator provenance is

```text
conditional_diffusion_estimator = deterministic_velocity_jacobian
l2p_diffusion_numerical_gate_pass = 1
l2p_diffusion_probe_converged = 0
```

The last line is intentionally zero: a deterministic Jacobian must not be
reported as a converged random-probe estimator.

Relative to `constant_full`, the exact tensor improves held negative
log-likelihood in every fold. The replicate-first summary is

```text
mean constant-to-tensor NLL improvement = 0.276920594908
standard error                         = 0.0220691587791
95% t interval                         = [0.206686682091, 0.347154507725]

mean trace-to-tensor NLL improvement    = 0.149501740755
95% t interval                         = [0.132789497961, 0.166213983549]
```

The exact tensor also reduces squared-residual memory by at least 25% in every
fold, rejects the time-permuted tensor null, and keeps the fitted isotropic
floor fraction between `0.2157` and `0.2202`, below the frozen `0.25` limit.
Thus the microscopic tensor contains real time-aligned amplitude and
orientation information that is absent from a constant or trace-only bath.

It is nevertheless not a complete closure. For the exact-tensor model, all
four held folds exceed both residual-shape limits:

```text
maximum absolute whitened correlation        = 0.0718 to 0.0917  (> 0.05)
maximum absolute squared-whitened correlation = 0.0700 to 0.1609  (> 0.05)
maximum component excess kurtosis             = 0.3725 to 0.6719  (> 0.35)
```

The mechanical verdict is therefore

```text
l2p_conditional_diffusion_supported = 0
l2p_conditional_diffusion_informative_but_insufficient = 1
l2p_tensor_orientation_required = 0
l3p_derivation_authorized = 1
```

`l2p_tensor_orientation_required` remains zero because orientation cannot be
promoted to a closure claim when the absolute exact-tensor gate fails, even
though its NLL improvement over `trace_only` is positive in every fold.

## Physical interpretation

The deterministic `Q_c` result closes one hypothesis and leaves another open.
The conditional covariance of the `L2p` innovation is genuinely inherited
from the instantaneous many-body force/cage geometry; it is not reproduced by
a constant covariance or a time permutation. But a Gaussian diffusion tensor
alone does not remove temporal variance memory or non-Gaussian residual shape.
The missing information must enter through a higher generator level,
non-Gaussian driving, or explicit history/environment coordinates.

The next preregistered calculation is the `L^3p` drift/noise quotient. It must
test whether the remaining squared-memory and excess-kurtosis failures are
reduced without fitting held residuals. Authorization to derive `L^3p` is not
evidence that `L^3p` will close the dynamics.

## Claim boundary

The run does not identify a unique scalar environment variable, prove an
autonomous single-particle GLE, derive Kramers escape, complete the event clock,
or derive thermodynamics. The following remain fixed:

```text
microscopic_environment_coordinate_z_allowed = 0
continuous_gaussian_langevin_bath_allowed = 0
autonomous_single_particle_gle_allowed = 0
complete_event_clock_closure_allowed = 0
kramers_escape_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

Machine-readable evidence is in
`data/renewal_cage_ka_l2p_deterministic_diffusion_T058_{details,convergence,summary}.csv`;
the unclipped held-fold diagnostic is
`figures/renewal_cage_ka_l2p_deterministic_diffusion_T058.svg`.
