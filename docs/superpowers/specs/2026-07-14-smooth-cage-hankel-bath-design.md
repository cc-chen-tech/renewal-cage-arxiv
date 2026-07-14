# Smooth-Cage Hankel Bath Design

## Objective

Test whether the temporal residual left by the exact-force Hankel bath is the
motion of a local cage coordinate that was omitted from the resolved state.
The coordinate must be a differentiable function of the full many-particle
configuration, not a hard neighbor graph or an event label.

## Microscopic Derivation

The unit-mass parent dynamics are

```text
dR = V dt,
dV = [F(R) - gamma V] dt + sqrt(2 gamma T) dW.
```

For the existing Wendland-C4 force-support cage, define

```text
u_i(R) = r_i - C_i(R),
J_i(R) = grad_R u_i(R),
p_i(R,V) = J_i(R) V.
```

Because `R` has finite variation in the underdamped SDE, Ito projection gives

```text
du_i = p_i dt,
dp_i = [J_i F + Hess(u_i):(V tensor V) - gamma p_i] dt
       + sqrt(2 gamma T) J_i dW.
```

Thus `u_i` and `p_i` are exact microscopic vector coordinates. Their
instantaneous noise covariance is `2 gamma T J_i J_i^T`; no empirical cage
timescale or event threshold enters their definition.

## Batch Extraction

Add a vectorized extractor for fixed tagged particles. It must reproduce the
existing scalar analytic Jacobian calculation for relative position,
relative velocity, and `J J^T`, preserve translation covariance, and use the
same fixed KA interaction support.

The four long `T=0.58` clones and the same 64 fixed A particles used by the
Hankel bath are authoritative. Full trajectories provide the neighbor
positions and velocities. Exact-force reduced-cache hashes and target indices
must match before cage arrays are accepted.

## Resolved Models

Fit each model on three clones and evaluate the fourth:

```text
H16    = [v, z_1, ..., z_16],
H16+u  = [v, z_1, ..., z_16, u],
H16+up = [v, z_1, ..., z_16, u, p].
```

The `z_m` are training-only temporal PCA modes of 64 exact-force frames.
Every state component is a rotationally equivariant three-vector. Use the
same covariance-contracted transition and non-Gaussian innovation blocks as
the stationary Hankel test. No macro observable enters the fit.

## Gates

The primary `H16+up` model is promoted only if, over four leave-one-clone-out
folds:

- mean held velocity `R2 >= 0.97`;
- maximum residual-state correlation `<= 0.20`;
- maximum residual lag correlation `<= 0.70` and at most `0.80` of the H16
  baseline;
- every fold improves residual lag correlation;
- the autonomous model has terminal diffusion error `<= 0.20`, multi-k `Fs`
  error `<= 0.20`, NGP error `<= 0.10`, and event-rate error `<= 0.20`.

The observed coordinate must also satisfy the saved-grid trapezoidal
kinematic identity for `du=p dt` to a reported normalized RMS error. This is a
diagnostic, not a fitted gate.

Failure means the smooth local cage coordinate is not the missing memory by
itself. It then becomes the physically constrained input to a training-only
slow-coordinate learner rather than grounds for another empirical hazard.

The following remain false unless every gate passes:

```text
smooth_cage_hankel_bath_allowed
autonomous_single_particle_gle_allowed
complete_event_clock_closure_allowed
kramers_escape_claim_allowed
thermodynamic_claim_allowed
```
