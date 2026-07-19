# Microscopic L3p Generator-Quotient Design

## Revision note after numerical canaries

The original frozen design was committed as `0addff1`. The four one-frame
remote canaries were evaluated before any held quotient score was read. Every
absolute prefix, position-step, cage-step, and acceleration-directional gate
passed by a wide margin, but clones 1 and 3 failed the strict cage-step
monotonicity comparison because primary/reference and coarse/reference median
errors were both approximately `4e-10` and reversed ordering within that
numerical floor. The classifier therefore treats primary and coarse medians as
monotonicity-equivalent when both are at most `sqrt(machine epsilon)`.

This numerical-only clarification does not alter trajectory inputs, targets,
frame counts, step ladders, trace prefixes, seeds, Mori/VAR orders, absolute
median or 95th-percentile error limits, or any held physical gate. The original
canary caches and verdicts are retained as prerevision audit artifacts.

## Status and objective

The deterministic conditional diffusion of `c=L^2p` carries significant
time-aligned microscopic information but fails every held-clone absolute
Gaussian closure gate. This design freezes the next authorized test: whether
the next exact generator coordinate

```text
d = Lc = L^3p
```

removes the remaining `c` squared-memory and excess-kurtosis defects. The
calculation is a generator-chain diagnostic, not a fitted latent-state model.
It uses the same four `T=0.58` KA clones, 64 targets, 200 frames, trajectory
hashes, smooth cage map, Mori order 40, VAR order 16, and exact `Q_c` caches as
the deterministic `L^2p` gate.

This stage does not construct the conditional diffusion of `d`. It first asks
whether `d` contains held information beyond the resolved history. Deriving
`Q_d` before that test would pay for fourth cage derivatives and second force
derivatives without evidence that the coordinate is useful.

## Microscopic parent and exact identity

The full unit-mass many-particle dynamics are

```text
dR = V dt,
dV = [F(R)-gamma V] dt + sqrt(2 gamma T) dW,

L = V.grad_R + [F-gamma V].grad_V + gamma T Delta_V.
```

For the fixed smooth cage-relative coordinate `u(R)`, define

```text
p = Lu,
b = Lp,
c = L^2p,
d = Lc = L^3p.
```

Writing `J=grad_R u`, `H=grad_R^2 u`, and `T3=grad_R^3 u`, the already
validated second-generator coordinate is

```text
c = c0 + 2 gamma T Delta_R u,

c0 = J DF[V] + T3[V,V,V] + 3 H[V,F]
     - 3 gamma H[V,V] - gamma JF + gamma^2 JV.
```

The deterministic velocity Jacobian from the preceding gate is

```text
A_c = D_V c = D_V c0.
```

Because `c0` is cubic in velocity,

```text
Delta_V c0 = 6 D_R(Delta_R u)[V] - 6 gamma Delta_R u.
```

Applying the many-particle generator and collecting the temperature terms
therefore gives the exact quotient identity

```text
L^3p = D_R c0[V]
     + A_c [F-gamma V]
     + 8 gamma T D_R(Delta_R u)[V]
     - 6 gamma^2 T Delta_R u.
```

No event labels, displacement observables, waiting times, or macroscopic fits
enter this expression. The identity also provides a component-level audit:
position transport, acceleration response, thermal-gradient response, and
thermal-friction response must sum to the cached `L^3p` value.

For any smooth vector observable `f`, the implementation also tests the
independent generator recursion

```text
D_V(Lf)[eta] = L(D_V f[eta]) + D_R f[eta] - gamma D_V f[eta].
```

This recursion is a numerical identity check only. It is not used to fit the
held result.

## Numerical construction

### Zero-temperature position transport

`D_R c0[V]` is evaluated by a centered configuration derivative. At
`R +/- h_R V`, the KA conservative force and the microscopic force tangent
`DF(R +/- h_R V)[V]` are recomputed from the selected pair potential. The
existing second-generator evaluator is then called with `temperature=0` and
no trace probes. Reusing the unperturbed force or force tangent is forbidden.

### Acceleration response

The already validated deterministic matrix `A_c` is applied to the full
microscopic acceleration

```text
a = F(R)-gamma V.
```

This term contains no random probes. Its matrix-vector product must agree with
the independent velocity-directional `L^2p` evaluator on synthetic and fixed
real frames.

### Cage Laplacian and thermal response

`Delta_R u` is estimated from centered Hessian actions of the analytic cage
Jacobian using one fixed nested Rademacher sequence. The same probes are reused
at `R+h_R V` and `R-h_R V`, so the directional derivative
`D_R(Delta_R u)[V]` is paired rather than differencing independent trace
noise. Prefixes `{4,8,16,32}` are retained separately.

The primary numerical controls are frozen as

```text
position step ladder       = {3e-6, 1e-5, 3e-5}
cage-Hessian step ladder   = {3e-6, 1e-5, 3e-5}
primary step               = 1e-5
inherited L2p directional step = 1e-5
inherited L2p phase-space step = 3e-6
trace prefixes             = {4,8,16,32}
trace seed                 = 20260731
real-frame direction seed  = 20260801
```

These controls may be revised only from synthetic identity failures before a
held score is read. A revision must be recorded as a new design revision and
must not alter any physical tolerance.

The two inherited `L2p` steps reproduce the committed second-generator cache
protocol and are not part of the outer `L3p` sensitivity ladder.

## Numerical gates

The held quotient is not evaluated unless all four clone canaries satisfy:

1. Every component and total `L^3p` value is finite.
2. The median relative norm difference between trace prefixes 16 and 32 is at
   most `0.10`; the 95th percentile is at most `0.25`.
3. Changing either primary step from `1e-5` to `3e-6` changes the total
   `L^3p` with median relative error at most `0.02` and 95th-percentile error
   at most `0.10`.
4. The median primary/reference error must not increase when moving from the
   coarse step `3e-5` to the primary step `1e-5`.
5. `A_c a` agrees with the independent velocity-directional derivative with
   median relative error at most `0.02` and 95th-percentile error at most
   `0.10`.
6. On small systems, the quotient identity agrees with a direct exhaustive
   phase-space generator evaluation to `rtol=2e-4`, `atol=2e-6`; linear
   harmonic cases agree to `rtol=2e-10`, `atol=2e-10`.

Failure is recorded as `l3p_numerical_gate_pass=0` and stops the physical
analysis. Trace counts or steps are not increased after inspecting held
closure scores.

## Frozen state quotient

All model families use identical frame support. The first frame is discarded
so the history null can use a backward difference without future leakage.
Training-only means and scales for the common `(u,p,Lp,L2p)` blocks are shared
across all models. The fifth block, when present, is standardized from the
training clones only.

The paired models are:

1. `l2p_exact_q_baseline`: resolved state `(u,p,Lp,L2p)` with the committed
   exact-`Q_c` covariance model.
2. `l3p_generator`: resolved state `(u,p,Lp,L2p,L3p)` with exact `Q_c` applied
   to the resulting `L2p` innovation.
3. `l3p_time_permuted`: the same fifth block after independently permuting
   frame order for each clone and target with seed `20260802`; each target's
   three Cartesian components share one permutation.
4. `l2p_backward_difference`: the fifth block is
   `(L2p_n-L2p_{n-1})/frame_time`, using only recorded history.

Every model refits its Mori-40 and VAR-16 operators on the three training
clones. Exact-`Q_c` scale and isotropic floor are likewise fitted only on the
training `L2p` innovation. The held clone supplies scores only. The time null
tests time alignment; the backward-difference null tests whether a generic
history derivative carries the same information as the microscopic
generator coordinate.

The analysis treats target paths as replicated single-particle observables.
It does not preserve or test spatial facilitation, and no spatial claim can be
made from this quotient.

## Held metrics and decisions

For the `L2p` innovation in every held clone, report the same exact-`Q_c`
diagnostics as the preceding gate: Gaussian NLL, Mahalanobis ratio, covariance
error, ordinary and squared whitened correlations through lag 40, component
excess kurtosis, Gaussian energy distance, fitted tensor scale, fitted floor,
trajectory hash, and numerical provenance.

`l3p_generator_coordinate_informative=1` only if all conditions hold:

- real `L3p` improves held NLL over the baseline in every fold;
- the replicate-first 95% Student-t interval for that improvement is strictly
  above zero;
- real `L3p` reduces maximum squared-whitened correlation by at least 25%
  relative to the baseline in every fold;
- real `L3p` reduces absolute excess kurtosis by at least 25% relative to the
  baseline in every fold;
- real `L3p` improves held NLL over both the time-permuted and backward-
  difference nulls in every fold;
- the numerical gate passes in every clone.

`l2p_residual_closed_by_l3p=1` additionally requires every real-`L3p` fold to
satisfy the unchanged absolute limits:

```text
maximum ordinary whitened correlation <= 0.05
maximum squared whitened correlation  <= 0.05
maximum absolute component kurtosis   <= 0.35
maximum whitened covariance error     <= 0.10
Mahalanobis ratio                      in [0.8,1.2]
isotropic floor variance fraction      <= 0.25
```

The fifth coordinate's own innovation is reported descriptively, but this
stage cannot set `finite_l3p_gaussian_closure_supported=1` because `Q_d` has
not been derived. If the real coordinate is informative, set
`l3p_diffusion_derivation_authorized=1`; otherwise keep it zero and leave the
next mechanism unresolved.

## Claim boundary

The output always records

```text
l3p_numerical_gate_pass = 0/1
l3p_generator_coordinate_informative = 0/1
l2p_residual_closed_by_l3p = 0/1
l3p_diffusion_derivation_authorized = 0/1
finite_l3p_gaussian_closure_supported = 0
microscopic_environment_coordinate_z_allowed = 0
continuous_gaussian_langevin_bath_allowed = 0
autonomous_single_particle_gle_allowed = 0
complete_event_clock_closure_allowed = 0
kramers_escape_claim_allowed = 0
spatial_facilitation_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

Even a positive result establishes only that the next directly generated
microscopic coordinate carries omitted single-particle information. It does
not prove a finite generator hierarchy, an autonomous environment variable,
Kramers escape, event-clock closure, spatial facilitation, or thermodynamics.

## Execution and provenance

- All production canaries and four-clone calculations run sequentially on the
  approved remote compute node; no production simulation runs locally.
- Only one remote numerical process may run at a time.
- Cache checkpoints include trajectory SHA256, target indices, potential
  protocol, every step, probe prefixes and seeds, frame count, component
  arrays, and all claim flags.
- A cache with mismatched provenance is rejected rather than overwritten.
- Held CSV/SVG/PDF artifacts are committed only after the numerical gate and
  mechanical classifier have run without changing frozen tolerances.

## Literature relation

The use of an iterated generator coordinate follows the recursive Mori-chain
view of exact generalized Langevin equations, not a claim that a finite chain
must close. See [Mori's recurrence formulation](https://doi.org/10.1103/PhysRevE.62.1769)
and the [recursive continued-fraction representation](https://doi.org/10.1143/PTP.61.850).
Generator-based coarse graining can also infer drift and diffusion from data
([gEDMD](https://arxiv.org/abs/1909.10638)); here the coordinate is instead
computed from the known many-particle KA generator and tested against held
clones. The covariance identity is the diffusion-generator carré-du-champ
construction. None of these formalisms guarantees that the span of
`(u,p,Lp,L2p,L3p)` is invariant.
