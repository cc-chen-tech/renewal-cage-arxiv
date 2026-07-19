# Nonlinear Bath Elimination Design

## Scope

This experiment tests one precise microscopic route from an extended
many-coordinate Langevin equation to a single-particle generalized Langevin
equation (GLE). It does not fit KA data yet. The first stage must establish the
exact elimination identity, fluctuation-dissipation structure, numerical
reconstruction, and cage first-passage behavior on a frozen synthetic system.
Only a passing synthetic stage authorizes a separate real-KA kernel-estimation
specification.

The construction is motivated by rigorous position-dependent and nonlinear
Mori-Zwanzig reductions. It addresses the current evidence that instantaneous
local scalar descriptors and linear Gaussian auxiliary baths do not close the
observed cage dynamics.

## Extended microscopic dynamics

Let `u,p` be one tagged cage-relative coordinate and velocity. Let `z_a` be
finite auxiliary coordinates representing projected many-particle bath modes.
For matrices `C_a(u)` and rates `alpha_a > 0`, define

```text
du = p dt,

dp = [-grad W(u) - gamma_0 p + sum_a C_a(u) z_a] dt
     + sqrt(2 gamma_0 T) dW_0,

dz_a = [-alpha_a z_a - C_a(u)^T p] dt
       + sqrt(2 alpha_a T) dW_a.
```

The `p-z_a` coupling is antisymmetric. For any smooth position-dependent
`C_a`, it preserves the quadratic bath energy because

```text
p^T C_a(u) z_a + z_a^T[-C_a(u)^T p] = 0.
```

Together with the matched diagonal thermostats, the invariant density is

```text
rho(u,p,z) proportional to
exp{-[W(u) + |p|^2/2 + sum_a |z_a|^2/2]/T}.
```

No renewal event, waiting-time law, escape rate, or phenomenological hazard is
present in this parent dynamics.

This invariance is an exact Fokker-Planck statement, not a numerical
equilibration assumption. The conservative vector field has zero phase-space
divergence and annihilates the total energy:

```text
p grad W + p(-grad W) = 0,
p sum_a C_a z_a + sum_a z_a(-C_a p) = 0.
```

For the momentum thermostat, division of its adjoint Fokker-Planck action by
the Gibbs density gives

```text
gamma_0(1-p^2/T) + gamma_0(p^2/T-1) = 0,
```

and each auxiliary thermostat gives the identical cancellation with
`gamma_0,p` replaced by `alpha_a,z_a`. Dependence of `C_a` on `u` introduces no
extra divergence because the coupled drifts are differentiated with respect to
`p` and `z_a`, not with respect to `u`.

## Exact elimination

Each auxiliary coordinate has the exact solution

```text
z_a(t) = exp(-alpha_a t) z_a(0)
       - integral_0^t exp[-alpha_a(t-s)] C_a(u_s)^T p_s ds
       + sqrt(2 alpha_a T)
         integral_0^t exp[-alpha_a(t-s)] dW_a(s).
```

Substitution into the `p` equation gives the single-particle GLE

```text
dp = -grad W(u_t) dt - gamma_0 p_t dt
     - integral_0^t K(t,s;u) p_s ds dt
     + xi_t dt + sqrt(2 gamma_0 T) dW_0,

K(t,s;u) = sum_a
  C_a(u_t) exp[-alpha_a(t-s)] C_a(u_s)^T,
  0 <= s <= t.
```

Introduce stationary OU bath forces `y_a` with

```text
E[y_a(t)y_a(s)^T] = T exp[-alpha_a |t-s|] I,
xi_t = sum_a C_a(u_t)y_a(t).
```

For a fixed externally supplied coordinate path, independent bath replay then
has covariance

```text
E_bath[xi_t xi_s^T | fixed u path]
  = T sum_a C_a(u_t) exp[-alpha_a |t-s|] C_a(u_s)^T.
```

This is a bath-level FDT construction, not a claim that naive conditioning on
an endogenous realized `u` path factorizes the same way. In the coupled
dynamics, `u` depends on the bath noise, so such pointwise conditioning would
require a separate projection argument and generally does not follow from a
finite basis.

For constant `C_a`, this reduces to the standard sum-of-exponentials GLE. For
position-dependent `C_a`, it is a multiplicative, path-dependent memory law,
not a scalar fluctuating-diffusivity ansatz.

## Frozen synthetic system

The first numerical test is one-dimensional and periodic:

```text
W(u) = V_0 [1 - cos(2 pi u / ell)],
C_a(u) = c_a [1 + epsilon_a cos(2 pi u / ell + phi_a)].
```

The frozen dimensionless parameters are

```text
T          = 0.58
gamma_0    = 1
ell        = 1
V_0        = 1.74
alpha      = [0.20, 1.00]
c          = [1.00, 0.55]
epsilon    = [0.45, 0.25]
phi        = [0, pi/2]
dt         = 0.001
burn-in    = 100 tau
production = 400 tau per trajectory
trajectories = 256
seed       = 20260811
```

Production simulation runs sequentially on the approved remote compute node.
No production trajectory is generated locally. The full parameter vector,
source hashes, Python/NumPy versions, seed, checkpoint count, and output hashes
are recorded before any event-clock score is read.

The frozen explicit scheme uses

```text
u_(n+1) = u_n + p_n dt,

p_(n+1) = p_n
  + [-W'(u_n) - gamma_0 p_n + sum_a C_a(u_n) z_(a,n)] dt
  + sqrt(2 gamma_0 T dt) G_(0,n),

z_(a,n+1) = rho_a z_(a,n)
  - phi_a C_a(u_n) p_n
  + sqrt[T(1-rho_a^2)] G_(a,n),

rho_a = exp(-alpha_a dt),
phi_a = [1-exp(-alpha_a dt)]/alpha_a.
```

All `G` values are independent standard normals. The exact-OU auxiliary update
avoids replacing the measured bath timescale by an Euler decay. A 16-trajectory,
`2 tau` canary retains every Brownian increment for pathwise reconstruction;
the full production cache stores seeds, checkpoints, downsampled coordinates,
and event records rather than duplicating all noise arrays.

The event coordinate is retained every `0.01 tau`; equilibrium `(u,p,z)`
samples are retained every `0.10 tau`. Canary initial conditions are sampled
from the known Gibbs density so the primary/half-step discretization comparison
does not depend on a two-tau burn-in. Production and null runs retain the
frozen `100 tau` burn-in.

## Numerical and physical gates

### 1. Algebraic reconstruction

For the canary trajectories, reconstruct `z_a(t)` from the exact discrete
variation-of-constants recurrence using the retained noise increments. The
maximum relative pathwise error must be at most `5e-11`; halving `dt` must not
increase the equilibrium-measure errors defined below.

### 2. Equilibrium measure

After burn-in, require

```text
|E[p^2]/T - 1| <= 0.05,
max_a |E[z_a^2]/T - 1| <= 0.05,
max_a |corr(p,z_a)| <= 0.05.
```

The periodic position histogram must agree with `exp[-W(u)/T]` with total
variation distance at most `0.08` on 40 fixed bins.

### 3. Fixed-path bath-level FDT replay

Use 12 fixed position bins and lags

```text
[0.01, 0.05, 0.10, 0.25, 0.50, 1.00] tau.
```

Freeze 32 stored coordinate paths, generate 512 fresh independent stationary
OU bath histories for each path, and compare the replay covariance with the
exact kernel. The pooled normalized RMSE must be at most `0.15`, and every lag
must have normalized RMSE at most `0.30`. Endogenous trajectory samples are
not reused as if they were conditionally independent bath draws.

The replay uses trajectories `0:32`, their first `20 tau` after production
burn-in, seed `20260813`, and the stored `0.01 tau` coordinate spacing. The
time-permuted coupling ablation uses seed `20260814`; it changes only the
coordinate order supplied to `C_a(u_t)` in the replay target, not the simulated
parent path or bath noise.

### 4. Cage first passage

A cage index is the nearest periodic minimum. An accepted cage transition
must cross the intervening maximum and remain in the new cage for `0.10 tau`.
The definition and persistence window are fixed for every trajectory.

Fit on the first 128 trajectories and evaluate on the remaining 128:

```text
constant hazard: h(t) = lambda,
delayed hazard:  h(t) = lambda [1 - exp(-t/tau_d)]^2.
```

The delayed fit profiles the analytic maximum-likelihood `lambda` over the
fixed grid `tau_d = geomspace(1e-3, 1e2, 801) tau`; it does not optimize or
extend this grid after seeing held trajectories.

Held survival is evaluated on the fixed grid `linspace(0, 100, 201) tau`.
Integrated survival error is the trapezoidal integral of the absolute
empirical Kaplan-Meier minus model-survival difference, divided by `100 tau`.
The delay confidence interval uses 400 complete-trajectory bootstrap samples
from the training half with seed `20260812`. A positive-delay result requires
the 95% lower bound to be strictly above the frozen grid minimum `1e-3 tau`;
mere positivity induced by the constrained grid is not sufficient.

The synthetic dynamics supports the delayed effective clock only if the held
delayed model has lower negative log likelihood, at least 10% lower integrated
survival error, and the trajectory-bootstrap 95% lower bound on `tau_d` is
strictly above the frozen grid minimum. Failure is retained as a negative
result; parameters are not tuned after reading this gate.

### 5. Nulls and ablations

Run with the same Brownian streams:

1. `epsilon_a = 0`, retaining the same bath times and mean coupling;
2. `c_a = 0`, retaining only the periodic underdamped Langevin particle;
3. time-permuted `C_a(u_t)` in the eliminated-kernel reconstruction only.

These distinguish ordinary Kramers hopping, linear exponential memory, and
state-aligned nonlinear memory. They do not change the primary gate.

## Outcome boundary

Machine-readable outputs always include

```text
exact_nonlinear_bath_elimination_supported = 0/1
synthetic_bath_level_fdt_replay_supported = 0/1
synthetic_delayed_hazard_emerges = 0/1
real_ka_position_dependent_kernel_authorized = 0/1
autonomous_single_particle_gle_allowed = 0
complete_event_clock_closure_allowed = 0
kramers_escape_claim_allowed = 0
spatial_facilitation_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

Authorization of a real-KA follow-up requires the algebraic, equilibrium, and
bath-replay FDT gates. The delayed-hazard gate is reported separately: it
tests whether this microscopic mechanism naturally produces the current
effective hazard, not whether the elimination identity is valid.

Even a complete synthetic pass does not establish that KA dynamics uses this
mechanism. A later real-data specification must estimate `W`, the kernel
spectrum, and the minimum auxiliary rank on training clones and must predict
held-clone first passage, MSD, NGP, and multi-`k F_s` without retuning.

## Literature relation

- Vroylandt and Monmarche rigorously derive nonlinear-force GLEs with
  position-dependent linear memory and corresponding fluctuation-dissipation
  relations, and provide Volterra equations for estimation from atomistic
  trajectories.
- Ayaz, Scalfi, Dalton, and Netz derive a hybrid-projection GLE with the PMF,
  linear velocity memory, nonlinear coordinate memory, and generally
  non-Gaussian random force for interacting many-body systems.
- Likelihood-based auxiliary-variable GLEs can reproduce first-passage
  distributions, but this experiment fixes the thermodynamically consistent
  extended SDE before fitting any first-passage data.
