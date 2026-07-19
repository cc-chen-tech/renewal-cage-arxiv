# Real-KA Position-Dependent Kernel Identifiability Design

## Pre-run revision note

The original committed design incorrectly required latent auxiliary-innovation
whiteness and a second-FDT target for both `M3` and `M4`. Before any production
run, implementation showed that a signed `M3` kernel has no positive covariance
target and that multiple latent `z_a` states cannot be uniquely reconstructed
from one observed three-vector force. The corrected gate treats `M3` as a
deterministic real-pole realization and tests second FDT only for the positive
`M4` kernel using directly observable residual-force covariance. Latent-state
innovation and full stochastic-bath claims remain closed. This revision does
not change trajectories, targets, folds, basis, supports, ridges, rank/pole
grids, seeds, or numerical tolerances.

## Objective

Determine which part of a single-particle generalized Langevin equation is
actually identifiable from the existing four microscopic KA Langevin clones.
The experiment must separate an exact finite-basis Mori-Zwanzig (MZ)
representation from the more restrictive thermodynamically embedded auxiliary
bath introduced in the nonlinear-bath elimination experiment.

This is a kernel-identifiability experiment. It does not fit cage events, MSD,
NGP, or scattering functions, and it cannot authorize a complete event clock.

## Microscopic starting point

For the complete many-particle underdamped Langevin state `X_t=(q_t,p_t)`, let

```text
u(X) = x_target - C_smooth(X)
v(X) = L u(X)
a(X) = L v(X).
```

The existing decomposed-drift caches provide frame-aligned values of `u`, `v`,
and the exact projected microscopic drift `a` for the same 64 A particles in
four isoconfigurational `T=0.58` clones. No finite-difference acceleration is
used.

For a finite radial vector basis

```text
h_0(u) = u,
h_1(u) = u s(u),
h_2(u) = u [s(u)^2 - 1],
s(u)   = (|u|^2 - mu_r2) / sigma_r2,
```

`mu_r2` and `sigma_r2` are fitted on training clones only. Every basis vector
is dimensionless after applying the training-only position and acceleration
scales used by all candidate models in that held fold.

## Exact finite-basis MZ baseline

Vroylandt and Monmarche show that a position observable and its velocity admit
the finite-basis projected equation

```text
du_t = v_t dt,
dv_t = f_b(u_t) dt
       - integral_0^t K_MZ(s,u_(t-s)) v_(t-s) ds dt
       + xi_t dt + resolved white-noise increment,
```

where

```text
f_b(u) = sum_j f_j h_j(u),
K_MZ(s,u)v = sum_j k_j(s) grad h_j(u) v.
```

The coefficients follow from the projected correlation identity

```text
<E_l(0), dO(t)/dt>
 = sum_j f_j <E_l(0), E_j(t)>
 + integral_0^t sum_j g_j(s)
       <E_l(0), E_j(t-s)> ds.
```

This first-kind Volterra system is ill-conditioned. The implementation must
therefore solve one joint, ridge-regularized lower-triangular system rather than
use unregularized pointwise recursion. Ridge strength and memory support are
selected without reading the held clone.

This MZ form depends on the historical position `u_(t-s)`. It is the
nonparametric physical baseline, not automatically a finite auxiliary SDE.

## Thermodynamic two-position realization

The previously derived reversible auxiliary bath is

```text
du = v dt,
dv = [-grad W(u) - gamma_0 v + sum_a C_a(u) z_a] dt
     + sqrt(2 gamma_0 T) dW_0,
dz_a = [-alpha_a z_a - C_a(u)^T v] dt
       + sqrt(2 alpha_a T) dW_a.
```

Exact elimination gives

```text
K_aux(t,s;u) = sum_a C_a(u_t)
                     exp[-alpha_a(t-s)]
                     C_a(u_s)^T.
```

This kernel depends on both `u_t` and `u_s`. It is not the same mathematical
family as `K_MZ(t-s,u_s)`. The real-KA gate must therefore include both of the
following constrained realizations:

```text
past_position_real_pole:
  K(s,u)v = sum_(a,j) w_(a,j) exp(-alpha_a s) grad h_j(u) v

two_position_prony:
  K(t,s;u)v = sum_a C_a(u_t) exp[-alpha_a(t-s)] C_a(u_s)^T v.
```

All `alpha_a` are positive. The signed coefficients `w_(a,j)` mean that the
past-position model tests a finite real-pole temporal family, not a positive
thermodynamic kernel. Only the two-position model has the positive outer-product
factorization required by the auxiliary bath. It uses an isotropic radial
coupling basis

```text
C_a(u) = c_(a,0) I + c_(a,1) s(u) I
         + c_(a,2) u u^T / max(|u|^2, epsilon_r2).
```

`epsilon_r2` is the first training percentile of `|u|^2`, frozen independently
inside every held fold. No event or macro observable enters this choice.

## Frozen data split and model selection

Use exactly the existing four `T=0.58` clones, their exact trajectory hashes,
the 64 fixed A-particle indices, and frame spacing `0.01 tau`.

For each outer held clone:

1. Fit the radial basis scales, projected mean force, kernel, and every
   regularization parameter on the other three clones.
2. Select memory support from `[4, 16, 40, 100]` frames by inner whole-clone
   validation among those three training clones.
3. Select ridge from `[0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]` after column
   normalization of the training design. Zero ridge is admissible only for a
   full-column-rank design with finite condition number.
   For support `S`, the projected-correlation system uses the `S` outer lags
   `S-1..2S-2` and one common time-origin window ending before lag `2S-2`.
   This is the finite-basis correlation identity, not a subsample of raw rows.
4. Select auxiliary rank from `[1, 2, 4, 8]` and positive decay rates from the
   fixed grid `logspace(log10(0.05), log10(50), 32)` by the same inner folds.
   Within each training fold, select poles greedily from this dictionary by the
   reduction in least-squares reconstruction error of that fold's fitted `M2`
   temporal coefficient matrix. Refit all selected exponential amplitudes after
   every added pole. The outer held clone never enters this OMP step.
5. Refit the selected model on all three training clones and score the held
   clone once.

Time rows from one clone may never be split across training and validation.
The outer held clone supplies scores only. Replicate cancellation cannot turn a
failed held clone into a pass.

## Candidate hierarchy

The frozen comparison is:

```text
M0 instantaneous_mean_force
M1 stationary_scalar_nonparametric_volterra
M2 finite_basis_MZ_position_kernel
M3 past_position_real_pole
M4 two_position_positive_prony
M5 time_permuted_position_null
```

`M2` is the least restrictive position-dependent MZ baseline. `M3` tests
whether it has a finite real-pole temporal realization. `M4` tests the distinct
positive thermodynamic auxiliary-bath factorization. `M5` preserves one-time feature
distributions and destroys ordered position-memory pairing.

Damped oscillatory or general matrix-pole baths are not fit in this experiment.
They can be authorized for a later frozen experiment only if `M2` is
identifiable and both the real-pole `M3` and positive auxiliary-bath `M4`
realizations fail.

## Held diagnostics

For every outer held clone and model, report:

- normalized deterministic drift RMSE and Gaussian quasi-likelihood;
- improvement over `M0` and over `M1`;
- correlation between prediction and exact microscopic drift;
- residual correlations with every resolved basis `h_j(u_t)` and
  `grad h_j(u_t)v_t` over lags `0..40`;
- residual covariance and the corresponding second-FDT target;
- kernel support, regularization, rank, pole locations, and condition number;
- all source, trajectory, drift-cache, and selected-target SHA-256 values.

Orthogonal MZ noise is generally colored. Whiteness is therefore not a gate for
`M1` or `M2`. Innovation whiteness is tested only for the Markovian auxiliary
states in `M3` and `M4`, after their fitted latent states are reconstructed.

## Mechanical decisions

The nonparametric position-dependent kernel is identifiable only if, in every
held clone:

```text
M2 drift RMSE <= 0.90 * M1 drift RMSE,
M2 drift NLL  < M1 drift NLL,
M2 beats M5 in both metrics,
maximum normalized resolved-basis residual correlation <= 0.20,
all fitted arrays and condition diagnostics are finite.
```

The improvement ratios are additionally summarized replicate-first with a
two-sided 95 percent `t` interval whose upper endpoint must remain below `1`.

The real-pole `M3` realization is supported only if it passes all of the
following in every held clone:

```text
drift RMSE <= 1.10 * M2 drift RMSE,
drift NLL  <= M2 drift NLL + 0.05 * held scalar-component count,
maximum normalized resolved-basis residual correlation <= 0.20,
all selected alpha_a > 0.
```

The positive-Prony `M4` realization must pass the same deterministic gates and
additionally requires its directly testable second-FDT residual-force
covariance normalized RMSE to be at most `0.30` in every held clone. Latent
auxiliary-innovation whiteness is descriptive only when an observation model
identifies the latent states; it cannot open a claim in this experiment.

The NLL allowance is therefore `0.05` per held scalar component, not a fixed
unnormalized total-likelihood allowance.

## Claim boundary

Every output always records

```text
real_ka_kernel_identifiability_test_required = 1
real_ka_position_dependent_mz_kernel_identified = 0/1
past_position_real_pole_identified_in_ka = 0/1
two_position_positive_prony_identified_in_ka = 0/1
positive_prony_kernel_identified_in_ka = 0/1
finite_auxiliary_rank_identified_in_ka = 0/1
latent_auxiliary_innovation_identified_in_ka = 0
stochastic_auxiliary_bath_identified_in_ka = 0
oscillatory_matrix_bath_authorized = 0/1
autonomous_single_particle_gle_allowed = 0
complete_event_clock_closure_allowed = 0
kramers_escape_claim_allowed = 0
spatial_facilitation_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

`positive_prony_kernel_identified_in_ka` is exactly the two-position `M4`
decision; `M3` cannot open it. `finite_auxiliary_rank_identified_in_ka=1`
additionally requires `M4` to pass and the same rank to be selected in all four
outer folds. A synthetic nonlinear-bath pass cannot open any real-KA field.

Even a real-KA positive-Prony pass identifies only a force-memory realization
on the isoconfigurational `T=0.58` data. It does not establish autonomous
long-time propagation, cage escape, delayed hazard, low-temperature transfer,
or equilibrium thermodynamics. Those require later held trajectory generation
from the fitted SDE and event-clock/macro gates.

## Literature relation

- Vroylandt and Monmarche derive the finite-basis position-dependent GLE,
  second fluctuation-dissipation relation, and Volterra equations used here.
- Ayaz, Scalfi, Dalton, and Netz show that nonlinear potential-of-mean-force and
  nonlinear memory terms depend on the projection choice; this prevents the
  harmonic Volterra fit from being treated as a universal microscopic GLE.
- Lei, Baker, and Li connect rational memory kernels to extended Markovian
  auxiliary variables. That construction motivates `M3/M4` only after the
  nonparametric `M2` baseline is identified.
- Lang and Lu emphasize regularized Prony estimation and identifiability. This
  motivates whole-clone model selection and separate realization claims rather
  than assuming a two-pole bath.
