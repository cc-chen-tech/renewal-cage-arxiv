# Second-Generator Krylov Response Design

## Objective

Test whether one additional observable fixed by the many-particle
Kob-Andersen (KA) Langevin generator extends the predictive horizon of the
tagged-particle response.  The resolved state is enlarged from

```text
Z1 = (delta x, delta v, delta F, delta L F)
```

to

```text
Z2 = (delta x, delta v, delta F, delta L F, delta L2 F).
```

The experiment is a response-level prerequisite for a stochastic
single-particle Langevin closure.  It does not fit or claim diffusion, NGP,
scattering, a renewal clock, a Kramers rate, or thermodynamics.

## Why This Is The Next Microscopic Test

The current 12-coordinate generator-constrained model fixes the first three
rows from the microscopic equations and projects `L2 F` onto
`(x,v,F,L F)`.  It fails held-out response despite using the exact pair force
and first force-generator mode.  Separately, longer linear force histories,
low-wavevector velocity fields, and force-conditioned innovations fail the
four-tau transport test.  Adding an arbitrary latent clock would therefore
skip the unresolved generator layer.

The second-generator state is already present in the matched full-particle
response records.  Testing it requires no new phenomenological coordinate and
no macroscopic fit.  It is the smallest remaining extension on the
Mori-Krylov ladder.

## Microscopic Starting Point

For unit particle mass, the C3-switched KA Langevin dynamics are

```text
dr_i = v_i dt
dv_i = [F_i(R) - gamma v_i] dt + sqrt(2 gamma T) dW_i,
F_i = -grad_i U_KA,C3(R).
```

The backward generator is

```text
L = sum_j v_j . grad_(r_j)
    + sum_j [F_j - gamma v_j] . grad_(v_j)
    + gamma T sum_j Delta_(v_j).
```

For the tagged force,

```text
G_i = L F_i = -sum_j H_ij v_j,
H2_i = L2 F_i.
```

The existing implementation evaluates `G_i` analytically and `H2_i` from
the exact acceleration term plus a centered directional derivative of the KA
Hessian along the measured many-particle velocity.  Its finite-step force
identity and Hessian-controlled multiplicative noise have independent tests.

For a matched common-noise displacement response, the exact tangent hierarchy
starts as

```text
d(delta x) = delta v dt
d(delta v) = [delta F - gamma delta v] dt
d(delta F) = delta G dt
d(delta G) = delta H2 dt + dM_G.
```

The tangent martingale `dM_G` is not discarded.  Its conditional covariance is
fixed by the response of the microscopic Hessian-noise geometry and has
already passed the C3 full-horizon covariance gate.  The present experiment
tests the deterministic projected drift needed before an autonomous
stochastic propagation can be attempted.

## Model Under Test

The second-generator closure retains `delta H2` as an explicit state and
projects only its next generator action:

```text
d(delta H2)/dt = B_x delta x + B_v delta v + B_F delta F
                 + B_G delta G + B_H delta H2 + R3.
```

Each `B_*` is a full `3 x 3` block.  Scalar isotropy is not imposed.  Absolute
position is used only as a tangent displacement from the perturbed initial
state, so translation invariance is preserved.

The last row is estimated in integrated weak form,

```text
delta H2_(n+1) - delta H2_n
  = dt B [Z2_(n+1) + Z2_n] / 2 + rho_n.
```

This avoids differentiating the already expensive `L2 F` observable.  The
continuous generator uses the exact first four rows above and the fitted last
row.  A fourth-order Runge-Kutta map supplies the saved-step transition.

## Controls

Three models are evaluated on identical splits:

1. `first_generator_constrained`: the existing 12-state `(x,v,F,L F)` model;
2. `second_generator_constrained`: the proposed 15-state model with only the
   `L3 F` projection row estimated;
3. `free_second_generator_transition`: an unconstrained 15-state linear map,
   used only as an overfitting/control bound.

The second-generator model is useful only if it improves held-out position
response over model 1 while remaining stable and retaining a substantially
smaller fitted surface than model 3.

## C3 Response Protocol

Generate a fresh smooth-force response panel from the existing verified C3
parent and pinned Lepton-enabled LAMMPS executable:

- state point: `T=0.58`, `gamma=1`;
- target particle: A particle id `821`;
- integration step: `0.001 tau`;
- duration: `1.0 tau`;
- saved interval: `0.005 tau`;
- displacement amplitudes: `epsilon=0.001,0.002`;
- eight independent velocity/Langevin seed pairs;
- matched `+epsilon/-epsilon` paths use identical random seeds;
- C3 switch from `2.0 sigma` to `2.5 sigma`;
- raw trajectories are reduced sequentially and deleted after verified NPZ
  extraction.

The C3 switch is mandatory.  The hard-cutoff response panel is retained only
as historical evidence because pair-support crossings make high-order tangent
derivatives non-smooth.

## Fitting And Held-Out Evaluation

All response coordinates are central differences of matched paths.  For every
leave-one-member-out fold:

- fit only `epsilon=0.001` paths from the other seven members;
- use fit horizons `0.05`, `0.10`, and `0.20 tau`;
- propagate autonomously from the held member's initial tangent state;
- evaluate both `epsilon=0.001` and `0.002`;
- report horizons `0.20`, `0.50`, and `1.00 tau`;
- fit no macroscopic observable.

Cross-epsilon mismatch determines whether a member/horizon remains in the
linear-response regime.  A horizon with position-response mismatch above
`0.02` is marked unidentified rather than scored as a model failure.

## Preregistered Gates

### Integrity

- every NPZ hash matches the manifest;
- all 32 paths have the same 201-frame grid;
- common-noise seeds match within each plus/minus pair;
- the manifest records `potential_protocol=ka_lj_c3_switch`;
- `fit_parameters_from_macro_observables=false` and
  `thermodynamic_claim_allowed=false` everywhere.

### Numerical identification

- the weak-form design has full 15-coordinate rank;
- the transition spectral radius is at most `1 + 1e-6`;
- the fitted last-row residual is finite;
- the fit interval leaves a temporal holdout.

### Response improvement

At each identified horizon, report paired errors for models 1 and 2.  The
second-generator state passes a horizon only when:

- mean held position-response relative L2 error improves by at least 20
  percent over model 1;
- every identified fold has error at most `0.20` at `0.20 tau`, `0.35` at
  `0.50 tau`, or `0.50` at `1.00 tau`;
- the maximum absolute held residual-state correlation is at most `0.20`.

A one-tau claim additionally requires at least six identified folds at both
epsilon values.  Otherwise the one-tau result is underidentified regardless
of the mean error.

### Claim gate

`second_generator_response_allowed` is true only if all integrity and
numerical gates pass and the `0.20 tau` response gate passes.  A stronger
`one_tau_generator_response_allowed` additionally requires the one-tau gate.
The following remain false in this phase:

```text
autonomous_stochastic_single_particle_gle_allowed
event_clock_claim_allowed
kramers_escape_claim_allowed
thermodynamic_claim_allowed
```

## Interpretation Ladder

1. If model 2 passes and model 1 fails, `L2 F` is a resolved microscopic bath
   mode and should be carried into the stochastic long-trajectory model.
2. If model 2 improves but does not pass, extend with orthogonalized slow force
   modes rather than blindly differentiating to higher generator order.
3. If model 2 does not improve, a short linear Krylov chain is not the missing
   state.  The next test is a time-lagged collective-force basis or a
   position-dependent nonlinear GLE.
4. Only after a stochastic model passes held-out `D`, NGP, multi-k `F_s`, and
   event-clock tests may it be called a microscopic closure of the effective
   theory.

## Deliverables

- weak-form second-generator fit and propagation routines;
- focused unit tests with exact synthetic 15-state systems;
- C3 one-tau response manifest and reduced NPZ paths under `tmp/`;
- tracked summary and response-curve CSV files;
- a claim-limited scientific report with formulas, gates, and the next branch;
- package tests that prevent promotion of response closure to thermodynamic or
  complete event-clock claims.
