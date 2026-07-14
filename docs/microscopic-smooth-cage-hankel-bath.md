# Microscopic Smooth-Cage Hankel Bath Test

## Question

The exact-force Hankel bath leaves a strongly colored orthogonal residual.
This experiment tests whether the omitted state is the continuous local cage
coordinate already derived from the full many-particle Langevin equation.

## Exact Coordinate Projection

The unit-mass parent process is

```text
dR = V dt,
dV = [F(R) - gamma V] dt + sqrt(2 gamma T) dW.
```

For tagged particle `i`, the Wendland-C4 force-support cage defines

```text
u_i(R) = r_i - C_i(R),
J_i(R) = grad_R u_i(R),
p_i(R,V) = J_i(R) V.
```

Since `R` has finite variation in the underdamped process, Ito projection is

```text
du_i = p_i dt,
dp_i = [J_i F + Hess(u_i):(V tensor V) - gamma p_i] dt
       + sqrt(2 gamma T) J_i dW.
```

Thus `u_i` and `p_i` are exact microscopic vector coordinates, not fitted
precursor variables. A new batched implementation analytically contracts the
same Jacobian blocks used by the scalar projection and returns `u`, `p`, and
`J J^T` for fixed target particles. Unit tests match the scalar Jacobian
projection at `1e-12`, verify rigid-motion covariance, and require positive
noise Gram matrices.

## Protocol

- four independent 10 tau Kob-Andersen Langevin clones at `T=0.58`;
- full 4096-particle positions, velocities, and forces every `0.01 tau`;
- the same 64 fixed A particles and trajectory hashes as the exact-force
  Hankel experiment;
- training-only 64-frame temporal PCA with 16 exact-force modes;
- leave one whole clone out;
- autonomous predictions of diffusion, NGP, multi-k `F_s`, and nonrecrossing
  event rate with no macro-observable fit.

The three resolved states are

```text
H16    = [v, z_1, ..., z_16],
H16+u  = [v, z_1, ..., z_16, u],
H16+up = [v, z_1, ..., z_16, u, p].
```

On the saved grid, the observed trapezoidal defect in `du=p dt` has normalized
RMS `0.03973` (maximum fold value `0.04009`). This verifies that the extracted
position and velocity coordinates are dynamically aligned at the analysis
resolution.

## Held-Clone Results

| Model | velocity R2 | max state corr. | max residual lag | velocity residual lag | force residual lag | D error | max Fs error | max NGP error | event-rate error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H16 | 0.97827 | 0.27100 | 0.85729 | 0.33586 | 0.85729 | 2.9840 | 0.8194 | 0.2519 | 0.4840 |
| H16+u | 0.97832 | 0.27305 | 0.95120 | 0.33399 | 0.85892 | 2.3996 | 0.7610 | 0.2650 | 0.2441 |
| H16+up | 0.97853 | 0.28144 | 0.85589 | 0.32131 | 0.85589 | 3.6344 | 0.8762 | 0.2552 | 1.8431 |

Position alone creates an almost deterministic unresolved mode: its residual
lag rises to `0.95120`. Adding the conjugate velocity removes that artifact
and lowers tagged-velocity residual lag by about 4.3% on the worst-fold
metric. Every held clone improves slightly.

The improvement is nevertheless far below the preregistered requirement. The
overall lag ratio is `0.99836`, because the force-history modes remain colored
at `0.85589`. The autonomous transport prediction also degrades: terminal
diffusion error is `3.6344` and event-rate error is `1.8431`.

## Physical Verdict

The smooth cage is a valid microscopic coordinate and contains a small part
of the missing tagged-velocity state. It is not, by itself, the orthogonal
force memory required to close the single-particle dynamics. Appending it to
a raw total-force Hankel state mixes two physical decompositions:

```text
x = C + u
```

but keeps the unresolved bath expressed in the total tagged force. The next
test should instead project deterministic acceleration into relative and cage
center pieces,

```text
b_u = J F + Hess(u):(V tensor V) - gamma p,
b_C = F_i - gamma v_i - b_u,
```

and learn separate causal histories for those two microscopic drifts. If that
still leaves colored residuals, `u,p,J J^T,b_u,b_C` become physically
constrained inputs to a training-only slow-coordinate learner such as a
TICA/VAMP basis; event labels and macro curves must remain excluded.

The claim boundary is

```text
smooth_cage_hankel_bath_allowed = 0
autonomous_single_particle_gle_allowed = 0
complete_event_clock_closure_allowed = 0
kramers_escape_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

## Literature Connection

The need to separate nonlinear reaction-coordinate drift from nonlinear
memory is consistent with the hybrid Mori-Zwanzig derivation of Ayaz,
Scalfi, Dalton, and Netz, Phys. Rev. E 105, 054138 (2022),
https://doi.org/10.1103/PhysRevE.105.054138. If interpretable projected
coordinates remain insufficient, the variational slow-coordinate principle
provides a controlled learning route; see Mardt et al., Nature Communications
9, 5 (2018), https://doi.org/10.1038/s41467-017-02388-1.

## Reproduction

```bash
python scripts/analyze_ka_smooth_cage_hankel_bath.py \
  tmp/isoconfigurational_force_velocity_long_T058 \
  --cache-directory tmp/hankel_slow_force_bath_reduced_T058 \
  --cage-cache-directory tmp/smooth_cage_hankel_reduced_T058 \
  --output-prefix data/renewal_cage_ka_smooth_cage_hankel_bath_T058
```
