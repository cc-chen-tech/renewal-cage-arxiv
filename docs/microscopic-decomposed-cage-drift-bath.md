# Microscopic Decomposed Cage-Drift Bath Test

## Question

The smooth cage coordinate is an exact projection of the many-particle
Langevin process, but appending it to a raw tagged-force history did not close
the orthogonal memory. This experiment asks whether the problem is merely the
mixing of cage-center and relative dynamics in one total-force bath.

## Exact Microscopic Split

For unit masses, the parent process is

```text
dR = V dt,
dV = [F(R) - gamma V] dt + sqrt(2 gamma T) dW.
```

Let `u_i(R)=r_i-C_i(R)`, `J_i=grad_R u_i`, `p_i=J_i V`, and
`w_i=v_i-p_i`. The exact deterministic drifts are

```text
b_u = J_i F + Hess(u_i):(V tensor V) - gamma p_i,
b_C = F_i - gamma v_i - b_u.
```

If `A_i` is the tagged-particle block of `J_i` and `G_i=J_i J_i^T`,
the projected thermostat covariance rates are

```text
Q_u  = 2 gamma T G_i,
Q_C  = 2 gamma T [I - A_i - A_i^T + G_i],
Q_Cu = 2 gamma T [A_i^T - G_i].
```

The assembled `6 x 6` covariance is positive semidefinite and reconstructs
the tagged thermostat exactly:

```text
Q_C + Q_u + Q_Cu + Q_Cu^T = 2 gamma T I.
```

This is a coordinate transformation of the full 4096-particle Langevin
dynamics. It does not assume renewal statistics, Kramers escape, or a fitted
precursor.

## Conservative-Force Audit

The original trajectory `fx,fy,fz` contains the Langevin thermostat force and
cannot be used as `F(R)`. Each clone is therefore rerun from the parent LAMMPS
restart with the Langevin fix absent. The clean conservative force is checked
against an independent exact KA pair-force implementation on the same 64
fixed A particles.

Across four clones, the largest component error is `0.07064`, the largest RMS
relative error is `1.174e-5`, and the minimum correlation is
`0.99999999993`. The isolated maximum errors occur when finite text-coordinate
precision moves a pair across the truncated-potential cutoff; the RMS and
correlation gates rule out a mismatched force law. The projected noise
reconstruction error is at most `3.32e-16`, the minimum joint covariance
eigenvalue is `0.06534`, and centered directional-step sensitivity is
`2.22e-9`. The geometric drift is not negligible: its RMS is `0.16663` of the
projected-force RMS.

## Held-Clone Protocol

- four independent 10 tau Kob-Andersen Langevin clones at `T=0.58`;
- 64 fixed A particles, 64-frame histories, and whole-clone holdout;
- `raw-H16 = [v,z_total(16)]`;
- `split-8 = [w,p,u,z_C(8),z_u(8)]`, the fair primary rank budget;
- `split-16 = [w,p,u,z_C(16),z_u(16)]`, a larger diagnostic model;
- tagged displacement reconstructed only by integrating `w+p`;
- no event labels or macro observables used in fitting.

## Results

| Model | velocity R2 | max state corr. | max residual lag | D error | max Fs error | max NGP error | event-rate error |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw-H16 | 0.97826 | 0.27100 | 0.85729 | 2.9840 | 0.8194 | 0.2519 | 0.4840 |
| split-8 | 0.96672 | 0.27952 | 0.91516 | 20.3305 | 1.0080 | 0.2495 | 16.0207 |
| split-16 | 0.97863 | 0.66952 | 0.85097 | 3.1424 | 0.8336 | 0.2516 | 0.5643 |

The fair split-8 model fails in every relevant sense. Its worst residual lag
is 1.06749 times the raw baseline, and its nearly marginal retained dynamics
greatly overdiffuses. The larger split-16 model lowers residual lag in every
held clone, but only from `0.85729` to `0.85097`. That small gain is accompanied
by a large instantaneous residual-state correlation of `0.66952` and no macro
closure.

The residual is localized in both deterministic drift sectors. For split-16,
the maximum cage-center and relative-drift residual lags are `0.84827` and
`0.84342`; neither bath becomes close to white after the exact split.

## Physical Verdict

The exact cage-center/relative decomposition is a valid microscopic result,
and it exposes a non-negligible geometric drift and multiplicative correlated
noise. It does not make the projected tagged dynamics autonomous. Therefore
the missing state is not just an algebraic mixture of center and relative
force histories.

The remaining orthogonal dynamics must depend on unresolved many-body
configuration and likely requires nonlinear, state-conditioned memory. A
controlled next test should condition the two drift sectors on measured local
collective coordinates, learned only from training trajectories, and compare
against this fixed linear baseline. Simply adding more linear Hankel modes is
not supported by the present result.

The claim boundary remains

```text
decomposed_cage_drift_bath_allowed = 0
autonomous_single_particle_gle_allowed = 0
complete_event_clock_closure_allowed = 0
kramers_escape_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

## Reproduction

```bash
python scripts/analyze_ka_decomposed_cage_drift_bath.py \
  tmp/isoconfigurational_force_velocity_long_T058 \
  --cache-directory tmp/hankel_slow_force_bath_reduced_T058 \
  --cage-cache-directory tmp/smooth_cage_hankel_reduced_T058 \
  --drift-cache-directory tmp/decomposed_cage_drift_reduced_T058 \
  --rerun-directory tmp/decomposed_cage_drift_rerun_T058 \
  --lammps-binary tmp/toolchains/lammps-22Jul2025_update4/build-lepton/lmp \
  --parent-restart /Users/luicy/AI/renewal-cage-arxiv/tmp/ka_langevin_canary_T058/equilibrated.restart \
  --output-prefix data/renewal_cage_ka_decomposed_cage_drift_bath_T058
```
