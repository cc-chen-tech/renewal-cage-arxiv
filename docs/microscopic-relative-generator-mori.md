# Microscopic Relative Generator Mori Checkpoint

## Question

The scalar harmonic Volterra model predicted held relative-coordinate
correlations but failed its scalar FDT test. This checkpoint asks whether the
failure came from forcing a matrix projected process into one scalar memory
kernel, and whether one additional coordinate can be obtained directly from
the many-particle Langevin generator rather than introduced phenomenologically.

## Microscopic Coordinate

The parent dynamics is

```text
dR = V dt
dV = [F(R) - gamma V] dt + sqrt(2 gamma T) dW.
```

For the smooth cage coordinate `u_i(R)=r_i-C_i(R)` and projected velocity
`p_i=J_i V`, Ito's formula gives the exact resolved drift

```text
L p_i = b_i
      = J_i F + Hess(u_i):(V tensor V) - gamma p_i.
```

The particle-specific offset `u0_i` is estimated from training clones only.
The two candidate resolved bases are

```text
g_2 = [u-u0, p]
g_3 = [u-u0, p, Lp].
```

Thus the third coordinate is the first Mori-Krylov generator image of the
relative velocity. It is recomputed from the full 4096-particle configuration,
clean conservative force, smooth-cage Jacobian, and Ito geometric term. It is
not a fitted precursor, event label, or macroscopic transport observable.

## Correct Discrete Mori Diagnostic

For the stationary matrix correlation

```text
C_n = <g_(t+n) g_t^T>,
```

the training operators follow the triangular recursion

```text
Omega_0 = C_1 C_0^-1,
Omega_k = [C_(k+1) - sum_(ell=0)^(k-1) Omega_ell C_(k-ell)] C_0^-1.
```

The corresponding time-origin-conditioned noise is

```text
W_k|i = g_(i+k+1) - sum_(ell=0)^k Omega_ell g_(i+k-ell).
```

This corrects an earlier exploratory diagnostic that treated a truncated
sliding residual as if it had to be white against every past state. Mori
orthogonality instead tests `W_k` against the resolved state at the projection
origin. The discrete generalized fluctuation-dissipation identity tested here
is

```text
Omega_k = -<W_k W_0^T> C(-Delta)^-1,  k >= 1.
```

These formulas follow the discrete Mori construction of Lin et al.,
[SIAM J. Appl. Dyn. Syst. 22, 2890 (2023)](https://doi.org/10.1137/21M1401759).
The distinction between a PMF, linear memory, and nonlinear memory is also
consistent with the hybrid projection derivation of Ayaz et al.,
[Phys. Rev. E 105, 054138 (2022)](https://doi.org/10.1103/PhysRevE.105.054138).

## Discovery

Four 10-tau `T=0.58` isoconfigurational clones were used in whole-clone
leave-one-out discovery. Memory orders `[1,4,16,32,40]` were scanned. The
selection therefore uses held discovery folds and is not confirmatory.

The phase-only basis never passed all gates through order 40. Adding `Lp`
strongly reduced the unresolved memory. The first all-gate candidate was
`g_3` at order 40, corresponding to `0.40 tau` of discrete memory.

| Discovery worst fold | `g_2`, order 40 | `g_3`, order 40 |
|---|---:|---:|
| noise/initial-state correlation | 0.06785 | 0.04900 |
| discrete-GFD NRMSE | 0.19900 | 0.09059 |
| minimum GFD shape correlation | 0.99232 | 0.99826 |
| target correlation RMSE | 0.06384 | 0.05845 |
| target correlation maximum error | 0.21715 | 0.19297 |

The generator basis passes the preregistered representation gate, while the
phase-only basis narrowly fails the maximum correlation error.

## Independent Validation

After fixing `g_3` and order 40, two new 10-tau clones were generated with
velocity seeds `92117,92139` and Langevin seeds `93117,93139`. The original
four clones supplied the bias, normalization, and Mori operators. No
validation trajectory was used for fitting or model selection.

The new conservative-force reruns agree with the independent exact KA force
implementation with relative RMS errors `4.66e-6` and `4.63e-6`, and force
correlations above `0.999999999989`. The fixed model passes every gate:

| Validation metric | clone 1 | clone 2 | required |
|---|---:|---:|---:|
| noise/initial-state correlation | 0.04129 | 0.03896 | <= 0.10 |
| discrete-GFD NRMSE | 0.07652 | 0.06855 | <= 0.20 |
| GFD shape correlation | 0.99853 | 0.99894 | >= 0.80 |
| target correlation RMSE | 0.04935 | 0.04560 | <= 0.08 |
| target correlation maximum error | 0.14162 | 0.15001 | <= 0.20 |

This is the first checkpoint in this project where a generator coordinate
computed from the many-particle Langevin dynamics gives a fixed held-out
matrix-memory representation that simultaneously predicts the relative
correlations and reproduces its projection-consistent noise/operator identity.

## What Is And Is Not Closed

The validated equation is

```text
g_(n+1) = sum_(ell=0)^40 Omega_ell g_(n-ell) + W_n,
g = [u-u0, p, Lp].
```

This is substantially more microscopic than the previous scalar renewal or
scalar Volterra closures. It identifies a concrete local force coordinate and
a finite `0.40 tau` memory scale from the many-particle trajectories.

Two stronger steps remain open.

First, the discrete GFD identity is a projection-consistency test. The parent
simulation uses a stochastic Langevin thermostat, whereas the simplest
thermal second-FDT derivation assumes a specific invariant inner product and
adjoint/time-reversal structure. That parity-aware thermal FDT has not yet
been derived and tested here. Projection formalisms for non-Hamiltonian
dynamics require this extra care; see Xing and Kim,
[J. Chem. Phys. 134, 044132 (2011)](https://doi.org/10.1063/1.3530071).

Second, `W_n` is reconstructed from microscopic validation trajectories. Its
conditional distribution and temporal generation law have not been closed.
Therefore this checkpoint cannot yet autonomously simulate a single tagged
particle, predict cage escapes, or reproduce `D`, NGP, and `F_s(k,t)` from the
reduced equation alone.

The claim boundary is

```text
generator_coordinate_is_microscopic = 1
generator_coordinate_memory_reduction_supported = 1
confirmatory_matrix_mori_gfd_closure_supported = 1
projected_relative_generator_mori_representation_allowed = 1
thermal_fdt_adjoint_audit_pass = 0
physical_relative_generator_gle_allowed = 0
orthogonal_noise_generation_closed = 0
autonomous_single_particle_gle_allowed = 0
complete_event_clock_closure_allowed = 0
kramers_escape_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

## Reproduction

```bash
python scripts/analyze_ka_relative_generator_mori.py \
  --training-drift-cache-directory tmp/decomposed_cage_drift_reduced_T058 \
  --output-prefix data/renewal_cage_ka_relative_generator_mori_discovery_T058

python scripts/cache_ka_decomposed_drift.py \
  tmp/isoconfigurational_force_velocity_mori_validation_T058 \
  --reduced-cache-directory tmp/relative_generator_mori_validation_reduced_T058 \
  --cage-cache-directory tmp/relative_generator_mori_validation_cage_T058 \
  --drift-cache-directory tmp/relative_generator_mori_validation_drift_T058 \
  --rerun-directory tmp/relative_generator_mori_validation_rerun_T058 \
  --lammps-binary tmp/toolchains/lammps-22Jul2025_update4/build-lepton/lmp \
  --parent-restart /Users/luicy/AI/renewal-cage-arxiv/tmp/ka_langevin_canary_T058/equilibrated.restart \
  --expected-clone-count 2

python scripts/analyze_ka_relative_generator_mori.py \
  --training-drift-cache-directory tmp/decomposed_cage_drift_reduced_T058 \
  --validation-drift-cache-directory tmp/relative_generator_mori_validation_drift_T058 \
  --fixed-memory-order 40 \
  --output-prefix data/renewal_cage_ka_relative_generator_mori_validation_T058
```
