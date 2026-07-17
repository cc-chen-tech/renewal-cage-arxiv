# Nonlinear Cage-Path Cumulant Gate Design

## Purpose

Determine whether the held-out low-temperature displacement statistics require
ordered single-particle path information beyond both the one-block displacement
distribution and the complete two-point displacement spectrum. The result is a
mechanism-selection gate, not yet a closed microscopic model.

The allowed claim is restricted to dynamical glass signatures. The gate must keep
`microdynamic_closure_claim_allowed=0`, `spatial_facilitation_claim_allowed=0`,
and `thermodynamic_claim_allowed=0`.

## Scientific hypothesis

Let `b_i(n)` be the three-dimensional block displacement of particle `i`. A
linear spectral path null is fixed by the one-block radial distribution and the
auto/cross spectrum of the three Cartesian components. If this null predicts the
held-out MSD but fails the NGP and self-intermediate scattering while the ordered
calibration path succeeds, then the missing information is a connected
higher-order temporal path cumulant.

This inference is single-particle and temporal. It does not establish a spatial
four-point correlation, collective facilitation, or a thermodynamic transition.

## Frozen data protocol

- Use the independently restarted Kob-Andersen ensembles already stored under
  `tmp/ka_replicates/T045` and `tmp/ka_replicates/T058`.
- The primary mechanism decision uses `T=0.45`, calibration time `5000`, block
  size `20`, and the existing held-out factorization rows.
- Score only held-out lags divisible by the block size. No held-out displacement,
  event sequence, MSD, NGP, or scattering value may enter a surrogate.
- Use the existing tolerances: maximum ensemble relative MSD error `0.10`,
  absolute NGP error `0.30`, and absolute multi-k scattering error `0.03`.
- `T=0.58` is a resolution sensitivity test at block sizes `20` and `10`. It
  cannot support a binary temperature-crossover claim unless both resolutions
  give the same mechanism verdict.

## Competing path mechanisms

1. `contiguous_empirical_path` is the existing calibration-only ordered path
   upper bound.
2. `within_particle_time_shuffle` preserves each particle's block-vector
   multiset but destroys all temporal order.
3. `phase_randomized_cross_spectrum` multiplies all three Fourier channels of a
   particle by a shared random phase at each frequency. It preserves that
   particle's full spectral matrix to numerical precision but does not preserve
   the non-Gaussian one-block distribution. It is a diagnostic, not the primary
   null.
4. `radial_multivariate_surrogate` alternates two projections:
   - a shared-phase Fourier projection toward the measured three-channel
     auto/cross spectrum;
   - a radial rank projection that exactly restores each particle's measured
     block-length multiset.

The final radial surrogate is required to preserve the one-block MSD and NGP
exactly up to floating-point error and the one-block multi-k scattering within a
fixed numerical tolerance. Fixed seeds and independent realizations are recorded.

## Surrogate quality gates

For every `T=0.45` replicate and realization:

- radial-distribution maximum absolute error at most `1e-12`;
- cross-spectral matrix NRMSE at most `0.015`;
- one-block relative MSD error at most `0.01`;
- one-block absolute NGP error at most `0.03`;
- one-block absolute scattering error at every k at most `0.003`.

Use at least eight deterministic realizations. Increase the count if the maximum
Monte Carlo standard errors exceed relative MSD `0.01`, NGP `0.03`, or scattering
`0.003`. Nonconverged realizations fail the gate; they are not discarded or
reseeded selectively.

## Stationarity control

Split the low-temperature calibration half into equal early and late quarters.
At common lags no longer than either quarter, compare early versus late, early
versus held-out, and late versus held-out ensemble MSD, NGP, and multi-k
scattering. All three comparisons must pass the unchanged curve tolerances before
phase-randomization failure can be interpreted as ordered path memory rather than
simple calibration drift.

## Mechanism decision

Set `nonlinear_single_particle_path_memory_required=1` only when all conditions
hold:

- the surrogate quality and Monte Carlo precision gates pass;
- the stationarity control passes;
- the existing contiguous empirical path passes all ensemble curve tolerances;
- the radial surrogate passes the ensemble MSD tolerance;
- the radial surrogate fails NGP or scattering at the ensemble level;
- every low-temperature replicate fails the normalized surrogate higher-order
  score, `max(NGP_error/0.30, Fs_error/0.03) > 1`;
- the contiguous path has a lower paired higher-order score than the radial
  surrogate in every low-temperature replicate.

Individual contiguous paths are not required to pass all tolerances. The primary
closure statement remains ensemble-level, and this distinction must be recorded
in the verdict.

If selected, also set:

- `one_block_radial_heterogeneity_sufficient=0`;
- `one_block_radial_plus_two_point_spectrum_sufficient=0`;
- `linear_spectrum_null_rejected=1`;
- `calibration_nonstationarity_supported=0`;
- `next_minimal_model=finite_lifetime_reversible_cage_state`;
- all microscopic-closure, spatial, and thermodynamic claim flags to zero.

## Connected-cumulant propagation

For an isotropic three-dimensional displacement with MSD `M(t)` and NGP
`alpha_2(t)`, the one-component cumulants are

```text
kappa_2^x(t) = M(t) / 3
kappa_4^x(t) = alpha_2(t) M(t)^2 / 3.
```

The fourth-order low-wave-number scattering diagnostic is therefore

```text
F_s^(4)(k,t) = exp[-k^2 M(t)/6 + k^4 alpha_2(t) M(t)^2/72].
```

This formula consumes supplied MSD and NGP values. When those values are observed,
the output is a diagnostic consistency test and must not be labeled a prediction.
It becomes predictive only after a calibration-only microscopic model predicts
both inputs.

For each temperature and wave number, report the longest tested lag for which the
absolute scattering error remains at most `0.03`. Do not extrapolate the truncated
cumulant expansion after it yields a scattering value above one or otherwise
leaves its measured validity range.

## Outputs

The implementation will produce:

- per-replicate and per-realization radial-surrogate rows;
- ensemble surrogate summaries and a low-temperature verdict;
- calibration-quarter stationarity rows and verdict;
- fourth-cumulant multi-k diagnostic rows and validity horizons;
- one combined mechanism-selection CSV;
- one deterministic SVG comparing errors normalized by their frozen tolerances.

Every artifact records seeds, iteration count, convergence errors, calibration and
held-out provenance, replicate count, and all claim-boundary flags.

## Tests

Pure tests must prove that shared Fourier phases preserve the three-channel
spectral matrix, radial projection preserves each particle's length multiset, the
iteration is deterministic for a fixed seed, and invalid or nonconverged inputs
fail explicitly. Decision-table tests cover every mechanism-selection condition,
including the ensemble-versus-individual distinction. Artifact tests freeze the
real low-temperature outcome and ensure that high-temperature resolution
sensitivity prevents an unsupported binary crossover claim.

## SOTA positioning

The surrogate construction follows multivariate phase randomization and iterative
amplitude-adjustment methods, but the research claim is their use as a
mechanism-selection test for glassy particle paths. Existing Kob-Andersen work
already reports correlated successive jumps and subdiffusion versus jump count at
`T=0.45`; this project must claim a stricter separation of one-block disorder,
two-point spectral memory, and connected higher-order path organization, not the
discovery of reversible or correlated cage jumps.

## Next model boundary

This gate does not uniquely identify or implement the reversible-cage model. If
selected, the recommended next candidate is a finite-lifetime semi-Markov cage
state whose parameters are estimated from calibration cage returns and
irreversible escapes. It must compete with a nonparametric higher-order path null
and predict the connected fourth path cumulant in addition to MSD before it is
allowed to claim low-k scattering closure.
