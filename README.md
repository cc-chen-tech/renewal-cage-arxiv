# Delayed Renewal Cage Model

[![CI](https://github.com/cc-chen-tech/renewal-cage-arxiv/actions/workflows/ci.yml/badge.svg)](https://github.com/cc-chen-tech/renewal-cage-arxiv/actions/workflows/ci.yml)

This repository contains a reproducible theoretical note on a minimal delayed
renewal cage model for dynamical glass signatures. It is an effective
microdynamic diagnostic theory with an explicit thermodynamic entropy closure,
not a first-principles derivation of the thermodynamic glass transition.

The model combines:

- local Ornstein-Uhlenbeck-like cage variance,
- delayed cage-renewal events,
- Gaussian cage-center jumps,
- closed-form MSD, NGP, self van Hove distribution, self-intermediate
  scattering function, temperature-dependent alpha relaxation, peak
  asymptotics, Stokes-Einstein decoupling, activated-barrier diagnostics,
  fractional Stokes-Einstein exponents, apparent alpha-activation/fragility
  diagnostics, alpha-shape time-temperature-superposition collapse residuals,
  renewal-count susceptibility, renewal-domain chi4/cooperative-size
  diagnostics, spatial facilitation chi4-length closure,
  thermodynamic entropy/Adam-Gibbs closure,
  effective MCT beta-window closure,
  NGP peak/alpha-relaxation coupling, finite-exchange
  heterogeneity diagnostics, a temperature-dependent facilitated-exchange law,
  a persistence/exchange renewal-clock decoupling diagnostic, a static-gamma
  mobility-disorder null model, late-time mechanism-selection diagnostics, a
  glass-dynamics phenomenon audit, barrier-facilitation signature phase
  diagrams, closed barrier-threshold requirements, finite-time consistency
  diagnostics, joint persistence/exchange inversion protocols, and observable
  inversion/falsifiability criteria, including a trajectory inversion readiness
  gate.

The core one-dimensional and three-dimensional NGP result is

```text
MSD(t) = L(t) + q R(t)

alpha_2(t) = q^2 R(t) / [L(t) + q R(t)]^2
```

where

```text
L(t) = A[1-exp(-t/tau_c)]

R(t) = lambda [t - 2 tau_d(1-exp(-t/tau_d))
               + (tau_d/2)(1-exp(-2t/tau_d))]
```

The square delay is the minimal integer choice in the generalized family
`r_m(t)=lambda[1-exp(-t/tau_d)]^m` that gives a regular zero-origin NGP.
The finite-time consistency diagnostic inverts the plateau relation
`alpha=beta*y/(1+y)^2`; the late-time check uses the `y>1` branch and requires
the observed late NGP to lie below the peak bound `beta/4`.
The scattering-transport inversion uses the measured Debye-Waller plateau,
`D`, `tau_alpha`, and `tau_d` to infer `A`, `q`, and `lambda`; it requires the
existence margin `D tau_d F(tau_alpha/tau_d) k^2 / [-log h]` to exceed one.
A fuller observable inversion uses `f_k`, `D`, NGP peak time, and NGP peak
height to infer `A`, `q`, `lambda`, and `tau_d`, leaving `tau_alpha` as a
held-out consistency check.
The persistence/exchange inversion uses `D` to fix `tau_x=q/(2D)`, solves
`G(a_k,tau_alpha;tau_p,tau_x)=h` for `tau_p`, and leaves late NGP recovery as a
held-out falsification observable.
The joint persistence/exchange protocol uses one anchor alpha time plus `D` to
infer the clocks, then holds out multi-`k` alpha times, late NGP recovery, and a
single-domain `chi4` proxy.
The uncertainty-weighted persistence/exchange protocol converts those held-out
residuals into `z` scores using measurement-error estimates.
The raw-curve persistence/exchange protocol starts one level earlier: it extracts
`tau_alpha(k)` from machine-readable `F_s(k,t)`, interpolates the late NGP,
takes the `chi4` peak from the raw susceptibility curve, and then runs the same
held-out `z`-score tests. A deliberately corrupted late-NGP raw curve is rejected
without refitting the inferred clocks.
The finite-exchange protocol adds a second data-level residual: late NGP gives
`c_NGP=R_l alpha_2(t_l)-1`, the alpha slope gives `c_alpha` through
`log(1+Gamma_k c_alpha)/c_alpha`, and `log(c_alpha/c_NGP)` tests whether both
observables share one exchange scale. With measurement uncertainties, the same
closed derivatives propagate this residual into a `z` score. A multi-`k`
extension requires all `c_alpha(k)` values from `F_s(k,t)` to collapse to the
same `c_NGP`.
A static-gamma null model gives `Var N=R+R^2/kappa0`,
`alpha_2(t)->1/kappa0`, and `-log Phi_alpha/R->0`; it can broaden alpha
relaxation but fails long-time Gaussian recovery.
The SOTA benchmark consistency table turns selected literature-level claims
into checkable rows, currently covering Debye-Waller cage localization,
Kob-Andersen beta-window visibility and MCT exponent-parameter consistency,
Gaussian-recovery mechanism selection against the static-disorder null,
cooling-induced NGP peak shift with late Gaussian recovery, fractional
Stokes-Einstein decoupling, dynamic-heterogeneity chi4 growth, alpha-shape
time-temperature-superposition breakdown, spatial facilitation-front
growth-law inversion, KWW stretched-alpha window fitting, persistence/exchange
transport inversion with held-out late-NGP validation, joint multi-k/chi4
persistence-exchange falsification, a van Hove tail/recovery consistency
check, an Angell/Adam-Gibbs fragility-growth scope check, and a
thermodynamic-transition no-go/scope-boundary row.
The SOTA claim-alignment audit then separates source claims into derived
dynamical support, effective-closure support, and explicit scope boundaries so
that spatial four-point claims and thermodynamic-transition claims are not
overstated as consequences of the renewal dynamics alone.
The SOTA signed-constraint audit goes one step further: each representative
paper contributes required trend signatures and forbidden overclaims, so the
repository can state which SOTA conclusions are matched, which are only
closure-assisted, and which thermodynamic claims remain outside renewal
dynamics.
The SOTA evidence-class gate then separates representative simulations,
colloid/molecular experiments, repository reanalysis candidates, protocol
canaries, and thermodynamic scope-boundary sources before any trend agreement
is promoted to uncertainty-weighted inversion.
The real-benchmark assimilation gate adds a stricter pre-fit check: a source
must have the required observables, a shared system/temperature grid,
machine-readable curves, and uncertainty estimates before it is promoted from
qualitative SOTA agreement to uncertainty-weighted inversion.
The trajectory-observable bridge adds the missing pre-curve layer for
trajectory-level SOTA datasets: particle positions are reduced to MSD, NGP,
self-intermediate scattering, and overlap-chi4 rows before raw-curve inversion.
The companion trajectory-uncertainty protocol uses time-origin block jackknife
estimates to attach uncertainty columns to those trajectory-derived observables,
turning the trajectory bridge into a quantitative pre-inversion input rather
than only a structural observable extractor.
The member-ensemble uncertainty protocol adds the independent-trajectory
version of the same pre-inversion layer: member-resolved trajectory observables
are aggregated into standard-error columns before persistence/exchange
inversion is allowed.
The trajectory inversion readiness gate then promotes trajectory-derived rows
only when the required structural observables, shared lag grid, shared particle
identity, and positive uncertainty columns are all present; otherwise it keeps
the source at the structural-trajectory-only stage.
The benchmark publication ladder collapses metadata, readiness, and held-out
prediction gates into manuscript-safe claim levels, separating current
GlassBench metadata readiness from synthetic protocol-canary evidence and
fit-only overclaim risks.
The cross-observable prediction ledger separates calibration inputs from
held-out predictions and external closures, making fit-only overclaims visible.
The simultaneous-closure gate sharpens this into a minimal dynamical test:
diffusion and one anchor `tau_alpha(k)` are calibration inputs, while multi-k
alpha shape, late NGP recovery, Stokes-Einstein growth, and the `chi4` proxy
must pass as held-out predictions; thermodynamic-transition claims remain
disallowed for this gate.
The inversion-identifiability audit then checks fit-rank margins, held-out
predictions, closure dependence, and parameter degeneracy before any real-data
fit is claimed.
The frontier benchmark horizon classifies recent SOTA sources as trajectory
reanalysis candidates, transport/heterogeneity candidates, closure horizons,
model-extension gaps, or thermodynamic scope boundaries.
The SOTA source-provenance gate then checks whether a cited source is actually
a reanalysis input by requiring explicit identifiers, machine-readable files,
raw trajectories or observable tables, protocol metadata, and reuse permission.
The SOTA data-accession manifest records DOI, landing URL, archive name,
checksum, size, license, schema/README status, and local-cache status before
any public dataset is treated as a completed real-data reanalysis.
The SOTA Zenodo-record fingerprint verifies the cached GlassBench API record
against DOI, CC-BY-4.0 license, README size/md5, and full-archive size/md5,
while still marking real trajectory reanalysis as blocked on the local
`GlassBench.zip` cache.
The remote ZIP central-directory gate uses a cached HTTP Range probe of the
official GlassBench archive to verify the real ZIP64 entry roots
`GlassBench/KA_trajectories`, `GlassBench/KA_models`, `GlassBench/KA_results`,
`GlassBench/KA2D_trajectories`, `GlassBench/KA2D_models`, and
`GlassBench/KA2D_results` without treating that as completed trajectory
reanalysis.
The GlassBench payload-index gate then maps those entries to concrete system
targets: KA2D has common trajectory/model/result payload temperatures
`0.23;0.30`, while KA has model/result temperatures `0.44;0.50;0.56;0.64`
but no remote trajectory payload beyond the README in the central-directory
cache.
The GlassBench trajectory-payload locator then sharpens the KA2D target to
concrete remote files: `GlassBench/KA2D_trajectories/T0.23.tar.xz` and
`GlassBench/KA2D_trajectories/T0.30.tar.xz`. This is the first file-level
trajectory reanalysis target in the SOTA chain, but it remains blocked at
`zip_entry_metadata` because the current central-directory cache stores paths
rather than per-entry offsets and compressed sizes.
The trajectory-entry metadata gate removes that ambiguity with small remote
range reads of the ZIP central directory and local file headers. It records CRC,
compression method, compressed size, uncompressed size, local-header offset, and
compressed-data byte range for both KA2D trajectory members. The gate is still
not a real trajectory fit: `T0.23.tar.xz` is 397.5 MB compressed and
`T0.30.tar.xz` is 2.98 GB compressed, so the current blocker becomes
`member_payload_size_policy`, with `trajectory_extraction_ready=0` and
`real_reanalysis_ready=0`.
The member-stream probe reads only the first 64 KB of each compressed trajectory
member and raw-deflate inflates a 1024-byte prefix. Both KA2D members begin with
the XZ magic `fd377a585a00`, so the payload type is verified as streamable
`tar.xz` without downloading the full members. This advances the evidence from
entry metadata to payload-prefix evidence, while still keeping
`trajectory_extraction_ready=0` until a streaming extraction and uncertainty
policy is implemented.
The inner-tar header probe pushes the same evidence chain one layer deeper with
4 MB range reads: after raw-deflate and XZ streaming, both KA2D members expose
`ustar` headers, root directories, and first `.npz` trajectory members
(`T0.23/test/N1290T0.23_202_tc05.npz` and
`T0.30/train/N1290T0.30_3_tc01.npz`). This verifies trajectory-layout evidence
without claiming full coordinate extraction or real persistence/exchange
inversion.
The NPZ schema probe opens those first streamed members and verifies actual
coordinate arrays. Both KA2D temperatures expose `positions.npy` with shape
`20x1290x2`, `types.npy` with length `1290`, and scalar `box.npy` metadata.
This is coordinate-schema evidence for real trajectory ingestion, but not a
full ensemble extraction or uncertainty-weighted model comparison.
The first-NPZ observable smoke test then reduces those streamed coordinates to
minimal-image frame-index MSD and two-dimensional NGP summaries. The first
`T=0.23` member has final MSD `3.98e-3` and peak NGP `0.150`; the first
`T=0.30` member has final MSD `5.41e-3` and peak NGP `0.179`. This proves
coordinate-to-observable ingestion, while still blocking physical time-series
comparison until time semantics, full ensemble extraction, and uncertainty
weights are available.
The first-NPZ observable-curve table retains the full 20-frame MSD, NGP,
multi-k self-intermediate-scattering, and single-origin overlap-chi4 proxy
sequence for both first members. It is the first real GlassBench coordinate
payload converted into the repository's raw-curve shape beyond MSD/NGP, but it
remains a frame-index, single-member artifact rather than a physical-time SOTA
inversion.
The GlassBench short-window trend canary then makes the first real
coordinate-level comparison quantitative: the `T=0.30` first member has a
final-frame MSD 1.36 times the `T=0.23` value, while both first members retain
positive two-dimensional NGP peaks. This passes only a short-window
MSD/NGP sanity check; the same row keeps SOTA inversion and thermodynamic
claims disabled until physical lag times, independent-member ensembles, and
uncertainty columns are supplied.
The trajectory-result timebase bridge then checks whether the cached
same-temperature GlassBench result time grids can calibrate those first-NPZ
frame indices. They cannot yet be attached: the `T=0.23` trajectory curve has
20 frames versus 8 result time points, and `T=0.30` has 20 frames versus 6
time points, with no explicit frame-time mapping. This converts the physical
time blocker into a quantitative requirement rather than an assumed unit
conversion.
The frame-time mapping audit then rejects the tempting shortcut of simply
interpolating the 20 trajectory frames onto the shorter result grids. The
current `T=0.23` and `T=0.30` rows fail both exact count matching and integer
stride subsampling, so endpoint interpolation remains only a provisional
candidate until `dump_interval`, saved-frame stride, frame origin, and the
result-time generation script are documented.
The GlassBench real-inversion gap ledger then collapses these gates into one
claim-level verdict. For both KA2D temperatures the allowed claim remains
`short_window_coordinate_trend_only`: the coordinate canary passes and
first-member `F_s`/overlap-chi4 proxies are cached, but the timebase gate
fails and physical lag-time semantics are still missing. The later extended
member-index and first-four-member observable gates add frame-index uncertainty
columns, but they do not promote the result to a calibrated physical-time
persistence/exchange inversion.
The real-inversion unlock protocol turns that boundary into a minimum data
payload: for each KA2D temperature it requires an explicit frame-time mapping,
independent `.npz` members, and positive uncertainty columns for MSD, NGP,
multi-k `F_s`, and overlap `chi4`. The frame-index member ensemble now
satisfies the member and uncertainty parts of that payload for the extracted
prefix members, while the physical timebase remains the promotion blocker.
Passing the full protocol would promote the comparison to
`uncertainty_weighted_real_trajectory_inversion`; it still would not enable a
thermodynamic glass-transition claim.
The first-NPZ inversion-readiness gate then turns that limitation into explicit
machine-readable blockers: physical lag times, multiple independent members,
and positive uncertainty columns are required before a real
persistence/exchange comparison is allowed. The first-member structural
observables are useful evidence, but they are not a calibrated physical-time
or ensemble-averaged inversion.
The NPZ member-index gate then removes the member-list ambiguity without
overstating the result. The earlier 1 MB inner-tar prefix exposed three KA2D
`.npz` members at both temperatures; an 8 MB extended byte-range probe now
indexes 9 valid `T=0.23` members and 10 valid `T=0.30` members. This passes
the four-member threshold for a future ensemble uncertainty calculation, but
no multi-member observable extraction or physical-time inversion is claimed.
The first-four member observable gate now performs that extraction for the
first four indexed members at each KA2D temperature and aggregates MSD, 2D NGP,
multi-k `F_s(k,t)`, and overlap-`chi4` into array-index means and standard
errors. The official KA2D trajectory README then corrects the semantics: the
leading `positions` axis is the 20-isoconfigurational-trajectory replica axis,
not a time axis, and the physical lag time is encoded by the file-name `tc`
code. The corrected time-code gate now maps the extracted 8 MB prefix members
to real lag times and recomputes fixed-time observables relative to
`initial_positions`. The result is asymmetric but useful: `T=0.23` covers all
eight official time codes (`tc05` through `tc40`) with at least four complete
members per time code, so it is a physical-time observable curve ready for the
next persistence/exchange inversion gate; `T=0.30` still covers only `tc01` and
therefore remains blocked by sparse time-code coverage.
The cached-particle observable-semantics audit now verifies the same boundary
at the single-structure cache level: raw coordinate MSD is rejected as a proxy,
while minimal-image displacement from cached `initial_positions` reproduces
the official GlassBench MSD for the structure-151 cold lag ladder to numerical
precision. It also resolves the official NGP convention: GlassBench reports
the mean over isoconfigurational replicas of each replica's two-dimensional
alpha2, not the pooled ratio over all replicas and particles. With that
convention, the cached displacements reproduce the official NGP as well. This
audit also reproduces the official multi-`k` self-intermediate scattering by
using the axis-averaged convention
`0.5*(<cos(k dx)> + <cos(k dy)>)`; a single-axis convention is explicitly
rejected. This upgrades the remaining blocker from fixed-lag observable
semantics to event segmentation, persistence/exchange clocks, and
alpha-threshold coverage.
The time-code curve bridge now runs that `T=0.23` curve through the same
trajectory pre-inversion schema used by the persistence/exchange protocol. It
promotes the row from metadata-only evidence to a real physical-time
observable curve, but it still blocks real inversion because the cached curve
does not cross the alpha threshold even though its latest point reaches about
`1.63 tau_alpha`. The alpha-horizon audit now estimates that the latest-lag
`F_s=e^-1` crossing would require `k*=2.70`, about `1.69x` the largest
published GlassBench wave number `k=1.6`; `T=0.30` remains blocked by sparse
time-code coverage.
The companion signature-support gate then asks what can already be concluded
without fitting: the real `T=0.23` curve supports MSD growth, substantial
anchor-`F_s` decay, a transient NGP peak with partial recovery, and a transient
overlap-`chi4` proxy peak with partial recovery. It still records
`thermodynamic_claim_allowed=0` and keeps the real persistence/exchange
inversion blocked by alpha-anchor wave-number coverage.
The alpha-anchor rescue protocol turns that blocker into a concrete
measurement target: recompute or extend GlassBench `F_s(k,t)` near `k*=2.70`
to test the same alpha threshold. It also records that this would only remove
the alpha-definition blocker; physical-time event clocks, threshold sweeps, and
held-out macro observables remain separate requirements before a real
persistence/exchange closed-loop claim.
The cached alpha-anchor `F_s` audit then tests that proposed wave number
directly on the structure-151 cached displacement ladder. At the latest lag,
`F_s(k=2.6966,t)` is `0.528`, still above `e^-1`. Fitting the same cached
latest-lag `F_s(k)` grid gives a structure-specific estimate `k*=3.01`, but a
direct bisection solve on the cached displacement tensor gives the stricter
root `k_root=4.80`. This refines the next measurement target without promoting
the result to an event-clock or thermodynamic claim.
The direct-alpha curve audit then evaluates all eight cached structure-151 lag
targets at this `k_root`: `F_s` decreases from `0.980` at `tc05` to `e^-1` at
`tc40`, so the cached coordinate ladder now contains a real structure-matched
alpha-threshold crossing. The audit still keeps `event_clock_trajectory_ready=0`
and `real_pe_inversion_ready=0`, because the NPZ axis is an isoconfigurational
replica axis rather than a physical event-clock trajectory.
The SOTA dynamic-signature alignment ledger then joins model diagnostics,
literature-level benchmarks, and the current GlassBench real curve. It marks
MSD growth/cage escape and transient NGP as model+literature+real-curve
supported, marks self-intermediate scattering as real-curve supported but still
pre-alpha-threshold, marks `chi4` as a proxy spatial-heterogeneity alignment,
keeps persistence/exchange decoupling blocked until real inversion, and leaves
thermodynamic transition as a scope boundary.
The visible-member ensemble audit adds the next guardrail: the prefix evidence
now shows member identities and split labels (`test` at `T=0.23`, `train` at
`T=0.30`) beyond the four-member threshold. It therefore marks the member-list
gate ready, while the real inversion remains blocked until the cached
GlassBench lag window is extended far enough for alpha-threshold crossing and
long-time diffusion checks.
The observable-coverage audit isolates the remaining real-inversion observable
gap: both current KA2D first-NPZ rows now expose frame index, MSD, 2D NGP,
multi-k `F_s(k,t)`, and a single-origin overlap-`chi4` proxy. They still lack
physical `lag_time`, and the audit explicitly forbids substituting `rhomax` or
ML feature curves for the dynamical observables in any later comparison.
The first-NPZ structural-observable plan then separates a missing-data issue
from a theory/protocol issue. The visible schemas contain `positions.npy`,
`box.npy`, and `types.npy`, and the repository already has a trajectory
observable protocol that can compute MSD, NGP, multi-k `F_s`, and overlap
`chi4`. The current cache records derived structural observables but does not
retain the raw coordinate bytes, so the gate records
`structural_observables_cached_raw_coordinates_not_retained` and keeps the
remaining blocker at physical lag-time semantics rather than coordinate
availability.
The SOTA remote result-curve cache adds the first byte-range verified numeric
curve layer from the public GlassBench archive. It verifies small KA time-grid
and `rhomax_md` result files and KA2D time-grid, `rhomax_md`, and `rhomax_bb`
files by CRC32, md5, byte range, and numeric row/column counts. The
KA `chi4` update file is visible in the remote central directory but is not
yet in the range cache, so it remains a targeted byte-range fetch issue rather
than a claimed observable comparison or inversion input.
The result-curve fetch-gap gate records this explicitly:
`GlassBench/KA_results/chi4_KA_T0.44_update.dat` is a
`dynamic_heterogeneity_chi4_proxy` target with `targeted_fetch_ready=1`,
`observable_comparison_ready=0`, and `real_inversion_ready=0`.
The target-fetch gate then records the actual 36-byte range payload for that
file. The payload is checksum-ready, but it contains only the header
`t True Shiba Alkemade Jung Francois` and no numeric rows, so the status is
`target_fetch_header_only_parse_blocked`, not a `chi4` comparison.
The published-curve semantic audit checks all cached GlassBench `FIG*.dat`
payload headers. Rows with headers such as `BOTAN`, `CAGE`, `GlassMLP`, `SE3`,
`DEN`, and `EPOT` are kept as ML/figure benchmark curves, not promoted to
physical observables such as MSD, `F_s`, NGP, diffusion, or `chi4`.
The companion payload adapter stores the actual numeric rows from those byte
ranges and pairs time grids with same-temperature `rhomax` curves. KA2D at
`T=0.30` has structurally adapter-ready `rhomax_md` and `rhomax_bb` rows; KA at
`T=0.44` remains blocked because the cached `rhomax_md` payload has missing
value entries at the last two time points. All rows remain
`real_inversion_ready=0` until uncertainty columns and model-observable
semantics are supplied.
The observable-semantics gate then keeps those `rhomax` rows as
overlap-density proxies rather than alpha, NGP, diffusion, or `chi4` inputs.
This lets the repository use real GlassBench result payloads without promoting
them to persistence/exchange inversion evidence before the required diagnostic
semantics and uncertainty columns exist.
The SOTA README-schema gate checks systems, folder tokens, reuse license, and
citation guidance before claiming that a remote dataset can support a local
trajectory adapter.
The translation-rotation protocol adds an effective rotational renewal clock
for Debye-Stokes-Einstein and near-Tg molecular-motion diagnostics without
claiming a microscopic orientational potential theory.
The alpha-shape TTS diagnostic rescales the cage-normalized alpha relaxation by
`tau_alpha`:

```text
Y_k(u;T) = -log Phi_alpha(k,u tau_alpha;T) / [-log h]
C_k(T) = Gamma_k(T) lambda(T) tau_d(T)
```

Within the minimal model, the whole scaled shape is controlled by `C_k`; exact
time-temperature superposition requires `C_k` to be temperature independent.
The NGP peak and alpha relaxation are also linked by renewal counts:
`R_peak=A/q` and
`R_alpha=-log(h)/[1-exp(-k^2 q/2)]`, so their time ratio is fixed by the same
delayed-renewal inverse.

## Repository Layout

```text
src/renewal_cage.py                         closed-form model functions
tests/test_renewal_cage.py                  unit tests
scripts/generate_renewal_cage_results.py    reproducible data/figure generator
scripts/build_arxiv_package.py              arXiv PDF figure and source builder
scripts/compile_latex.sh                    LaTeX compile helper
data/                                      generated CSV outputs
figures/                                   generated SVG figures
docs/                                      derivation and literature positioning notes
docs/arxiv-readiness-checklist.md          submission-readiness checklist
manuscript/renewal-cage-arxiv-draft.md      prose draft
paper/main.tex                             arXiv-style LaTeX manuscript
paper/references.bib                       bibliography
dist/renewal-cage-arxiv-source.zip          arXiv source package
```

## Reproduce

```bash
python3 -m unittest discover -s tests -v
python3 scripts/generate_renewal_cage_results.py
python3 scripts/build_arxiv_package.py
```

If a TeX distribution is installed, compile the manuscript with:

```bash
bash scripts/compile_latex.sh
```

The GitHub Actions workflow runs the tests, regenerates data and figures, builds the
arXiv source package, installs TeX Live, and compiles `paper/main.tex`.

Expected outputs include:

```text
figures/renewal_cage_results.svg
figures/renewal_cage_dimensionless.svg
figures/renewal_cage_scattering.svg
figures/renewal_cage_temperature.svg
figures/renewal_cage_alpha_shape.svg
figures/renewal_cage_facilitated_exchange.svg
figures/renewal_cage_persistence_exchange.svg
figures/renewal_cage_persistence_exchange_protocol.svg
figures/renewal_cage_persistence_exchange_joint_protocol.svg
figures/renewal_cage_persistence_exchange_uncertainty_protocol.svg
figures/renewal_cage_translation_rotation_protocol.svg
figures/renewal_cage_glass_audit.svg
figures/renewal_cage_glass_phase_diagram.svg
figures/renewal_cage_spatial_chi4.svg
figures/renewal_cage_thermodynamic_closure.svg
figures/renewal_cage_mct_beta_closure.svg
figures/renewal_cage_sota_benchmark_consistency.svg
figures/renewal_cage_sota_claim_alignment.svg
figures/renewal_cage_sota_signed_constraints.svg
figures/renewal_cage_sota_evidence_class.svg
figures/renewal_cage_simultaneous_closure.svg
figures/renewal_cage_real_benchmark_assimilation_gate.svg
figures/renewal_cage_cross_observable_prediction_ledger.svg
figures/renewal_cage_inversion_identifiability_audit.svg
figures/renewal_cage_frontier_benchmark_horizon.svg
figures/renewal_cage_sota_source_provenance.svg
figures/renewal_cage_sota_data_accession.svg
figures/renewal_cage_sota_zenodo_record_fingerprint.svg
figures/renewal_cage_sota_remote_zip_central_directory.svg
figures/renewal_cage_sota_glassbench_payload_index.svg
figures/renewal_cage_sota_glassbench_trajectory_payload_locator.svg
figures/renewal_cage_sota_glassbench_trajectory_entry_metadata.svg
figures/renewal_cage_sota_glassbench_trajectory_member_stream_probe.svg
figures/renewal_cage_sota_glassbench_trajectory_inner_tar_header_probe.svg
figures/renewal_cage_sota_glassbench_trajectory_npz_schema_probe.svg
figures/renewal_cage_sota_glassbench_trajectory_first_npz_observable_smoke.svg
figures/renewal_cage_sota_glassbench_trajectory_first_npz_observable_curve.svg
figures/renewal_cage_sota_glassbench_short_window_trend_canary.svg
figures/renewal_cage_sota_glassbench_trajectory_timebase_bridge.svg
figures/renewal_cage_sota_glassbench_frame_time_mapping_audit.svg
figures/renewal_cage_sota_glassbench_real_inversion_gap_ledger.svg
figures/renewal_cage_sota_glassbench_real_inversion_unlock_protocol.svg
figures/renewal_cage_sota_glassbench_trajectory_first_npz_inversion_readiness.svg
figures/renewal_cage_sota_glassbench_trajectory_npz_member_index.svg
figures/renewal_cage_sota_glassbench_trajectory_member_ensemble_observable.svg
figures/renewal_cage_sota_glassbench_ka2d_timecode_semantics.svg
figures/renewal_cage_sota_glassbench_timecode_curve_bridge.svg
figures/renewal_cage_sota_glassbench_alpha_threshold_horizon.svg
figures/renewal_cage_sota_glassbench_alpha_anchor_rescue_protocol.svg
figures/renewal_cage_sota_glassbench_alpha_anchor_cached_fs.svg
figures/renewal_cage_sota_glassbench_direct_alpha_curve.svg
figures/renewal_cage_sota_glassbench_timecode_signature_support.svg
figures/renewal_cage_sota_dynamic_signature_alignment.svg
figures/renewal_cage_sota_glassbench_cage_jump_proxy_canary.svg
figures/renewal_cage_sota_glassbench_cached_particle_timecode_bridge.svg
figures/renewal_cage_sota_glassbench_multilag_particle_cache_targets.svg
figures/renewal_cage_sota_glassbench_cached_particle_observable_semantics.svg
figures/renewal_cage_sota_glassbench_event_clock_threshold_readiness.svg
figures/renewal_cage_sota_glassbench_first_npz_particle_cache_contract.svg
figures/renewal_cage_sota_glassbench_microdynamic_closed_loop.svg
figures/renewal_cage_sota_glassbench_trajectory_npz_ensemble_horizon.svg
figures/renewal_cage_sota_glassbench_visible_member_ensemble_audit.svg
figures/renewal_cage_sota_glassbench_observable_coverage_audit.svg
figures/renewal_cage_sota_glassbench_first_npz_structural_observable_plan.svg
figures/renewal_cage_sota_remote_result_curve_cache.svg
figures/renewal_cage_sota_remote_result_curve_fetch_gap.svg
figures/renewal_cage_sota_remote_result_curve_target_fetch.svg
figures/renewal_cage_sota_remote_result_curve_published_semantics.svg
figures/renewal_cage_sota_remote_result_curve_payload_adapter.svg
figures/renewal_cage_sota_remote_result_curve_observable_semantics.svg
figures/renewal_cage_sota_readme_schema.svg
figures/renewal_cage_trajectory_adapter_contract.svg
figures/renewal_cage_literature_inversion_readiness.svg
figures/renewal_cage_observable_falsification_matrix.svg
figures/renewal_cage_benchmark_fusion_readiness.svg
figures/renewal_cage_raw_curve_ingestion_contract.svg
figures/renewal_cage_raw_curve_diagnostic_readiness.svg
figures/renewal_cage_raw_curve_persistence_exchange_protocol.svg
figures/renewal_cage_trajectory_observable_protocol.svg
figures/renewal_cage_trajectory_cage_jump_events.svg
figures/renewal_cage_trajectory_event_clock_macro_predictions.svg
figures/renewal_cage_trajectory_event_clock_threshold_robustness.svg
figures/renewal_cage_trajectory_uncertainty_protocol.svg
figures/renewal_cage_trajectory_member_ensemble_uncertainty.svg
figures/renewal_cage_trajectory_inversion_readiness.svg
figures/renewal_cage_benchmark_publication_ladder.svg
figures/renewal_cage_barrier_requirements.svg
figures/renewal_cage_barrier.svg
figures/renewal_cage_heterogeneity.svg
figures/renewal_cage_heterogeneity_map.svg
figures/renewal_cage_static_null.svg
figures/renewal_cage_mechanism_selection.svg
figures/renewal_cage_inversion.svg
data/renewal_cage_main.csv
data/renewal_cage_sweeps.csv
data/renewal_cage_dimensionless.csv
data/renewal_cage_diagnostics.csv
data/renewal_cage_consistency.csv
data/renewal_cage_sota_comparison.csv
data/renewal_cage_van_hove.csv
data/renewal_cage_tail_ratios.csv
data/renewal_cage_scattering.csv
data/renewal_cage_peak_relaxation.csv
data/renewal_cage_temperature.csv
data/renewal_cage_alpha_shape.csv
data/renewal_cage_kww_alpha.csv
data/renewal_cage_facilitated_exchange.csv
data/renewal_cage_persistence_exchange.csv
data/renewal_cage_persistence_exchange_protocol.csv
data/renewal_cage_persistence_exchange_joint_protocol.csv
data/renewal_cage_persistence_exchange_uncertainty_protocol.csv
data/renewal_cage_translation_rotation_protocol.csv
data/renewal_cage_glass_audit.csv
data/renewal_cage_glass_phase_diagram.csv
data/renewal_cage_spatial_chi4.csv
data/renewal_cage_spatial_facilitation_inversion.csv
data/renewal_cage_thermodynamic_closure.csv
data/renewal_cage_mct_beta_closure.csv
data/renewal_cage_sota_benchmark_consistency.csv
data/renewal_cage_sota_claim_alignment.csv
data/renewal_cage_sota_signed_constraints.csv
data/renewal_cage_sota_evidence_class.csv
data/renewal_cage_simultaneous_closure.csv
data/renewal_cage_real_benchmark_assimilation_gate.csv
data/renewal_cage_cross_observable_prediction_ledger.csv
data/renewal_cage_inversion_identifiability_audit.csv
data/renewal_cage_frontier_benchmark_horizon.csv
data/renewal_cage_sota_source_provenance.csv
data/renewal_cage_sota_data_accession.csv
data/renewal_cage_sota_zenodo_record_fingerprint.csv
data/renewal_cage_sota_archive_preflight.csv
data/renewal_cage_sota_remote_zip_central_directory.csv
data/renewal_cage_sota_glassbench_payload_index.csv
data/renewal_cage_sota_glassbench_trajectory_payload_locator.csv
data/renewal_cage_sota_glassbench_trajectory_entry_metadata.csv
data/renewal_cage_sota_glassbench_trajectory_member_stream_probe.csv
data/renewal_cage_sota_glassbench_trajectory_inner_tar_header_probe.csv
data/renewal_cage_sota_glassbench_trajectory_npz_schema_probe.csv
data/renewal_cage_sota_glassbench_trajectory_first_npz_observable_smoke.csv
data/renewal_cage_sota_glassbench_trajectory_first_npz_observable_curve.csv
data/renewal_cage_sota_glassbench_short_window_trend_canary.csv
data/renewal_cage_sota_glassbench_trajectory_timebase_bridge.csv
data/renewal_cage_sota_glassbench_frame_time_mapping_audit.csv
data/renewal_cage_sota_glassbench_real_inversion_gap_ledger.csv
data/renewal_cage_sota_glassbench_real_inversion_unlock_protocol.csv
data/renewal_cage_sota_glassbench_trajectory_first_npz_inversion_readiness.csv
data/renewal_cage_sota_glassbench_trajectory_npz_member_index.csv
data/renewal_cage_sota_glassbench_trajectory_member_ensemble_observable.csv
data/renewal_cage_sota_glassbench_ka2d_timecode_semantics.csv
data/renewal_cage_sota_glassbench_timecode_curve_bridge.csv
data/renewal_cage_sota_glassbench_alpha_threshold_horizon.csv
data/renewal_cage_sota_glassbench_alpha_anchor_rescue_protocol.csv
data/renewal_cage_sota_glassbench_alpha_anchor_cached_fs.csv
data/renewal_cage_sota_glassbench_direct_alpha_curve.csv
data/renewal_cage_sota_glassbench_timecode_signature_support.csv
data/renewal_cage_sota_dynamic_signature_alignment.csv
data/renewal_cage_sota_glassbench_cage_jump_proxy_canary.csv
data/renewal_cage_sota_glassbench_cached_particle_timecode_bridge.csv
data/renewal_cage_sota_glassbench_multilag_particle_cache_targets.csv
data/renewal_cage_sota_glassbench_multilag_particle_cache_manifest.csv
data/renewal_cage_sota_glassbench_cached_particle_observable_semantics.csv
data/renewal_cage_sota_glassbench_event_clock_threshold_readiness.csv
data/renewal_cage_sota_glassbench_first_npz_particle_cache_contract.csv
data/renewal_cage_sota_glassbench_first_npz_particle_cache_manifest.csv
data/renewal_cage_sota_glassbench_microdynamic_closed_loop.csv
data/renewal_cage_sota_glassbench_trajectory_npz_ensemble_horizon.csv
data/renewal_cage_sota_glassbench_visible_member_ensemble_audit.csv
data/renewal_cage_sota_glassbench_observable_coverage_audit.csv
data/renewal_cage_sota_glassbench_first_npz_structural_observable_plan.csv
data/renewal_cage_sota_remote_result_curve_cache.csv
data/renewal_cage_sota_remote_result_curve_fetch_gap.csv
data/renewal_cage_sota_remote_result_curve_target_fetch.csv
data/renewal_cage_sota_remote_result_curve_published_semantics.csv
data/renewal_cage_sota_remote_result_curve_payload_adapter.csv
data/renewal_cage_sota_remote_result_curve_observable_semantics.csv
data/renewal_cage_sota_readme_digest.csv
data/third_party/glassbench/zenodo_record_10118191.json
data/third_party/glassbench/remote_zip_central_directory_10118191.json
data/third_party/glassbench/trajectory_entry_metadata_10118191.json
data/third_party/glassbench/trajectory_member_stream_probe_10118191.json
data/third_party/glassbench/trajectory_inner_tar_header_probe_10118191.json
data/third_party/glassbench/trajectory_npz_schema_probe_10118191.json
data/third_party/glassbench/trajectory_first_npz_observable_smoke_10118191.json
data/third_party/glassbench/trajectory_first_npz_observable_curve_10118191.json
data/third_party/glassbench/range_result_curve_cache_10118191.json
data/third_party/glassbench/range_result_curve_values_10118191.json
data/third_party/glassbench/range_result_curve_target_fetch_10118191.json
data/renewal_cage_sota_local_cache_verification.csv
data/renewal_cage_sota_zip_structure.csv
data/renewal_cage_sota_reanalysis_state.csv
data/renewal_cage_sota_evidence_verdict.csv
data/renewal_cage_sota_readme_schema.csv
data/renewal_cage_trajectory_adapter_contract.csv
data/renewal_cage_literature_inversion_readiness.csv
data/renewal_cage_observable_falsification_matrix.csv
data/renewal_cage_benchmark_fusion_readiness.csv
data/renewal_cage_raw_curve_ingestion_contract.csv
data/renewal_cage_raw_curve_diagnostic_readiness.csv
data/renewal_cage_raw_curve_persistence_exchange_protocol.csv
data/renewal_cage_trajectory_observable_protocol.csv
data/renewal_cage_trajectory_cage_jump_events.csv
data/renewal_cage_trajectory_event_clock_macro_predictions.csv
data/renewal_cage_trajectory_event_clock_threshold_robustness.csv
data/renewal_cage_trajectory_adapter_demo.csv
data/renewal_cage_trajectory_csv_adapter_source.csv
data/renewal_cage_trajectory_csv_adapter_demo.csv
data/renewal_cage_trajectory_curve_bridge.csv
data/renewal_cage_trajectory_curve_pe_gate.csv
data/renewal_cage_trajectory_pe_heldout_predictions.csv
data/renewal_cage_trajectory_prediction_falsification.csv
data/renewal_cage_trajectory_uncertainty_protocol.csv
data/renewal_cage_trajectory_member_ensemble_uncertainty.csv
data/renewal_cage_trajectory_inversion_readiness.csv
data/renewal_cage_benchmark_publication_ladder.csv
data/renewal_cage_barrier_requirements.csv
data/renewal_cage_susceptibility.csv
data/renewal_cage_chi4.csv
data/renewal_cage_barrier.csv
data/renewal_cage_heterogeneity.csv
data/renewal_cage_heterogeneity_diagnostics.csv
data/renewal_cage_heterogeneity_map.csv
data/renewal_cage_heterogeneity_protocol.csv
data/renewal_cage_heterogeneity_multik.csv
data/renewal_cage_static_null.csv
data/renewal_cage_mechanism_selection.csv
data/renewal_cage_inversion.csv
data/renewal_cage_full_inference.csv
paper/figures/renewal_cage_results.pdf
paper/figures/renewal_cage_dimensionless.pdf
paper/figures/renewal_cage_scattering.pdf
paper/figures/renewal_cage_temperature.pdf
paper/figures/renewal_cage_alpha_shape.pdf
paper/figures/renewal_cage_facilitated_exchange.pdf
paper/figures/renewal_cage_persistence_exchange.pdf
paper/figures/renewal_cage_persistence_exchange_protocol.pdf
paper/figures/renewal_cage_persistence_exchange_joint_protocol.pdf
paper/figures/renewal_cage_persistence_exchange_uncertainty_protocol.pdf
paper/figures/renewal_cage_translation_rotation_protocol.pdf
paper/figures/renewal_cage_glass_audit.pdf
paper/figures/renewal_cage_glass_phase_diagram.pdf
paper/figures/renewal_cage_spatial_chi4.pdf
paper/figures/renewal_cage_thermodynamic_closure.pdf
paper/figures/renewal_cage_mct_beta_closure.pdf
paper/figures/renewal_cage_sota_benchmark_consistency.pdf
paper/figures/renewal_cage_sota_claim_alignment.pdf
paper/figures/renewal_cage_sota_signed_constraints.pdf
paper/figures/renewal_cage_sota_evidence_class.pdf
paper/figures/renewal_cage_simultaneous_closure.pdf
paper/figures/renewal_cage_real_benchmark_assimilation_gate.pdf
paper/figures/renewal_cage_cross_observable_prediction_ledger.pdf
paper/figures/renewal_cage_inversion_identifiability_audit.pdf
paper/figures/renewal_cage_frontier_benchmark_horizon.pdf
paper/figures/renewal_cage_sota_source_provenance.pdf
paper/figures/renewal_cage_sota_data_accession.pdf
paper/figures/renewal_cage_sota_zenodo_record_fingerprint.pdf
paper/figures/renewal_cage_sota_remote_zip_central_directory.pdf
paper/figures/renewal_cage_sota_glassbench_payload_index.pdf
paper/figures/renewal_cage_sota_glassbench_trajectory_payload_locator.pdf
paper/figures/renewal_cage_sota_glassbench_trajectory_entry_metadata.pdf
paper/figures/renewal_cage_sota_glassbench_trajectory_member_stream_probe.pdf
paper/figures/renewal_cage_sota_glassbench_trajectory_inner_tar_header_probe.pdf
paper/figures/renewal_cage_sota_glassbench_trajectory_npz_schema_probe.pdf
paper/figures/renewal_cage_sota_glassbench_trajectory_first_npz_observable_smoke.pdf
paper/figures/renewal_cage_sota_glassbench_trajectory_first_npz_observable_curve.pdf
paper/figures/renewal_cage_sota_glassbench_short_window_trend_canary.pdf
paper/figures/renewal_cage_sota_glassbench_trajectory_timebase_bridge.pdf
paper/figures/renewal_cage_sota_glassbench_frame_time_mapping_audit.pdf
paper/figures/renewal_cage_sota_glassbench_real_inversion_gap_ledger.pdf
paper/figures/renewal_cage_sota_glassbench_real_inversion_unlock_protocol.pdf
paper/figures/renewal_cage_sota_glassbench_trajectory_first_npz_inversion_readiness.pdf
paper/figures/renewal_cage_sota_glassbench_trajectory_npz_member_index.pdf
paper/figures/renewal_cage_sota_glassbench_trajectory_member_ensemble_observable.pdf
paper/figures/renewal_cage_sota_glassbench_ka2d_timecode_semantics.pdf
paper/figures/renewal_cage_sota_glassbench_timecode_curve_bridge.pdf
paper/figures/renewal_cage_sota_glassbench_alpha_threshold_horizon.pdf
paper/figures/renewal_cage_sota_glassbench_alpha_anchor_rescue_protocol.pdf
paper/figures/renewal_cage_sota_glassbench_alpha_anchor_cached_fs.pdf
paper/figures/renewal_cage_sota_glassbench_direct_alpha_curve.pdf
paper/figures/renewal_cage_sota_glassbench_timecode_signature_support.pdf
paper/figures/renewal_cage_sota_dynamic_signature_alignment.pdf
paper/figures/renewal_cage_sota_glassbench_cage_jump_proxy_canary.pdf
paper/figures/renewal_cage_sota_glassbench_cached_particle_timecode_bridge.pdf
paper/figures/renewal_cage_sota_glassbench_multilag_particle_cache_targets.pdf
paper/figures/renewal_cage_sota_glassbench_cached_particle_observable_semantics.pdf
paper/figures/renewal_cage_sota_glassbench_event_clock_threshold_readiness.pdf
paper/figures/renewal_cage_sota_glassbench_first_npz_particle_cache_contract.pdf
paper/figures/renewal_cage_sota_glassbench_microdynamic_closed_loop.pdf
paper/figures/renewal_cage_sota_glassbench_trajectory_npz_ensemble_horizon.pdf
paper/figures/renewal_cage_sota_glassbench_visible_member_ensemble_audit.pdf
paper/figures/renewal_cage_sota_glassbench_observable_coverage_audit.pdf
paper/figures/renewal_cage_sota_glassbench_first_npz_structural_observable_plan.pdf
paper/figures/renewal_cage_sota_remote_result_curve_cache.pdf
paper/figures/renewal_cage_sota_remote_result_curve_fetch_gap.pdf
paper/figures/renewal_cage_sota_remote_result_curve_target_fetch.pdf
paper/figures/renewal_cage_sota_remote_result_curve_published_semantics.pdf
paper/figures/renewal_cage_sota_remote_result_curve_payload_adapter.pdf
paper/figures/renewal_cage_sota_remote_result_curve_observable_semantics.pdf
paper/figures/renewal_cage_sota_readme_schema.pdf
paper/figures/renewal_cage_trajectory_adapter_contract.pdf
paper/figures/renewal_cage_literature_inversion_readiness.pdf
paper/figures/renewal_cage_observable_falsification_matrix.pdf
paper/figures/renewal_cage_benchmark_fusion_readiness.pdf
paper/figures/renewal_cage_raw_curve_ingestion_contract.pdf
paper/figures/renewal_cage_raw_curve_diagnostic_readiness.pdf
paper/figures/renewal_cage_raw_curve_persistence_exchange_protocol.pdf
paper/figures/renewal_cage_trajectory_observable_protocol.pdf
paper/figures/renewal_cage_trajectory_cage_jump_events.pdf
paper/figures/renewal_cage_trajectory_event_clock_macro_predictions.pdf
paper/figures/renewal_cage_trajectory_event_clock_threshold_robustness.pdf
paper/figures/renewal_cage_trajectory_uncertainty_protocol.pdf
paper/figures/renewal_cage_trajectory_member_ensemble_uncertainty.pdf
paper/figures/renewal_cage_trajectory_inversion_readiness.pdf
paper/figures/renewal_cage_benchmark_publication_ladder.pdf
paper/figures/renewal_cage_barrier_requirements.pdf
paper/figures/renewal_cage_barrier.pdf
paper/figures/renewal_cage_heterogeneity.pdf
paper/figures/renewal_cage_heterogeneity_map.pdf
paper/figures/renewal_cage_static_null.pdf
paper/figures/renewal_cage_mechanism_selection.pdf
paper/figures/renewal_cage_inversion.pdf
dist/renewal-cage-arxiv-source.zip
```

## Current Status

This is a research draft intended to become a short arXiv note. The model,
figures, arXiv source package, and LaTeX manuscript build are reproducible in
CI. The remaining submission-level checks are tracked in
`docs/arxiv-readiness-checklist.md`.
