# Delayed Renewal Cage Model

[![CI](https://github.com/cc-chen-tech/renewal-cage-arxiv/actions/workflows/ci.yml/badge.svg)](https://github.com/cc-chen-tech/renewal-cage-arxiv/actions/workflows/ci.yml)

This repository contains a reproducible theoretical note on a minimal delayed
renewal cage model for dynamical glass signatures. It is an effective
microdynamic diagnostic theory with an explicit external thermodynamic entropy closure,
not a first-principles derivation of the thermodynamic glass transition.

The real two-temperature segment-splice memory gate is documented in
`docs/segment-splice-memory-gate.md`. Its frozen 64-realization run is
fail-closed: the low-temperature precision gate fails at two short
within-particle horizons, only the excluded full-path controls pass all
low-temperature curve limits, and the warm stationarity control is unresolved.
Thus both finite memory lengths and the global mechanism verdict remain
unresolved. Both nulls preserve a shared global source-segment schedule, so any
later positive interpretation must remain conditional on that retained
cross-particle time structure.

A post-run paired-excess diagnostic is documented in
`docs/segment-splice-paired-excess.md`. After subtracting each replicate's own
full-path score, it finds a short-horizon excess-loss prefix through
`tau_L=200`, but the full-path replicate baseline still fails, so the mechanism
state remains unresolved and no finite-memory microscopic coordinate is added.
The three restart labels share one parent trajectory, so their t95 summaries
are exploratory correlated-parent diagnostics, not independent-sample
confidence intervals.

A post-run transport-clock / shape quotient further decomposes the failed
T=0.45 full-path transfer. Reparameterizing the calibration observables by
calibration MSD brings `Fs(k=2)` and `Fs(k=4)` within their frozen tolerances,
while NGP and `Fs(k=7.25)` retain residual failures. Heldout MSD is a diagnostic
input, not a blind prediction: this rejects clock-only closure without selecting
finite exchange, static disorder, or spatial facilitation. T=0.58 remains a
canary because stationarity and independent-parent controls are unresolved.

A gamma variance-mixture Langevin diagnostic then tests the minimal positive
scalar-mobility explanation. Heldout MSD and NGP predict `Fs(k=2)` and
`Fs(k=4)` within tolerance but fail at the T=0.45 cage-scale wave number
`k=7.25` (maximum normalized error `1.3854`). A squared-OU mobility simulation
validates the analytic slow-field limit, while an added OU-cage variance has
insufficient exact inversion support. This is a diagnostic rejection of the
scalar-mobility closure, not a blind microdynamic or spatial mechanism claim.

A fixed-MSD variance-mixture shape quotient then asks whether the remaining
multi-`k` residual is determined by the NGP residual. The ordinary fourth-order
cumulant truncation fails at high `k` (maximum normalized error `5.667` at
`T=0.45`), while both parameter-free gamma and inverse-Gaussian resummations
pass every supported `k=2,4,7.25` row (worst errors `0.466` and `0.423` in
tolerance units). Here, held-out MSD and NGP are diagnostic inputs, not blind
predictions. The result supports an exploratory transient marginal
variance-mixture closure but does not select a unique variance-mixture family,
static disorder, finite exchange, or a microscopic mechanism. `T=0.58` remains
a stationarity-unresolved canary.

An activated cage-jump geometry quotient then replaces the gamma shape by the
full calibration jump-vector characteristic. The measured broad jump law
improves the T=0.45 physical cage/count decomposition from `3/21` fixed-length
rows to `8/21`, but still misses the 80% support gate and has maximum
`Fs(k=7.25)` error `0.04164 > 0.03`. Thus one-jump geometry alone is
insufficient: count overdispersion, correlated jumps, or cage-jump coupling
must enter the next cumulant closure. Heldout MSD and NGP remain diagnostic
inputs, and all microscopic, spatial, unique-potential, and thermodynamic claim
flags remain zero.

A count-overdispersed geometry quotient now adds the calibration two-clock
Fano factor to the same cumulant inversion. It restores T=0.45 physical support
from `8/21` to `20/21`, but the maximum `Fs(k=7.25)` error remains
`0.04109 > 0.03`; count fluctuations fix the moment budget but not cage-scale
shape. A continuous transient-periodic Langevin canary then couples the tagged
coordinate to a slow elastic coordinate `q` and a squared-OU barrier coordinate
`z` in one FDT-consistent potential. Frozen ablations show that `z` raises
count Fano while `q` produces negative successive cage-step correlation, and
the full model produces both without discrete event rules. This is synthetic
mechanism capability, not identification of `q,z` in KA or blind macro closure;
all microscopic, spatial, unique-potential, and thermodynamic flags remain
zero. See `docs/microscopic-transient-periodic-langevin.md`.

A common-grid shape mechanism selection now puts the surviving marginal
variance-mixture class and calibration-measured event-path corrections under
the same T=0.45 support mask. Independent jumps fail only at high `k`; the
mildest disjoint-pair correction lowers that residual but breaks `k=2,4`, and
pair-eigenmode and full empirical-path kernels fail more strongly. In contrast,
both gamma and inverse-Gaussian variance-mixture resummations pass all 18 cells
and all three wave numbers. This selects a transient displacement shape class,
not a unique variance-mixture family or microscopic mechanism. Heldout MSD and
NGP remain diagnostic inputs, the three restart labels share one parent sample,
and all static/finite-exchange, cage-jump-coupling, spatial, microdynamic, and
thermodynamic claims remain closed. See `docs/ka-shape-mechanism-selection.md`.

A deterministic microscopic `L^2p` diffusion gate now removes the unresolved
32-probe numerical bottleneck. The full velocity Jacobian of `L^2p` is
contracted with exact KA pair Hessians to form
`Q_c=2 gamma T D_V(L^2p)D_V(L^2p)^T`. Four 200-frame held-clone tests show
positive constant-to-tensor NLL improvement in every fold (replicate-first
mean `0.27692`, 95% t interval `[0.20669,0.34715]`) and reject a time-permuted
tensor null. However, every exact-tensor fold still fails the frozen absolute
whitening gate through residual squared memory and excess kurtosis. The result
is mechanically `informative_but_insufficient`, authorizes an `L^3p`
derivation, and does not establish an autonomous single-particle Gaussian
Langevin bath. All environment, GLE, event-clock, Kramers, and thermodynamic
claim flags remain zero. See
`docs/microscopic-l2p-deterministic-diffusion.md`.

The model combines:

- local Ornstein-Uhlenbeck-like cage variance,
- a Langevin/Smoluchowski-to-Kramers bridge from local cage curvature and
  barrier inputs to effective renewal-cage clocks,
- delayed cage-renewal events,
- Gaussian cage-center jumps,
- closed-form MSD, NGP, self van Hove distribution, self-intermediate
  scattering function, temperature-dependent alpha relaxation, peak
  asymptotics, Stokes-Einstein decoupling, activated-barrier diagnostics,
  fractional Stokes-Einstein exponents, apparent alpha-activation/fragility
  diagnostics, alpha-shape time-temperature-superposition collapse residuals,
  renewal-count susceptibility, renewal-domain chi4/cooperative-size
  diagnostics, spatial facilitation chi4-length closure,
  external thermodynamic entropy/Adam-Gibbs closure,
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
The Langevin bridge starts one level earlier: in an overdamped local basin it
derives `D0=T/gamma`, `cage_variance=T/kappa_c`, `cage_tau=gamma/kappa_c`,
and Kramers rates `k=sqrt(kappa_c kappa_s)/(2 pi gamma) exp(-Delta F/T)`.
Those rates are then mapped to `tau_p=1/k_p` and `tau_x=1/k_x`, producing a
conditional coarse-graining bridge for the persistence/exchange clocks. The
full effective theory remains partly phenomenological because the delayed
renewal law, metastable-basin partition, barrier estimates, and jump scale are
not derived from an arbitrary many-body Langevin equation.
This matches the existing literature only in pieces: Langevin exit theory
supports Kramers-to-jump coarse graining, and glass simulations have used
cage-jump/CTRW descriptions. The specific delayed hazard
`r(t)=lambda[1-exp(-t/tau_d)]^2` plus closed-form NGP, `F_s(k,t)`, and SE
diagnostics is the extra modeling compression, not a standard automatic output
of ordinary Kramers theory.
The new periodic-softness bridge makes that compression less ad hoc. It starts
from a Vorselaars-style periodic cage potential
`U(x)=DeltaU[1-cos(2*pi*x/L)]/2`, derives the curvature
`kappa=2*pi^2*DeltaU/L^2`, computes the long-time Kramers rate, and then lets
two independent precursor readiness probabilities multiply:
`p_i(t)=1-exp(-t/tau_d)`. This gives
`r(t)=lambda p_1(t)p_2(t)=lambda[1-exp(-t/tau_d)]^2`.
Equivalently, cage rearrangement is not inserted as an ordinary Langevin drift:
the working stochastic picture is `dy_t=-(1/tau_c)y_t dt+sqrt(2D_c)dW_t`,
`dC_t=eta_t dN_t`, and `x_t = y_t + C_t`. The first term is local Gaussian
cage vibration; the second is the discrete cage-center renewal layer that
creates non-Gaussian displacement mixtures.
The broader physical picture is not a single static one-dimensional potential.
It is an extended coarse-grained landscape such as
`U(x,C,s1,s2,zeta)=(kappa/2)|x-C|^2+[DeltaU0+chi*zeta-eps*s1*s2]B(x-C)+W1(s1)+W2(s2)+Wzeta(zeta)`.
Different projections give different modules: the harmonic projection gives the
OU cage, the periodic projection gives the Vorselaars-type cage-to-cage
baseline, the softness-gate projection gives the delayed hazard, and the
mobility-environment projection gives finite-exchange heterogeneity. A
configurational-entropy or Kauzmann-type layer would require an additional
basin-counting landscape, not just this single-particle cage escape coordinate.
The machine-readable version is `data/renewal_cage_potential_taxonomy.csv`,
which maps each potential projection to derived parameters, supported effective
modules, supported observables, and the remaining many-body assumption.
The companion artifact `data/renewal_cage_landscape_parameterization.csv`
shows the first two numerical closures: basin adjacency gives the jump variance
`q`, and a discrete inherent-state density `Omega(e)` gives
`Z_conf`, `F_conf`, `s_c`, and `Delta c_p`.
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
The microdynamic prediction scorecard makes the core claim more compact:
particle event-clock statistics can pass or fail a no-macro-fit
micro-to-macro prediction canary, while the current GlassBench rows remain real
dynamical-signature support rather than completed microdynamic predictions.
The microdynamic minimality audit then checks that this is not a loose
post-hoc accounting exercise: persistence, exchange, jump variance, and cage
scale are all required before a micro-to-macro prediction claim is allowed.
The SOTA experimental verdict matrix consolidates the literature trend checks,
GlassBench evidence, microdynamic prediction scorecard, and thermodynamic
scope boundary into final manuscript-safe claim levels.
The GlassBench real-evidence claim synthesis then compresses the real-data
chain into five manuscript-facing rows: real dynamical signatures, cached
multi-k alpha-shape prediction, conditional alpha/transport PE bound, real
mechanism-selection readiness, and the thermodynamic scope boundary. It
therefore makes the current claim boundary explicit: real dynamic signatures
are supported, post-alpha and mechanism-selection checks are preregistered but
not observed, and thermodynamic glass-transition claims remain disallowed.
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
The structure-matched observable-renewal canary then asks a stricter question:
can the eight real fixed-lag observables be treated directly as a single
delayed-renewal event clock? The answer is no. For the cold structure-151
ladder, the real observables are strong (`max NGP=7.27`, maximum multi-`k`
`F_s` decay `0.311`), but the jump variance inferred from the minimal
plateau-plus-renewal moment equations has `q_eff` coefficient of variation
`1.76` and span `899`. This rejects the naive lag-clock closure while making
the next required step precise: segment particle-level cage-jump events rather
than fitting the lag ladder as if it were already a renewal clock.
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
The direct-alpha shape selection then compares this real cached alpha curve
with a threshold-anchored exponential null and a KWW shape fit. The current
curve strongly favors a stretched-alpha candidate (`beta=0.159`, KWW log-shape
RMSE `0.510` versus exponential RMSE `7.77`). Frame-block standard errors from
the cached particle tensor now provide machine-readable `sigma_F_s` values. The
only upward step has `z=0.78`, so the sparse nonmonotonicity is statistically
compatible with a monotone alpha decay under a two-sigma rule.
The direct-alpha multi-k shape gate then recomputes the same cached tensor at
three high wave numbers (`k=4.80`, `5.4`, and `6.0`). All three curves cross the
alpha threshold with a consistent KWW shape (`beta` spread `0.013`) and
uncertainty-compatible monotonicity (`z_max=0.804`). This is now a multi-k
stretched-alpha candidate, but not a completed real alpha-shape claim: all
three threshold crossings occur at the final cached lag `tc40`, so the remaining
blocker is post-alpha window depth rather than mechanism selection or sparse
nonmonotonicity.
The held-out multi-k alpha prediction gate then makes this stricter: each of
the three high-`k` curves is held out, the other two calibrate the KWW shape
exponent, and the held-out curve is predicted without refitting that curve. The
maximum held-out beta error is `0.00995`, and the maximum normalized shape RMSE
is `0.226`. This moves the GlassBench alpha-shape evidence from consistency
toward prediction, while preserving the same `tc40` edge-crossing blocker.
The post-alpha prediction-target gate then turns that blocker into explicit
future checks. Using the existing `tc50` late-recovery timecode target and a
log-time interpolated `tc45`, it preregisters high-`k` predictions:
`F_s(tc45)=0.236--0.241` and `F_s(tc50)=0.125--0.133`, with a fixed
absolute-log tolerance band. These rows are falsification targets for a future
GlassBench extension, not observed post-window evidence.
The post-alpha verdict gate then maps those preregistered targets and any
future post-window `F_s` observations into supported, rejected, indeterminate,
or not-ready rows. The current GlassBench verdict remains not-ready because no
post-window observation has been ingested, so this is a falsifiable verdict
protocol rather than observed support.
The direct-alpha transport proxy then matches that same `tc40` crossing to the
reproduced GlassBench displacement observable: `MSD=0.9747508406`, giving
`D_eff=1.6246e-7` and `D_eff tau_alpha=0.24369` in 2D. This is a useful
alpha/transport anchor for the real-data loop, not a persistence/exchange
ratio or a thermodynamic glass-transition claim.
The direct-alpha PE feasibility bound uses that anchor to test what would be
identifiable if a particle event clock supplied the per-event jump variance.
It finds that treating the full `tc40` MSD as a single jump variance is already
infeasible for the finite-exchange alpha/transport equations; the feasible
event jump variance is bounded by `q <= 0.48556`, about `0.498 MSD`. Under the
explicit conditional reference `q=0.2 MSD`, the same equations give
`tau_x=6.0e5`, `tau_p=1.409e6`, and `tau_p/tau_x=2.35`. This converts the
real GlassBench proxy into a falsifiable jump-variance target while keeping
`real_pe_inversion_ready=0`.
The direct-alpha displacement-tail bound then compares the raw cached `tc40`
displacement distribution with that PE single-event bound. In the
structure-151 cache, `q_all/q_max=1.004`, about `23.5%` of displacement samples
already exceed the single-event bound, and the above-bound tail has mean
`q_tail/q_max=3.69`. This rules out treating the broad direct-lag displacement
tail as one measured jump variance and makes cage-jump event segmentation the
next required real-data step.
The multi-lag crossing canary then scans the cached structure-151 lag ladder
against the same `q_max` threshold. About `24.0%` of replica-particle samples
cross at least once, most first crossings occur at `tc40`, and the mean
first-crossing `q` is `3.44 q_max`; however, `23.6%` of post-crossing samples
recross below the threshold and the leading axis remains an isoconfigurational
replica axis. The canary therefore supplies a concrete segmentation target but
still records `persistence_exchange_event_clock_ready=0`.
The real threshold-sweep canary then checks whether that segmentation target is
stable under reasonable changes of the jump threshold. Sweeping
`0.5--1.5 q_max` changes the inferred exponential persistence mean by a factor
`2.03`, and the maximum post-crossing recross fraction is `0.256`. This rejects
promotion of the current fixed-lag threshold crossing into a robust event clock:
the next input must be a true-time trajectory plus a preregistered
non-recrossing cage-jump definition.
The threshold-sweep ensemble verdict keeps that negative result from being
overgeneralized. In the present cached GlassBench payload, the cold KA2D
`T=0.23` structure has eight lag targets and fails by threshold sensitivity and
recrossing, while the hotter `T=0.30` structure has only one cached lag and is
blocked by coverage before any temperature comparison. The real-data claim is
therefore a falsified fixed-lag event-clock shortcut, not a completed
cross-temperature persistence/exchange inversion.
The payload contract makes this next step explicit rather than rhetorical:
`T=0.30` needs an official multi-lag member index and particle coordinates for
at least two additional lag targets before it can enter the same threshold
comparison, while the cold ladder needs a preregistered non-recrossing event
definition rather than more fixed-lag points.
The outcome matrix preregisters how the next payload will be interpreted: if
the hot ladder passes the same threshold-sensitivity and recrossing criteria on
a physical-time axis it becomes only a threshold-robust event-clock candidate;
if it fails, the fixed-lag threshold clock is rejected. Neither outcome permits
a completed PE inversion or thermodynamic glass-transition claim.
The decision-power plan adds the statistical guardrail for that future test:
the current cache has one independent member per temperature, so the
threshold-sweep decision remains blocked until at least three independent
member/structure-level sweeps are available. Pooled replica-particle counts are
recorded as `pooled_particle_decision_allowed=0`, because they do not supply
independent evidence for threshold robustness.
The real-cached microdynamic verdict then consolidates this lag ladder into a
stronger real-data statement: the interval-censored fit quantifies a cached
persistence-clock candidate with `tau_p/tau_alpha=3.65` and a crossed-fraction
residual below `10^-3`, while the conditional finite-exchange envelope gives a
`tau_p/tau_x` lower bound under the explicit exchange-clock assumption. It still
blocks full persistence/exchange inversion because the exchange clock, true
physical-time trajectory axis, and late-recovery measurement are not present.
The interval-censored waiting-law selection then asks whether the real cached
first-crossing ladder justifies adding a stretched/Weibull persistence law. It
does not: the fitted Weibull shape is near exponential (`shape=1.06`), and the
AIC penalty favors retaining the one-parameter exponential law for the current
sparse cache. This constrains the phenomenological waiting-time law without
claiming a stretched waiting distribution or thermodynamic glass transition.
The SOTA dynamic-signature alignment ledger then joins model diagnostics,
literature-level benchmarks, and the current GlassBench real curve. It marks
MSD growth/cage escape and transient NGP as model+literature+real-curve
supported, marks self-intermediate scattering as real-curve supported but still
pre-alpha-threshold, marks `chi4` as a proxy spatial-heterogeneity alignment,
keeps persistence/exchange decoupling blocked until real inversion, and leaves
thermodynamic transition as a scope boundary.
The direct four-point claim gate then prevents that `chi4` proxy from being
over-promoted: overlap-`chi4` can support a qualitative dynamic-heterogeneity
signature, but direct four-point susceptibility and dynamic-length claims remain
blocked until physical time, uncertainty-weighted four-point data, and a
measured dynamic length are present.
The real-data closure priority ledger then ranks the minimum next payloads that
would turn the current GlassBench evidence into stronger falsification tests:
physical-time event clocks and cage-jump segmentation first, post-alpha multi-k
`F_s` targets second, late-NGP recovery third, and direct four-point/dynamic
length data fourth. Every row keeps thermodynamic-transition claims disallowed.
The acquisition-design ledger turns that priority list into three concrete
next panels: a multi-temperature threshold-sweep member-power panel, a
physical-time event-clock inversion panel, and a `tc50` late-recovery
mechanism-power panel. It records four additional independent threshold-sweep
members and two additional hot-ladder lags as the present minimum, rejects
pooled replica-particle substitution, and still keeps thermodynamic claims at
zero.
The acquisition-outcome matrix preregisters how those future panels will be
read: threshold-panel pass/fail can only accept or reject a threshold-robust
event-clock candidate, the event-clock panel is the only branch allowed to
promote a real PE-inversion candidate, and the `tc50` panel selects or rejects
finite exchange against static disorder. Every branch keeps
`thermodynamic_claim_allowed=0`.
The manuscript-claim registry is the final wording lock: current text may claim
real dynamical-signature support before inversion, future PE-inversion wording
requires the physical-time event-clock panel to pass, and event-clock or
late-recovery failures become explicit rejection obligations rather than
ambiguous negative results.
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
docs/langevin-coarse-graining-bridge.md   Langevin/Kramers to renewal-cage bridge
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
figures/renewal_cage_glass_signature_claim_ladder.svg
figures/renewal_cage_glass_phase_diagram.svg
figures/renewal_cage_langevin_bridge.svg
figures/renewal_cage_periodic_softness_gate.svg
figures/renewal_cage_spatial_chi4.svg
figures/renewal_cage_thermodynamic_closure.svg
figures/renewal_cage_thermodynamic_nonidentifiability.svg
figures/renewal_cage_mct_beta_closure.svg
figures/renewal_cage_sota_benchmark_consistency.svg
figures/renewal_cage_sota_claim_alignment.svg
figures/renewal_cage_sota_signed_constraints.svg
figures/renewal_cage_sota_evidence_class.svg
figures/renewal_cage_simultaneous_closure.svg
figures/renewal_cage_microdynamic_prediction_scorecard.svg
figures/renewal_cage_microdynamic_minimality_audit.svg
figures/renewal_cage_sota_experimental_verdict_matrix.svg
figures/renewal_cage_sota_glassbench_real_evidence_claim_synthesis.svg
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
figures/renewal_cage_sota_glassbench_direct_alpha_shape_selection.svg
figures/renewal_cage_sota_glassbench_direct_alpha_multik_shape.svg
figures/renewal_cage_sota_glassbench_direct_alpha_multik_heldout_prediction.svg
figures/renewal_cage_sota_glassbench_direct_alpha_post_window_prediction_targets.svg
figures/renewal_cage_sota_glassbench_direct_alpha_post_window_verdict.svg
figures/renewal_cage_sota_glassbench_direct_alpha_transport.svg
figures/renewal_cage_sota_glassbench_direct_alpha_pe_bound.svg
figures/renewal_cage_sota_glassbench_direct_alpha_displacement_tail_bound.svg
figures/renewal_cage_sota_glassbench_direct_alpha_multilag_crossing_canary.svg
figures/renewal_cage_sota_glassbench_real_threshold_sweep_canary.svg
figures/renewal_cage_sota_glassbench_threshold_sweep_ensemble_verdict.svg
figures/renewal_cage_sota_glassbench_threshold_sweep_payload_contract.svg
figures/renewal_cage_sota_glassbench_threshold_sweep_outcome_matrix.svg
figures/renewal_cage_sota_glassbench_threshold_sweep_decision_power_plan.svg
figures/renewal_cage_sota_glassbench_direct_alpha_event_clock_contract.svg
figures/renewal_cage_sota_glassbench_sparse_lag_event_clock.svg
figures/renewal_cage_sota_glassbench_interval_censored_first_crossing_clock.svg
figures/renewal_cage_sota_glassbench_interval_censored_persistence_fit.svg
figures/renewal_cage_sota_glassbench_waiting_law_selection.svg
figures/renewal_cage_sota_glassbench_finite_exchange_envelope.svg
figures/renewal_cage_sota_glassbench_real_cached_microdynamic_verdict.svg
figures/renewal_cage_sota_glassbench_late_recovery_protocol.svg
figures/renewal_cage_sota_glassbench_late_recovery_ingestion_contract.svg
figures/renewal_cage_sota_glassbench_late_recovery_timecode_target.svg
figures/renewal_cage_sota_glassbench_late_recovery_cache_request_contract.svg
figures/renewal_cage_sota_glassbench_late_recovery_membership_probe_contract.svg
figures/renewal_cage_sota_glassbench_late_recovery_public_timecode_ceiling.svg
figures/renewal_cage_sota_glassbench_censored_window_claim_audit.svg
figures/renewal_cage_sota_glassbench_public_window_verdict.svg
figures/renewal_cage_sota_glassbench_late_recovery_experiment_design.svg
figures/renewal_cage_sota_glassbench_late_recovery_uncertainty_verdict.svg
figures/renewal_cage_sota_glassbench_late_recovery_outcome_matrix.svg
figures/renewal_cage_sota_glassbench_late_recovery_decision_power_plan.svg
figures/renewal_cage_sota_glassbench_timecode_signature_support.svg
figures/renewal_cage_sota_dynamic_signature_alignment.svg
figures/renewal_cage_sota_glassbench_direct_four_point_claim_gate.svg
figures/renewal_cage_sota_glassbench_real_data_closure_priority.svg
figures/renewal_cage_sota_glassbench_real_data_acquisition_design.svg
figures/renewal_cage_sota_glassbench_real_data_acquisition_outcome_matrix.svg
figures/renewal_cage_sota_glassbench_manuscript_claim_registry.svg
figures/renewal_cage_sota_glassbench_cage_jump_proxy_canary.svg
figures/renewal_cage_sota_glassbench_cached_particle_timecode_bridge.svg
figures/renewal_cage_sota_glassbench_multilag_particle_cache_targets.svg
figures/renewal_cage_sota_glassbench_cached_particle_observable_semantics.svg
figures/renewal_cage_sota_glassbench_observable_renewal_canary.svg
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
data/renewal_cage_glass_signature_claim_ladder.csv
data/renewal_cage_glass_phase_diagram.csv
data/renewal_cage_spatial_chi4.csv
data/renewal_cage_spatial_facilitation_inversion.csv
data/renewal_cage_thermodynamic_closure.csv
data/renewal_cage_thermodynamic_nonidentifiability.csv
data/renewal_cage_mct_beta_closure.csv
data/renewal_cage_sota_benchmark_consistency.csv
data/renewal_cage_sota_claim_alignment.csv
data/renewal_cage_sota_signed_constraints.csv
data/renewal_cage_sota_evidence_class.csv
data/renewal_cage_simultaneous_closure.csv
data/renewal_cage_microdynamic_prediction_scorecard.csv
data/renewal_cage_microdynamic_minimality_audit.csv
data/renewal_cage_sota_experimental_verdict_matrix.csv
data/renewal_cage_sota_glassbench_real_evidence_claim_synthesis.csv
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
data/renewal_cage_sota_glassbench_direct_alpha_shape_selection.csv
data/renewal_cage_sota_glassbench_direct_alpha_multik_shape.csv
data/renewal_cage_sota_glassbench_direct_alpha_multik_heldout_prediction.csv
data/renewal_cage_sota_glassbench_direct_alpha_post_window_prediction_targets.csv
data/renewal_cage_sota_glassbench_direct_alpha_post_window_verdict.csv
data/renewal_cage_sota_glassbench_direct_alpha_transport.csv
data/renewal_cage_sota_glassbench_direct_alpha_pe_bound.csv
data/renewal_cage_sota_glassbench_direct_alpha_displacement_tail_bound.csv
data/renewal_cage_sota_glassbench_direct_alpha_multilag_crossing_canary.csv
data/renewal_cage_sota_glassbench_real_threshold_sweep_canary.csv
data/renewal_cage_sota_glassbench_threshold_sweep_ensemble_verdict.csv
data/renewal_cage_sota_glassbench_threshold_sweep_payload_contract.csv
data/renewal_cage_sota_glassbench_threshold_sweep_outcome_matrix.csv
data/renewal_cage_sota_glassbench_threshold_sweep_decision_power_plan.csv
data/renewal_cage_sota_glassbench_direct_alpha_event_clock_contract.csv
data/renewal_cage_sota_glassbench_sparse_lag_event_clock.csv
data/renewal_cage_sota_glassbench_interval_censored_first_crossing_clock.csv
data/renewal_cage_sota_glassbench_interval_censored_persistence_fit.csv
data/renewal_cage_sota_glassbench_waiting_law_selection.csv
data/renewal_cage_sota_glassbench_finite_exchange_envelope.csv
data/renewal_cage_sota_glassbench_real_cached_microdynamic_verdict.csv
data/renewal_cage_sota_glassbench_late_recovery_protocol.csv
data/renewal_cage_sota_glassbench_late_recovery_ingestion_contract.csv
data/renewal_cage_sota_glassbench_late_recovery_timecode_target.csv
data/renewal_cage_sota_glassbench_late_recovery_cache_request_contract.csv
data/renewal_cage_sota_glassbench_late_recovery_membership_probe_contract.csv
data/renewal_cage_sota_glassbench_late_recovery_public_timecode_ceiling.csv
data/renewal_cage_sota_glassbench_censored_window_claim_audit.csv
data/renewal_cage_sota_glassbench_public_window_verdict.csv
data/renewal_cage_sota_glassbench_late_recovery_experiment_design.csv
data/renewal_cage_sota_glassbench_late_recovery_uncertainty_verdict.csv
data/renewal_cage_sota_glassbench_late_recovery_outcome_matrix.csv
data/renewal_cage_sota_glassbench_late_recovery_decision_power_plan.csv
data/renewal_cage_sota_glassbench_timecode_signature_support.csv
data/renewal_cage_sota_dynamic_signature_alignment.csv
data/renewal_cage_sota_glassbench_direct_four_point_claim_gate.csv
data/renewal_cage_sota_glassbench_real_data_closure_priority.csv
data/renewal_cage_sota_glassbench_real_data_acquisition_design.csv
data/renewal_cage_sota_glassbench_real_data_acquisition_outcome_matrix.csv
data/renewal_cage_sota_glassbench_manuscript_claim_registry.csv
data/renewal_cage_sota_glassbench_cage_jump_proxy_canary.csv
data/renewal_cage_sota_glassbench_cached_particle_timecode_bridge.csv
data/renewal_cage_sota_glassbench_multilag_particle_cache_targets.csv
data/renewal_cage_sota_glassbench_multilag_particle_cache_manifest.csv
data/renewal_cage_sota_glassbench_cached_particle_observable_semantics.csv
data/renewal_cage_sota_glassbench_observable_renewal_canary.csv
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
data/renewal_cage_langevin_bridge.csv
data/renewal_cage_periodic_softness_gate.csv
data/renewal_cage_potential_taxonomy.csv
data/renewal_cage_landscape_parameterization.csv
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
paper/figures/renewal_cage_glass_signature_claim_ladder.pdf
paper/figures/renewal_cage_glass_phase_diagram.pdf
paper/figures/renewal_cage_spatial_chi4.pdf
paper/figures/renewal_cage_thermodynamic_closure.pdf
paper/figures/renewal_cage_thermodynamic_nonidentifiability.pdf
paper/figures/renewal_cage_mct_beta_closure.pdf
paper/figures/renewal_cage_sota_benchmark_consistency.pdf
paper/figures/renewal_cage_sota_claim_alignment.pdf
paper/figures/renewal_cage_sota_signed_constraints.pdf
paper/figures/renewal_cage_sota_evidence_class.pdf
paper/figures/renewal_cage_simultaneous_closure.pdf
paper/figures/renewal_cage_microdynamic_prediction_scorecard.pdf
paper/figures/renewal_cage_microdynamic_minimality_audit.pdf
paper/figures/renewal_cage_sota_experimental_verdict_matrix.pdf
paper/figures/renewal_cage_sota_glassbench_real_evidence_claim_synthesis.pdf
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
paper/figures/renewal_cage_sota_glassbench_direct_alpha_shape_selection.pdf
paper/figures/renewal_cage_sota_glassbench_direct_alpha_multik_shape.pdf
paper/figures/renewal_cage_sota_glassbench_direct_alpha_multik_heldout_prediction.pdf
paper/figures/renewal_cage_sota_glassbench_direct_alpha_post_window_prediction_targets.pdf
paper/figures/renewal_cage_sota_glassbench_direct_alpha_post_window_verdict.pdf
paper/figures/renewal_cage_sota_glassbench_direct_alpha_transport.pdf
paper/figures/renewal_cage_sota_glassbench_direct_alpha_pe_bound.pdf
paper/figures/renewal_cage_sota_glassbench_direct_alpha_displacement_tail_bound.pdf
paper/figures/renewal_cage_sota_glassbench_direct_alpha_multilag_crossing_canary.pdf
paper/figures/renewal_cage_sota_glassbench_real_threshold_sweep_canary.pdf
paper/figures/renewal_cage_sota_glassbench_threshold_sweep_ensemble_verdict.pdf
paper/figures/renewal_cage_sota_glassbench_threshold_sweep_payload_contract.pdf
paper/figures/renewal_cage_sota_glassbench_threshold_sweep_outcome_matrix.pdf
paper/figures/renewal_cage_sota_glassbench_threshold_sweep_decision_power_plan.pdf
paper/figures/renewal_cage_sota_glassbench_direct_alpha_event_clock_contract.pdf
paper/figures/renewal_cage_sota_glassbench_sparse_lag_event_clock.pdf
paper/figures/renewal_cage_sota_glassbench_interval_censored_first_crossing_clock.pdf
paper/figures/renewal_cage_sota_glassbench_interval_censored_persistence_fit.pdf
paper/figures/renewal_cage_sota_glassbench_waiting_law_selection.pdf
paper/figures/renewal_cage_sota_glassbench_finite_exchange_envelope.pdf
paper/figures/renewal_cage_sota_glassbench_real_cached_microdynamic_verdict.pdf
paper/figures/renewal_cage_sota_glassbench_late_recovery_protocol.pdf
paper/figures/renewal_cage_sota_glassbench_late_recovery_ingestion_contract.pdf
paper/figures/renewal_cage_sota_glassbench_late_recovery_timecode_target.pdf
paper/figures/renewal_cage_sota_glassbench_late_recovery_cache_request_contract.pdf
paper/figures/renewal_cage_sota_glassbench_late_recovery_membership_probe_contract.pdf
paper/figures/renewal_cage_sota_glassbench_late_recovery_public_timecode_ceiling.pdf
paper/figures/renewal_cage_sota_glassbench_censored_window_claim_audit.pdf
paper/figures/renewal_cage_sota_glassbench_public_window_verdict.pdf
paper/figures/renewal_cage_sota_glassbench_late_recovery_experiment_design.pdf
paper/figures/renewal_cage_sota_glassbench_late_recovery_uncertainty_verdict.pdf
paper/figures/renewal_cage_sota_glassbench_late_recovery_outcome_matrix.pdf
paper/figures/renewal_cage_sota_glassbench_late_recovery_decision_power_plan.pdf
paper/figures/renewal_cage_sota_glassbench_timecode_signature_support.pdf
paper/figures/renewal_cage_sota_dynamic_signature_alignment.pdf
paper/figures/renewal_cage_sota_glassbench_direct_four_point_claim_gate.pdf
paper/figures/renewal_cage_sota_glassbench_real_data_closure_priority.pdf
paper/figures/renewal_cage_sota_glassbench_real_data_acquisition_design.pdf
paper/figures/renewal_cage_sota_glassbench_real_data_acquisition_outcome_matrix.pdf
paper/figures/renewal_cage_sota_glassbench_manuscript_claim_registry.pdf
paper/figures/renewal_cage_sota_glassbench_cage_jump_proxy_canary.pdf
paper/figures/renewal_cage_sota_glassbench_cached_particle_timecode_bridge.pdf
paper/figures/renewal_cage_sota_glassbench_multilag_particle_cache_targets.pdf
paper/figures/renewal_cage_sota_glassbench_cached_particle_observable_semantics.pdf
paper/figures/renewal_cage_sota_glassbench_observable_renewal_canary.pdf
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
