# Cage-Anchor Return Gate Design

## Purpose

Identify the minimum missing state variable behind the low-temperature failure of
the existing path-spectrum null. The gate distinguishes a local one-step recoil
law from a reversible cage topology that remembers an earlier cage center. It is
a mechanism-identification result, not yet a fitted semi-Markov closure.

The allowed claim remains restricted to dynamical glass signatures.
`microdynamic_closure_claim_allowed=0`,
`spatial_facilitation_claim_allowed=0`, and
`thermodynamic_claim_allowed=0` are mandatory.

## Frozen data protocol

- Use the existing independently restarted Kob-Andersen ensembles at `T=0.45`
  and `T=0.58`.
- Use calibration times `5000` and `750`, respectively, with no held-out event,
  displacement, MSD, NGP, or scattering value entering a calibration statistic.
- Reuse the calibration Debye-Waller factors and the existing non-recrossing
  cage-jump extractor with fluctuation half-window `5`.
- Use type-A particles, physical lag time, block size `20`, wave numbers
  `k={2,4,7.25}`, and the existing held-out factorization tables.
- Aggregate macro predictions and observations across independent replicates
  before the primary curve decision. Keep replicate-level quality and mechanism
  rows.

This protocol was frozen after an exploratory audit. Its outputs are therefore a
reproducible identification result, not a prospectively preregistered discovery
test.

## Cage-center return statistic

For consecutive non-recrossing jumps of one particle, let `v_n` and `v_{n+1}`
be the measured cage-center jump vectors and let `a_DW` be the square root of the
calibration Debye-Waller factor. Define a return at radius scale `s` by

`|v_n + v_{n+1}| <= s a_DW`.

Use `s={0.5,1.0,1.5}`, with `s=1.0` primary. The length-preserving isotropic null
keeps both observed jump lengths and randomizes only their relative direction.
In three dimensions its return probability is evaluated analytically from the
uniform distribution of `cos(theta)`.

For each replicate and scale report the observed return fraction, analytic null
fraction, excess ratio, return-run mean, geometric-run mean `1/(1-p_return)`,
and physical-time run-duration quantiles. A long-lived binary "return-prone"
state is not selected when the observed run mean stays within 15 percent of the
geometric value. This does not remove the cage anchor: repeated geometric
returns can still preserve a finite-lived cage topology.

The cooling-induced cage-anchor signal is ready only if, at every radius scale,
the minimum low-temperature return fraction exceeds the maximum high-temperature
return fraction. At the primary radius, every low-temperature replicate must
also exceed its isotropic null by a factor of at least `1.35`.

## One-step recoil Markov null

Construct calibration-only block paths separately for every particle. Bin each
particle's source block radii into eight equal-count bins. Starting from an
observed block vector, generate every next vector by sampling an empirical
transition from the current radius bin, retaining the sampled target radius and
relative cosine while drawing an isotropic azimuth. This preserves particle
identity, radial disorder, and the complete one-step recoil law in expectation,
but removes cage-anchor closure and all higher-order path order.

Use 16 deterministic realizations per replicate. The null is valid only if all
replicates satisfy:

- relative radial mean and standard-deviation errors at most `0.02`;
- lag-one cosine mean error at most `0.02`;
- maximum error over cosine quantiles `{0.1,0.25,0.5,0.75,0.9}` at most `0.03`;
- normalized lag-one dot-correlation error at most `0.02`;
- Monte Carlo standard errors at most `0.01` for relative MSD, `0.03` for NGP,
  and `0.003` for every scattering function.

Score the same held-out tolerances as the preceding gates: maximum ensemble
relative MSD error `0.10`, absolute NGP error `0.30`, and absolute multi-k
scattering error `0.03`.

Select `cage_anchor_memory_required=1` only if the recoil null passes all quality
and precision gates at both temperatures, closes all three held-out curve classes
at `T=0.58`, and fails both NGP and multi-k scattering at `T=0.45`, while the
ordered calibration path remains the established low-temperature upper bound.

## Interpretation and failure modes

Passing the gate identifies a topological memory of a previous cage center that
is absent from a one-step vector Markov law. It supports an anchor-aware
reversible-cage semi-Markov model as the next candidate. It does not prove that
candidate is sufficient, uniquely exclude static environment effects, or exclude
cross-particle facilitation.

If cage returns do not separate by temperature, reject the reversible-anchor
interpretation. If the recoil null fails its quality gates, report the mechanism
as unresolved. If it passes quality but fails at both temperatures, do not claim a
cooling crossover. If it closes low temperature, stop without adding a cage-anchor
state.

## Outputs

- `scripts/analyze_ka_cage_anchor_returns.py`
- `scripts/analyze_ka_recoil_markov_transfer.py`
- `scripts/summarize_ka_cage_anchor_gate.py`
- per-temperature return, recoil-null quality, prediction, summary, and verdict
  CSV files under `data/`
- `data/renewal_cage_ka_cage_anchor_gate.csv`
- `figures/renewal_cage_ka_cage_anchor_gate.svg`

Tests cover analytic null probabilities, return-run accounting, deterministic
Markov generation, information-preservation quality, calibration/held-out
provenance, replicate-first aggregation, decision truth tables, artifact schema,
and all three claim boundaries.
