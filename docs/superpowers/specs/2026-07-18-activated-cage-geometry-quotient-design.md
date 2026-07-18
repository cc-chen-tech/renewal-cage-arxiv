# Activated Cage Geometry Quotient Design

**Status:** Approved under the user's standing instruction to continue microscopic exploration without repeated approval prompts.

## Objective

Test whether the low-temperature high-wave-number residual left by the gamma
total-variance mixture is explained by the measured geometry of activated cage
jumps.  This is a diagnostic quotient, not a blind trajectory prediction and
not yet a derivation of the jump distribution from a many-particle potential.

## Scientific Question

The gamma variance mixture fixes the integrated-variance distribution using
held-out MSD and NGP, but overpredicts the T=0.45 self-intermediate scattering
function at `k=7.25`.  The sign means that the gamma law leaves too much weight
near zero displacement variance.  A Kramers cage escape instead produces a
finite displacement whose characteristic function need not be gamma-shaped.

The primary question is therefore:

> After fixing the jump-vector law from calibration cage escapes, do held-out
> MSD and NGP determine a nonnegative cage variance and event count that predict
> held-out multi-k `F_s(k,t)` within the frozen absolute tolerance 0.03?

## Compared Closures

### Fixed-length periodic-potential null

For isotropic jumps of fixed length `ell`, a Poisson event count with mean `n`,
and an independent Gaussian cage contribution `a`,

```text
F_s(k) = exp[-k^2 a + n(sinc(k ell) - 1)]
M = 6 a + n ell^2
alpha_2 M^2 / 3 = n ell^4 / 5.
```

Using calibration `ell^2 = <|J|^2>` gives a nonnegative `a` for only 3 of the
21 T=0.45 replicate-lag rows.  This frozen-step null is retained as a negative
control and is not promoted to a candidate closure.

### Empirical activated-jump geometry quotient

For calibration jump vectors `J`, define

```text
m2       = <|J|^2>
m4x      = <J_x^4> averaged over x, y, z
phi_J(k) = <cos(k J_x)> averaged over events and x, y, z.
```

For a compound-Poisson number of independent jumps plus an independent
Gaussian cage displacement, the one-coordinate fourth cumulant gives

```text
n(t) = alpha_2(t) M(t)^2 / [3 m4x]
a(t) = [M(t) - n(t) m2] / 6
F_s(k,t) = exp[-k^2 a(t) + n(t)(phi_J(k) - 1)].
```

This formulation uses the empirical component characteristic directly.  It
does not impose a fixed jump length or a Gaussian jump law.  A row is supported
only when `n` and `a` are finite and nonnegative.

## Frozen Inputs

- Primary temperature: T=0.45.
- Canary temperature: T=0.58 only; its source stationarity control fails, so it
  cannot support a cooling comparison.
- Calibration trajectories: the first 5000 time units of each registered KA
  replicate in the existing ensemble manifests.
- Event definition: existing Debye-Waller finite-duration cage jumps with
  `fluctuation_half_window=5` and the existing replicate-specific calibration
  Debye-Waller thresholds.
- Diagnostic held-out inputs and targets: the existing gamma variance-mixture
  rows, which contain replicate-first held-out `M`, `alpha_2`, and multi-k
  `F_s` at the frozen lag grids.
- Wave numbers: `k={2,4,7.25}`.
- No held-out event, jump-vector, or cage-residual data may enter calibration.

## Gates And Outputs

The summarizer writes row-level and temperature-level CSV files plus one SVG.
It reports support coverage, maximum normalized errors, signed residuals, and
the fixed-length negative-control support.

The exploratory empirical-geometry closure is supported only if:

1. T=0.45 provenance and source stationarity controls pass.
2. At least 80% of the T=0.45 replicate-lag rows have nonnegative `n` and `a`.
3. Every supported T=0.45 wave-number curve has maximum absolute error at most
   0.03.
4. Every one of the three T=0.45 replicates contains at least one supported row.

The fixed 0.03 curve tolerance, trajectory inputs, event definition, lag grid,
wave-number grid, and temperature roles are not tuned after seeing the result.

## Claim Boundary

Even if the quotient passes, it establishes only that calibration jump
geometry is sufficient to close this held-out shape diagnostic conditional on
held-out MSD and NGP.  It does not establish a blind prediction, a unique
potential, a finite-exchange mechanism, a static-environment mechanism, spatial
facilitation, or a many-particle microscopic closure.

The following flags remain zero in all outcomes:

```text
blind_prediction_claim_allowed
finite_exchange_resolved
static_environment_resolved
spatial_facilitation_resolved
activated_cage_geometry_resolved
microdynamic_closure_claim_allowed
thermodynamic_claim_allowed
```

A separate field,
`empirical_activated_jump_geometry_supported_exploratory`, may become one only
when all primary gates pass.  T=0.58 remains a canary and cannot set this field.

## Next-Stage Decision

If the empirical geometry quotient passes, the next phase will construct a
continuous overdamped Langevin landscape whose Kramers exits reproduce the
calibration jump-vector law, then compare its basin-to-basin characteristic
function with the compound-Poisson closure.  Because the fixed-length null is
already incompatible with most rows, that landscape must include distributed
basin spacing or multi-channel exits rather than a single one-dimensional
cosine potential.

If the quotient fails, no additional potential parameters are added in this
phase.  The result instead localizes the missing physics to correlated events,
non-Poisson count cumulants, cage-jump dependence, or richer path/environment
state.

## Verification

- Formula tests cover deterministic synthetic jump laws and unsupported
  negative-cage cases.
- Extraction tests prove that only calibration trajectory prefixes are read.
- Artifact tests recompute every committed CSV and the SVG byte-for-byte under
  Python 3.12.
- The full repository test suite, result generator, arXiv package builder, and
  `git diff --check` must pass before publication.
