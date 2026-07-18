# Microscopic Activated Cage-Jump Geometry Diagnostic

## Question and boundary

The gamma variance-mixture diagnostic uses held-out MSD and NGP to predict the
full wave-number dependence of `F_s(k,t)`.  It succeeds at `k=2` and `k=4` but
overpredicts `F_s(k=7.25)` at T=0.45.  This step tests whether replacing the
gamma shape by the measured geometry of calibration cage jumps removes that
residual.

The held-out MSD and NGP are diagnostic inputs, not a blind prediction.  The
jump vectors, their moments, and their characteristic function come only from
the calibration half of each trajectory.  No held-out event or cage-residual
path enters the geometry table.

## Compound-Poisson derivation

Write one displacement as an independent Gaussian cage contribution plus a
Poisson number `N` of identically distributed cage jumps `J`:

```text
Delta R = Delta Y + sum_(j=1)^N J_j,
N ~ Poisson(n).
```

For the calibration jump-vector distribution define

```text
m2       = <|J|^2>,
m4x      = <J_x^4> averaged over x, y, z,
phi_J(k) = <cos(k J_x)> averaged over events and x, y, z.
```

The Gaussian cage has zero fourth cumulant.  The compound-Poisson fourth
cumulant is `n m4x`.  Isotropy relates the observed three-dimensional NGP to
the one-coordinate fourth cumulant:

```text
kappa_4,x = alpha_2 M^2 / 3 = n m4x,
n = alpha_2 M^2 / (3 m4x),
a = (M - n m2) / 6,
F_s(k) = exp[-k^2 a + n(phi_J(k)-1)].
```

Here `a` is the variance coordinate used in `exp(-k^2 a)`.  A row is physical
only when both `n>=0` and `a>=0`.

For the fixed-length periodic-potential null, `|J|=ell` gives

```text
m2 = ell^2,
m4x = ell^4/5,
phi_J(k) = sinc(k ell).
```

This is the high-barrier basin-to-basin limit expected from a single-spacing
periodic cage potential.  It is a useful null, but it is not the full
many-particle landscape.

## Calibration geometry

The extractor reads only the first `calibration_time+1` frames from each raw
LAMMPS trajectory.  It uses the existing Debye-Waller finite-duration event
definition and replicate-specific calibration thresholds.

At T=0.45:

| replicate | events | `<|J|^2>` | `<|J|^4>` | `<J_x^4>` | `phi_J(7.25)` |
|---:|---:|---:|---:|---:|---:|
| 1 | 82,267 | 0.066858 | 0.017188 | 0.003432 | 0.642299 |
| 2 | 82,671 | 0.066963 | 0.017392 | 0.003487 | 0.641191 |
| 3 | 80,085 | 0.065322 | 0.017011 | 0.003362 | 0.647912 |

The event counts and second moments exactly reproduce the previously committed
calibration-event summaries.  The measured component fourth moment is roughly
four times the fixed-length value `m2^2/5`, so the real jump-length law is much
broader than the single-spacing periodic null.

## Real-trajectory verdict

The broader empirical jump law increases T=0.45 physical support from `3/21`
fixed-length rows to `8/21` rows.  This remains below the frozen 80% support
requirement.  All three replicates have at least one supported row, so the
failure is support coverage rather than a missing replicate.

On the eight supported T=0.45 rows:

| wave number | maximum absolute `F_s` error | tolerance units |
|---:|---:|---:|
| 2 | 0.0000737 | 0.00246 |
| 4 | 0.0032261 | 0.10754 |
| 7.25 | 0.0416405 | 1.38802 |

Thus the empirical jump law closes low/intermediate wave numbers but still
overpredicts the worst high-k value.  The failure is not caused solely by
approximating a broad jump distribution with one fixed cage spacing.

T=0.58 has `25/25` empirical-support rows, but its maximum `k=7.25` error is
`0.0434154`, also above tolerance.  More importantly, its source stationarity
gate fails, so it remains a canary and cannot support a cooling comparison.

## Physical interpretation

The inversion assigns the entire observed fourth cumulant to independent
Poisson jump marks.  At low temperature this often requires so many jumps that
`n m2 > M`, making the inferred cage variance negative.  The measured NGP must
therefore contain contributions absent from this closure, including one or
more of:

```text
Var(N) != <N>                 count overdispersion / renewal memory,
Cov(J_i,J_j) != 0             recoil or path correlations,
P(J | N,cage) != P(J)         cage-jump or environment coupling,
non-additive cage/event terms finite-duration rearrangement geometry.
```

This result blocks a direct move to a single periodic potential.  A
distributed-basin potential would reproduce the measured one-jump law, but the
one-jump law is already insufficient.  The next microscopic closure must first
derive the joint count-jump cumulants from precursor-gated or environment-
conditioned Langevin dynamics, then predict multi-k scattering without using
held-out `F_s` as an input.

## Claim flags

The empirical geometry quotient is not supported at T=0.45, and even a passing
quotient would not identify a unique potential.  The machine-readable flags
therefore remain:

```text
blind_prediction_claim_allowed = 0
finite_exchange_resolved = 0
static_environment_resolved = 0
spatial_facilitation_resolved = 0
activated_cage_geometry_resolved = 0
microdynamic_closure_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

