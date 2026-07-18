# KA Shape-Mechanism Selection

## Question and common grid

The count-overdispersed cage-jump quotient restores the T=0.45 moment budget
but leaves a maximum `Fs(k=7.25)` error above the frozen `0.03` tolerance. This
test asks whether calibration-measured successive jump directions close that
residual, or whether only a marginal variance-mixture shape remains viable.

Every model is evaluated on the same 18 T=0.45 replicate-lag cells and
`k=(2,4,7.25)`. Held-out MSD and NGP remain diagnostic inputs, and no model has
a fitted macro parameter. The three restart labels descend from one parent
sample, so this is an exploratory common-grid diagnostic rather than an
independent-parent confirmation.

## Event-path closures

Let the inferred count mean be `m`, the count Fano factor be `F`, and let
`psi_n(k)` be the calibration characteristic function of the net displacement
over `n` consecutive cage jumps. The negative-binomial count probability
generating function is

```text
G(z) = [1 + (F-1)(1-z)]^[-m/(F-1)],
```

with the Poisson limit `G(z)=exp[m(z-1)]` at `F=1`. Four factorized event-path
closures are tested before multiplication by the same independent Gaussian
cage factor `exp(-k^2 a_c)`.

The independent-jump closure uses `G(psi_1)`. The disjoint-pair closure uses

```text
psi_(2j)   = psi_2^j,
psi_(2j+1) = psi_1 psi_2^j,
```

which gives the exact even/odd PGF combination

```text
0.5 [(1+psi_1/s) G(s) + (1-psi_1/s) G(-s)],  s=sqrt(psi_2).
```

The Pair-eigenmode closure uses the leading two-step transfer estimate
`lambda=psi_2/psi_1` and

```text
psi_n = psi_1 lambda^(n-1),  n >= 1,
E[psi_N] = P(N=0) + (psi_1/lambda)[G(lambda)-P(N=0)].
```

Finally, the empirical-path closure evaluates `sum_n P(N=n) psi_n` using every
committed calibration kernel through `n=60` at T=0.45. The maximum omitted
count probability is `2.97e-4`, below the frozen `0.01` tail tolerance.

## Result

Maximum absolute `Fs` errors divided by the frozen `0.03` tolerance are:

| closure | k=2 | k=4 | k=7.25 | all-k verdict |
|---|---:|---:|---:|---|
| independent jump | 0.080 | 0.697 | 1.370 | fail |
| disjoint pair | 1.040 | 1.507 | 1.103 | fail |
| pair eigenmode | 2.063 | 3.240 | 2.573 | fail |
| empirical path | 2.447 | 4.022 | 3.218 | fail |
| gamma variance mixture | 0.062 | 0.393 | 0.466 | pass |
| inverse-Gaussian variance mixture | 0.059 | 0.375 | 0.423 | pass |

The mild disjoint-pair correction lowers the worst high-k residual, but it
simultaneously pushes the lower-k errors outside tolerance. More complete use
of the measured ordered path increasingly overcorrects the scattering shape.
Thus the tested factorized event-path closures fail on the common grid, while
two distinct positive variance-mixture resummations survive.

This selects a shape class, not a microscopic mechanism. It does not identify
a unique mixing family, static disorder, finite exchange, cage-jump coupling,
or spatial facilitation. The next discriminating model must therefore test a
nonfactorized coupling between cage variance, jump geometry, and a finite-lived
environmental state. T=0.58 remains a stationarity-unresolved canary.

```text
blind_prediction_claim_allowed = 0
unique_variance_mixture_family_selected = 0
static_environment_resolved = 0
finite_exchange_resolved = 0
cage_jump_coupling_identified = 0
factorized_event_path_family_excluded_beyond_tested_closures = 0
microdynamic_closure_claim_allowed = 0
spatial_facilitation_claim_allowed = 0
thermodynamic_claim_allowed = 0
```
