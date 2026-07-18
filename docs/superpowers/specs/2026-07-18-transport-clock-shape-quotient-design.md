# Transport-Clock / Shape Quotient Design

## Scope

This is a post-run diagnostic of the committed segment-splice full-path rows.
It asks whether calibration-to-heldout error is primarily a change of transport
clock or a change of displacement shape. It performs no trajectory simulation,
fits no macro observable, and does not revise the segment-splice verdict.

The decomposition is

```text
X(t) = Y(Theta(t))
```

where MSD is used only as an observed coordinate for `Theta`. Because heldout
MSD enters the quotient, this is a diagnostic and never a blind prediction.

## Frozen inputs

Consume the committed T045 and T058 segment-splice row tables, their
stationarity tables, and the local ensemble manifests. Select the within-model
full-path controls (`L=250` at T045 and `L=37` at T058), after verifying that
the cross-model full-path observables agree within `1e-12`. Require the exact
replicate and lag grids, finite values, no duplicate cells, zero heldout leakage
and macro-fit counts, and closed source claim flags.

Both manifests must remain explicit that
`independently_prepared_parent_samples=false`. T058 is a canary because its
ensemble stationarity gate fails; it cannot establish a cooling trend.

## Quotient

For each temperature, replicate, and observable
`O in {NGP, Fs(k=2), Fs(k=4), Fs(k=7.25)}`:

1. form the strictly increasing calibration curve `O_cal(MSD_cal)`;
2. evaluate it by piecewise-linear interpolation at each heldout MSD;
3. prohibit extrapolation;
4. retain only supported lags for matched comparisons.

Each replicate must retain at least 80% of its frozen lags and at least one
supported anchor-alpha point with `0.1 <= observed Fs(k=4) <= 0.9`.
Comparisons use the original absolute tolerances: `0.30` for NGP and `0.03`
for every scattering function. Same-time and MSD-matched errors must be
reported on the identical supported rows; full-grid same-time errors are
reported separately.

## Classification

T045 may set `clock_shape_separation_supported_exploratory=1` only when:

- support and anchor-alpha requirements pass for every replicate;
- same-time `Fs(k=2)` and `Fs(k=4)` fail on matched support;
- their MSD-matched residuals pass for every replicate; and
- either MSD-matched NGP or `Fs(k=7.25)` still fails.

This result rejects a clock-only closure but does not identify the residual
mechanism. T058 is reported as `high_temperature_canary_only=1`; its matched
pass cannot support a cooling claim because stationarity and independent-parent
controls are unresolved.

Keep all of these zero:

```text
blind_prediction_claim_allowed
clock_only_closure_allowed
cooling_enhanced_shape_memory_claim_allowed
finite_exchange_resolved
static_environment_resolved
spatial_facilitation_resolved
microdynamic_closure_claim_allowed
thermodynamic_claim_allowed
```

The confirmatory next step is new independently prepared parent trajectories
with replicate-resolved early/late/heldout stationarity, followed by a blind
event-clock prediction of MSD before applying this shape quotient.

## Outputs

- `scripts/summarize_ka_transport_clock_shape_quotient.py`
- `tests/test_ka_transport_clock_shape_quotient.py`
- `data/renewal_cage_ka_transport_clock_shape_quotient_rows.csv`
- `data/renewal_cage_ka_transport_clock_shape_quotient_gate.csv`
- `figures/renewal_cage_ka_transport_clock_shape_quotient.svg`
- one arXiv artifact recomputation test and a concise README result entry

