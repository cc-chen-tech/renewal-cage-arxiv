# Task 4 Report: Two-Temperature Cage-Anchor Gate

## Status and Scope

Implemented Task 4 on `codex/cage-anchor-analysis` without modifying `src/`,
the manuscript, README, package/build scripts, prior artifacts, or unrelated
files. The combined gate selects `cage_anchor_memory_required=1` only after
every frozen return, recoil-quality, precision, crossover, and ordered-path
condition passes. All three broader claim flags remain zero.

Implementation commit:

`0d4aadd identify cooling induced cage anchor memory`

## TDD Evidence

### Initial RED

Decision and schema tests were written before the summarizer existed.

Command:

```bash
/usr/bin/time -p python3 -m unittest tests.test_ka_replicates.KACageAnchorGateTests.test_combined_gate_selects_anchor_only_for_every_frozen_condition tests.test_ka_replicates.KACageAnchorGateTests.test_combined_gate_rejects_schema_and_frozen_protocol_mismatches -v
```

Result: exit `1`; 2 tests errored in `0.001s` (`real 0.27s`). Both errors were
the expected `FileNotFoundError` for
`scripts/summarize_ka_cage_anchor_gate.py`.

The package test was also written before artifact generation.

```bash
/usr/bin/time -p python3 -m unittest tests.test_arxiv_package.ArxivPackageTests.test_cage_anchor_gate_artifacts_enforce_frozen_crossover -v
```

Result: exit `1`; 1 test errored in `0.001s` (`real 0.40s`) on the expected
missing `data/renewal_cage_ka_cage_anchor_returns_rows.csv`.

### Initial GREEN

After implementing the strict combined classifier:

```bash
/usr/bin/time -p python3 -m unittest tests.test_ka_replicates.KACageAnchorGateTests.test_combined_gate_selects_anchor_only_for_every_frozen_condition tests.test_ka_replicates.KACageAnchorGateTests.test_combined_gate_rejects_schema_and_frozen_protocol_mismatches -v
```

Result: exit `0`; 2 tests passed in `0.004s` (`real 0.14s`). The decision table
independently disables return separation, primary null excess, low quality,
high precision, high closure, low NGP failure, low Fs failure, and the ordered
path bound; every mutation forces `cage_anchor_memory_required=0`.

### Self-Review RED/GREEN

Self-review found that return-row claim flags were not validated at input. A
regression assertion was added first.

```bash
/usr/bin/time -p python3 -m unittest tests.test_ka_replicates.KACageAnchorGateTests.test_combined_gate_rejects_schema_and_frozen_protocol_mismatches -v
```

RED result: exit `1`; 1 failure in `0.002s` (`real 0.08s`), specifically
`ValueError not raised` for `thermodynamic_claim_allowed=1` in a return row.

After requiring all three claim flags to be present and zero on return and
recoil inputs, the same command exited `0`; 1 test passed in `0.003s`
(`real 0.09s`).

Package self-review also replaced sole reliance on `quality_pass` with direct
assertions for all five named quality thresholds and all frozen controls stored
in the final gate CSV.

## Exact Real Commands and Runtimes

### Cage returns

```bash
/usr/bin/time -p python3 scripts/analyze_ka_cage_anchor_returns.py tmp/ka_replicates/T045 tmp/ka_replicates/T058 --low-calibration-time 5000 --high-calibration-time 750 --fluctuation-half-window 5 --output-prefix data/renewal_cage_ka_cage_anchor_returns
```

Exit `0`; `real 63.42s`, `user 53.72s`, `sys 7.07s`.

### T=0.45 recoil transfer

```bash
/usr/bin/time -p python3 scripts/analyze_ka_recoil_markov_transfer.py tmp/ka_replicates/T045 --calibration-time 5000 --heldout-factorization data/renewal_cage_ka_replicates_T045_event_oracle_factorization_rows.csv --block-size 20 --surrogate-realizations 16 --seed 781031 --output-prefix data/renewal_cage_ka_replicates_T045_recoil_markov
```

Exit `0`; `real 84.38s`, `user 70.01s`, `sys 11.67s`.

### T=0.58 recoil transfer

```bash
/usr/bin/time -p python3 scripts/analyze_ka_recoil_markov_transfer.py tmp/ka_replicates/T058 --calibration-time 750 --heldout-factorization data/renewal_cage_ka_replicates_T058_event_oracle_factorization_rows.csv --block-size 20 --surrogate-realizations 16 --seed 781031 --output-prefix data/renewal_cage_ka_replicates_T058_recoil_markov
```

Exit `0`; `real 13.78s`, `user 12.51s`, `sys 1.16s`.

### Combined gate and figure

```bash
/usr/bin/time -p python3 scripts/summarize_ka_cage_anchor_gate.py data/renewal_cage_ka_cage_anchor_returns_rows.csv data/renewal_cage_ka_replicates_T045_recoil_markov_verdict.csv data/renewal_cage_ka_replicates_T058_recoil_markov_verdict.csv data/renewal_cage_ka_replicates_T045_empirical_path_verdict.csv --output-csv data/renewal_cage_ka_cage_anchor_gate.csv --output-svg figures/renewal_cage_ka_cage_anchor_gate.svg
```

Exit `0`; `real 0.04s`, `user 0.02s`, `sys 0.00s`.

The fourth positional input is an existing Task 2 ordered empirical-path
verdict, not a new scientific parameter. The summarizer requires its contiguous
path row to close the frozen held-out tolerances with three low-temperature
replicates before marking the ordered calibration path as the established upper
bound.

## Artifact List

- `data/renewal_cage_ka_cage_anchor_returns_rows.csv` (24 rows)
- `data/renewal_cage_ka_cage_anchor_returns_verdict.csv` (1 row)
- `data/renewal_cage_ka_replicates_T045_recoil_markov_rows.csv` (336 rows)
- `data/renewal_cage_ka_replicates_T045_recoil_markov_quality.csv` (48 rows)
- `data/renewal_cage_ka_replicates_T045_recoil_markov_summary.csv` (7 rows)
- `data/renewal_cage_ka_replicates_T045_recoil_markov_verdict.csv` (1 row)
- `data/renewal_cage_ka_replicates_T058_recoil_markov_rows.csv` (400 rows)
- `data/renewal_cage_ka_replicates_T058_recoil_markov_quality.csv` (80 rows)
- `data/renewal_cage_ka_replicates_T058_recoil_markov_summary.csv` (5 rows)
- `data/renewal_cage_ka_replicates_T058_recoil_markov_verdict.csv` (1 row)
- `data/renewal_cage_ka_cage_anchor_gate.csv` (1 row)
- `figures/renewal_cage_ka_cage_anchor_gate.svg`

No visual inspection is claimed; Task 4 validation parsed the SVG as XML and
checked required text and finite serialized values only. Visual review belongs
to Task 5.

## Frozen-Condition Audit

### Protocol and completeness

- Temperatures/calibration times: `0.45/5000` and `0.58/750`.
- Independent replicate counts: `3` and `5`.
- Return radius scales: exactly `0.5, 1.0, 1.5`; primary `1.0`.
- Primary low-temperature return/null threshold: `1.35`.
- Recoil block size: `20`; radial bins: `8`; realizations: `16` per replicate.
- Wave numbers in prediction tables: exactly `2, 4, 7.25`.
- Curve tolerances: MSD `0.10`, NGP `0.30`, multi-k Fs `0.03`.
- MC SE tolerances: relative MSD `0.01`, NGP `0.03`, every Fs `0.003`.
- Quality rows: `3*16=48` at `T=0.45`, `5*16=80` at `T=0.58`.
- Replicate-first summary rows: 7 low-temperature lags and 5 high-temperature
  lags, each with `replicate_first_aggregation=1`.

### Return separation

Actual `(minimum low, maximum high)` return fractions were:

- radius `0.5`: `(0.1564038645, 0.0100221309)`;
- radius `1.0`: `(0.4290533418, 0.0594372431)`;
- radius `1.5`: `(0.6374099292, 0.1513323278)`.

All three strict separations pass. The minimum low-temperature primary
return/null ratio is `1.4977206337`, above `1.35`.

### Recoil information-preservation quality

Actual maxima at `(T=0.45, T=0.58)` were:

- radial mean relative error: `(0.0016240112, 0.0067065759)` <= `0.02`;
- radial standard-deviation relative error: `(0.0042475639, 0.0079854420)`
  <= `0.02`;
- lag-one cosine mean absolute error: `(0.0018545721, 0.0041753078)` <= `0.02`;
- cosine-quantile maximum absolute error: `(0.0043482760, 0.0077024019)`
  <= `0.03`;
- normalized lag-one dot-correlation absolute error:
  `(0.0047038315, 0.0043765533)` <= `0.02`.

Both quality and realization-completeness gates equal `1`.

### Precision and held-out crossover

Actual maximum MC errors at `(T=0.45, T=0.58)` were:

- relative MSD SE: `(0.0091058594, 0.0017987710)` <= `0.01`;
- NGP SE: `(0.0024672108, 0.0031686635)` <= `0.03`;
- Fs SE: `(0.0007276277, 0.0005997721)` <= `0.003`.

At `T=0.58`, actual maximum held-out errors are MSD `0.0363088023`, NGP
`0.0962128051`, and Fs `0.0169884008`; all close their frozen tolerances and
`curve_transfer_pass=1`.

At `T=0.45`, maximum NGP error is `1.9760066636` and maximum Fs error is
`0.5366481932`; both required failures are present and
`curve_transfer_pass=0`. The existing ordered contiguous path upper-bound
condition is `1`.

Therefore the combined CSV records `cage_anchor_memory_required=1` and
`mechanism_state=cage_anchor_memory_required`. It retains:

- `microdynamic_closure_claim_allowed=0`;
- `spatial_facilitation_claim_allowed=0`;
- `thermodynamic_claim_allowed=0`.

## GREEN Verification

Focused gates after final self-review:

- `python3 -m unittest tests.test_ka_replicates.KACageAnchorGateTests -q`:
  10 tests passed in `0.053s` (`real 0.16s`).
- `python3 -m unittest tests.test_arxiv_package.ArxivPackageTests.test_cage_anchor_gate_artifacts_enforce_frozen_crossover -v`:
  1 test passed in `0.007s` (`real 0.15s`).

Final full modules from the final source state:

```bash
/usr/bin/time -p python3 -m unittest tests.test_ka_replicates -q
```

Exit `0`; 123 tests passed in `0.503s` (`real 0.57s`).

```bash
/usr/bin/time -p python3 -m unittest tests.test_arxiv_package -q
```

Exit `0`; 188 tests passed in `3.774s` (`real 3.91s`).

Static checks:

```bash
/usr/bin/time -p python3 -m py_compile scripts/summarize_ka_cage_anchor_gate.py tests/test_ka_replicates.py tests/test_arxiv_package.py
git diff --check
git diff --cached --check
```

All exited `0`; `py_compile` took `real 0.06s`; both diff checks produced no
output. A programmatic artifact audit checked 11 CSV files and 904 rows: every
numeric field was finite, all claim flags were zero, the gate selected anchor
memory, and the SVG parsed as valid XML.

## Self-Review

- Confirmed the staged implementation commit contains exactly the owned
  summarizer, two test files, 11 Task 4 CSVs, and one final SVG.
- Confirmed no existing analyzer, `src/`, manuscript, README, package/build
  script, or prior artifact changed.
- The selector independently recomputes return separation and primary null
  excess instead of trusting the return verdict.
- The selector independently checks recoil protocol metadata, replicate counts,
  quality/completeness flags, MC maxima, high-temperature curve errors, both
  required low-temperature failures, ordered-path evidence, and input/output
  claim boundaries.
- Package tests inspect replicate counts, every named quality threshold, frozen
  controls, curve thresholds, mechanism selection, all claim boundaries, and
  non-NaN/non-infinite SVG serialization.
- No external reviewer subagent was available in this environment; the diff,
  ownership list, generated schemas, and numerical conditions were reviewed
  directly before the implementation commit.

## Concerns

The low-temperature recoil null also fails MSD strongly: maximum relative MSD
error is `3.3981894621`, although the frozen selector only requires
low-temperature failures in NGP and multi-k Fs. This does not invalidate the frozen
decision table, but Task 5 interpretation should avoid implying that the missing
cage anchor is proven to be the unique cause of every low-temperature curve
error. Static-environment and cross-particle facilitation alternatives remain
unexcluded, exactly as stated in the design.
