# KA interval-conditioned event transfer

## Question

The earlier calibration-only event-clock closure used a count law and a global
`n`-jump propagator. It failed held-out MSD, NGP, and multi-`k` `F_s`
simultaneously at `T=0.45`. This diagnostic asks which discarded dependence is
responsible, without adding a fitted mechanism.

For each physical lag `t`, the event displacement is represented exactly as

```text
F_event(k,t) = sum_n P_t(n) K_t(k | n),
```

where `P_t(n)` is the sliding-window event-count distribution and `K_t` is the
net event-path characteristic conditioned on both the count and the physical
lag. The total displacement is then tested with the existing independent
event/cage-residual convolution. Count, path kernel, and cage residual are each
taken from either the calibration or held-out half, producing all eight
counterfactual combinations from `ccc` to `hhh`.

The primary retrospective transfer row is `ccc`: all three inputs come from the
calibration half. The `hhh` row is an oracle representation check, not a
prediction.

## Result

The pooled ensemble NGP is reconstructed from pooled second and fourth moments;
replicate NGP values are not averaged directly.

| Gate | `T=0.45` lag-pooled `K(n)` | `T=0.45` lag-conditioned `K_t(n)` | `T=0.58` held-out oracle `hhh` |
|---|---:|---:|---:|
| maximum ensemble MSD relative error | 0.0887 | 0.0723 | 0.3654 |
| maximum ensemble NGP absolute error | 0.1893 | 0.2362 | 0.5727 |
| maximum ensemble multi-`k` `F_s` error | 0.0331 | 0.0239 | 0.0557 |
| curve gate | fail | pass | fail |
| derived-scalar gate | pass | pass | not used |

At `T=0.45`, the same calibration inputs also give relative errors `0.0343` for
`D`, `0.0306` for `tau_alpha(k=7.25)`, and `0.00484` for
`D tau_alpha`. The joint empirical interval closure replaces the failed earlier
global-hybrid ensemble gate, whose MSD/NGP/`F_s` maxima were
`0.1965/0.5284/0.0755`.

The matched ablation pools the calibration path sufficient statistics over all
tested lags with their sample counts, so it has the same count support, the
same empirical `P_t(N)`, and the same calibration residual as `K_t(n)`. It
passes MSD, NGP, `D`, `tau_alpha`, and `D tau_alpha`; only the frozen multi-`k`
shape gate distinguishes the two kernels. Retaining lag lowers the maximum
`F_s` error by `27.9%` and closes that ensemble gate. This is incremental
evidence that count alone is insufficient for the tested low-temperature
alpha-shape transfer, not evidence that every observable requires lag.

The positive statement remains narrow. Individual restart trajectories still
fail (`0.3007/0.4000/0.1141` maximum errors for the lag-conditioned closure),
and only two of three restart labels improve in mean `F_s` error relative to
the lag-pooled kernel. All three restarts descend from one source trajectory.
The independent-parent count is one, so neither a preregistered prediction nor
a general lag-conditioning claim is allowed. No finite-memory law is identified.

At `T=0.58`, even `hhh` fails. The cage-event factorization is therefore not a
universal liquid representation; it is, at most, a low-temperature caged-regime
coarse graining.

## Relation to prior CTRW work

This is not a claim to have invented correlated CTRW or an `n`-jump
propagator.

- Rubner and Heuer directly measured the `n`-metabasin propagator and used the
  waiting-time distribution to predict wave-vector-dependent structural
  relaxation. Their CTRW conditions separate temporal counting from spatial
  propagation.
- Helfferich et al. directly measured `p_k(r)` after `k` jumps and showed that
  forward-backward and longer jump correlations invalidate an iid convolution.
- Pastore, Coniglio, and Pica Ciamarra connected cage-jump counts,
  persistence/exchange statistics, diffusion, and structural relaxation in KA
  and related liquids.

The incremental contribution here is the held-out diagnostic: measure
`K_t(displacement | N=n)` in fixed physical-time intervals, compare it against
the matched sample-weighted `K(n)` ablation, and localize
calibration-to-heldout drift through the complete count/path/residual
counterfactual cube while scoring MSD, pooled NGP, multi-`k` `F_s`, `D`,
`tau_alpha`, and `D tau_alpha` together.

Primary references:

- [Rubner and Heuer, Phys. Rev. E 78, 011504 (2008)](https://doi.org/10.1103/PhysRevE.78.011504)
- [Helfferich et al., Phys. Rev. E 89, 042603 (2014)](https://doi.org/10.1103/PhysRevE.89.042603)
- [Helfferich et al., Phys. Rev. E 89, 042604 (2014)](https://doi.org/10.1103/PhysRevE.89.042604)
- [Pastore et al., Scientific Reports 5, 11770 (2015)](https://doi.org/10.1038/srep11770)
- [Pastore et al., J. Chem. Phys. 155, 074501 (2021)](https://doi.org/10.1063/5.0059622)

## Provenance and reproduction

These rows use the Obadiya-Sussman KA trajectory archive, DOI
`10.5281/zenodo.7469766`. They are not GlassBench trajectory rows.

```bash
python3 scripts/analyze_ka_interval_conditioned_event_transfer.py \
  --trajectory-root /path/to/ka_replicates \
  --cache-directory /tmp/ka_interval_stats
python3 scripts/summarize_ka_interval_conditioned_event_transfer.py
python3 -m unittest tests.test_ka_interval_conditioned_event_transfer -v
```

Interval caches are bound to the full trajectory SHA-256 and the complete event
extraction protocol. Legacy or mismatched caches are rejected; pass
`--rebuild-cache` to recompute and replace them explicitly.

The next decisive test is a preregistered calibration/held-out transfer on at
least three independently prepared parent trajectories per temperature. The
high-temperature branch requires a different displacement representation before
any universal crossover claim is considered.
