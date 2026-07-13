# Empirical Path Transfer Gate Design

## Purpose

Determine whether the low-temperature failure of the two-state Markov displacement
kernel is caused by single-particle multiblock path memory before adding another
latent clock or a spatial facilitation parameter. The gate is diagnostic rather
than a final microscopic theory: it identifies which information must be retained
by the next minimal closed model.

The claim remains restricted to dynamical glass signatures.
`thermodynamic_claim_allowed=0` and `spatial_facilitation_claim_allowed=0` remain
mandatory. Because the empirical kernel consumes calibration displacement paths,
`microdynamic_closure_claim_allowed=0` also remains mandatory.

## Frozen data protocol

- Use the existing independently restarted Kob-Andersen ensembles at `T=0.45`
  and `T=0.58`.
- Keep the established calibration and held-out halves, Debye-Waller jump rule,
  physical time axis, and block size of 20 trajectory samples.
- Construct full physical block-displacement vectors. No held-out displacement,
  event, or observable may enter a calibration kernel.
- Score only lags divisible by the block size and present in the existing
  factorization tables.
- Aggregate predictions and observations over independent replicates before the
  primary pass/fail decision; retain replicate rows for uncertainty and drift
  diagnostics.

## Competing path mechanisms

All mechanisms use the same calibration blocks and held-out observations.

1. `contiguous_empirical_path` retains each calibration particle's ordered,
   contiguous block vectors. Its cumulative displacement distribution is the
   nonparametric upper bound for a stationary single-particle path theory.
2. `within_particle_time_shuffle` permutes block order separately for each
   particle. It preserves particle identity and the one-block vector distribution
   while destroying temporal path memory.
3. `direction_randomized_path` preserves every observed block length and its time
   order but replaces each direction by an independent isotropic direction. It
   preserves amplitude persistence while removing recoil and directional path
   memory.
4. `two_state_markov_kernel` is the existing state-conditioned finite-window
   Green-Kubo prediction. It is included unchanged as the parametric Markov null.

The randomized mechanisms use fixed, recorded seeds and enough deterministic
realizations for Monte Carlo standard errors to be below one tenth of each scoring
tolerance. They add no fitted macro-observable parameter.

## Scoring and decision

At every common held-out lag, score ensemble MSD, NGP, and all available
self-intermediate scattering functions. Preserve the existing preregistered
tolerances:

- maximum relative MSD error: `0.10`;
- maximum absolute NGP error: `0.30`;
- maximum absolute multi-k `F_s` error: `0.03`.

Select `single_particle_multiblock_path_memory_required=1` only if:

- the contiguous empirical path passes all three curve tolerances at `T=0.45`;
- the two-state Markov kernel fails at least one low-temperature higher-order
  observable;
- the within-particle time shuffle fails the same low-temperature observable;
- at least two of the three `T=0.45` replicates have a smaller paired error for
  the contiguous kernel than for the time-shuffled null on that observable; and
- the contiguous empirical path also passes the ensemble curve tolerances at
  `T=0.58`.

If the contiguous path fails, classify the result as nonstationarity or missing
cross-particle information and do not fit a single-particle recoil model. If it
passes but the shuffled nulls also pass, no additional path-memory parameter is
supported.

## Outputs and tests

Produce per-replicate rows, ensemble summaries, a one-row verdict, a two-temperature
crossover table, and one comparison SVG. Tests must cover information-preserving
properties of both randomizations, deterministic seeds, held-out exclusion,
multi-k scoring, the decision truth table, artifact provenance, and the existing
microdynamic, thermodynamic, and spatial claim boundaries.

The next parametric kernel is not part of this change. It will be designed only
after this gate identifies whether ordered recoil, amplitude persistence, or data
nonstationarity is the necessary missing channel.
