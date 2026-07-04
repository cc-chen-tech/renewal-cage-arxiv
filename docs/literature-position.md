# Literature Position for the Delayed Renewal Cage Model

This note records the current positioning of the renewal-cage result against nearby
literature.

## Nearby Literature

- Chubynsky and Slater's diffusing-diffusivity model established a compact mechanism
  for anomalous yet Brownian diffusion with diffusivity memory but no directional
  memory: https://doi.org/10.1103/PhysRevLett.113.098302
- Chechkin, Seno, Metzler, and Sokolov developed Brownian yet non-Gaussian diffusion
  through superstatistics and subordination of fluctuating diffusivities:
  https://arxiv.org/abs/1611.06202
- Fickian non-Gaussian diffusion has been studied directly in glass-forming liquids:
  https://doi.org/10.1103/PhysRevLett.128.168001
- There is also an explicit critique of treating supercooled liquids as simply Fickian
  yet non-Gaussian:
  https://arxiv.org/abs/2210.07119
- Cage-size and jump-precursor studies support the physical relevance of cage
  rearrangement events before jumps:
  https://www.researchgate.net/publication/315328573_Cage_Size_and_Jump_Precursors_in_Glass-Forming_Liquids_Experiment_and_Simulations

## What Is Not Novel Enough

The following claims alone are not enough for an arXiv-worthy result:

```text
1. Non-Gaussian diffusion can arise from fluctuating diffusivity.
2. Gaussian mixtures can produce a nonzero NGP.
3. Cage jumps occur in glass-forming systems.
4. A harmonic cage can produce an MSD plateau.
```

Each point is already known or too direct.

## Current Novel Angle

The delayed renewal cage model combines four ingredients in a closed-form way:

```text
local OU-like cage variance:
  L(t) = A[1-exp(-t/tau_c)]

delayed renewal intensity:
  r(t) = lambda[1-exp(-t/tau_d)]^2

Gaussian cage-center jump variance:
  q

exact NGP:
  alpha_2(t) = q^2 R(t) / [L(t)+qR(t)]^2
```

This gives three analytic regimes:

```text
short time:
  alpha_2(t) ~ [q^2 lambda tau_c^2/(3A^2 tau_d^2)] t

peak condition:
  R'(t)[L(t)-qR(t)] - 2R(t)L'(t) = 0
  plateau approximation: qR(t*) = A

long time:
  alpha_2(t) ~ 1/(lambda t)

observable peak diagnostics:
  q/A ~= 4 alpha_2(t*)
  R(t*) ~= 1/[4 alpha_2(t*)]
  lambda ~= 1/[4 alpha_2(t*) tau_d F(t*/tau_d)]
```

The publishable claim should therefore be:

> A delayed renewal count regularizes the short-time singularity of memoryless jump
> models while preserving a closed-form NGP peak, simple peak-height diagnostics,
> and long-time Gaussian recovery.

## What Still Needs Strengthening

Before arXiv submission, the work should add:

```text
1. Final prose polish and narrower claims.
2. Reference metadata checks.
3. Author and affiliation confirmation.
4. A final rendered-PDF read-through.
```
