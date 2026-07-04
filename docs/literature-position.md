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
- Fickian non-Gaussian diffusion has been studied directly in glass-forming liquids,
  with later comment/reply discussion about whether that language captures the
  relevant supercooled-liquid heterogeneity:
  https://doi.org/10.1103/PhysRevLett.128.168001
  https://doi.org/10.1103/PhysRevLett.131.119801
  https://doi.org/10.1103/PhysRevLett.131.119802
- Cage-rearrangement and cage-size precursor studies support the physical
  relevance of renewal-like rearrangement events before jumps:
  https://doi.org/10.1103/PhysRevLett.89.095704
  https://doi.org/10.1021/acs.jpclett.7b00187

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
  generalized check: r_m(t) = lambda[1-exp(-t/tau_d)]^m

Gaussian cage-center jump variance:
  q

exact NGP:
  alpha_2(t) = q^2 R(t) / [L(t)+qR(t)]^2
```

This gives three analytic regimes:

```text
short time:
  alpha_2(t) ~ [q^2 lambda tau_c^2/(3A^2 tau_d^2)] t
  generalized exponent:
    alpha_2(t) ~ [q^2 lambda tau_c^2/((m+1)A^2 tau_d^m)] t^(m-1)
    m<1 singular origin, m=1 finite origin, m>1 regular zero origin
    m=2 is the minimal integer regular choice

peak condition:
  R'(t)[L(t)-qR(t)] - 2R(t)L'(t) = 0
  plateau approximation: qR(t*) = A

long time:
  alpha_2(t) ~ 1/(lambda t)

observable peak diagnostics:
  q/A ~= 4 alpha_2(t*)
  R(t*) ~= 1/[4 alpha_2(t*)]
  lambda ~= 1/[4 alpha_2(t*) tau_d F(t*/tau_d)]

finite-time consistency check:
  beta = q/A ~= 4 alpha_2(t*)
  alpha_l = beta y_l/(1+y_l)^2 on the late branch
  lambda_l ~= y_l/[beta tau_d F(t_l/tau_d)]
  compare lambda_l with the peak-inferred lambda
```

The publishable claim should therefore be:

> A delayed renewal count regularizes the short-time singularity of memoryless jump
> models while preserving a closed-form NGP peak, simple peak-height diagnostics,
> finite-time consistency diagnostics, and long-time Gaussian recovery.

The square-delay choice is now justified as the minimal integer member of a
generalized delay-exponent class that gives a regular zero-origin NGP.

This should be stated as a minimal diagnostic model, not as a replacement for
microscopic glass theory or as a claim that all supercooled-liquid dynamics is
Fickian non-Gaussian.

The novelty should be positioned around delayed discrete cage-center renewal,
not around random diffusivity. The model's added glass-literature observable is
the closed-form self-intermediate scattering function:

```text
F_s(k,t) = exp[-k^2 L(t)/2 + R(t)(exp(-k^2 q/2)-1)]
Phi_alpha(k,t) = exp[-(1-exp(-k^2 q/2)) R(t)]
tau_alpha(k)^-1 ~= lambda[1-exp(-k^2 q/2)]
```

Competing mechanisms should be asked to reproduce the same four observables:
the MSD plateau, the early-time NGP exponent, peak/late-time renewal-rate
consistency, and the long-time `1/t` NGP decay.

## What Still Needs Strengthening

Before arXiv submission, the work should add:

```text
1. Final prose polish and narrower claims.
2. Author and affiliation confirmation.
3. A final rendered-PDF read-through.
```
