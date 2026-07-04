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
the closed-form self-intermediate scattering function and its temperature
extension:

```text
F_s(k,t) = exp[-k^2 L(t)/2 + R(t)(exp(-k^2 q/2)-1)]
Phi_alpha(k,t) = exp[-(1-exp(-k^2 q/2)) R(t)]
tau_alpha(k)^-1 ~= lambda[1-exp(-k^2 q/2)]

temperature law:
  lambda(T) = lambda0 exp[-E_lambda(1/T-1/T0)]
  tau_d(T) = tau_d0 exp[E_d(1/T-1/T0)]
  q(T)/A(T) = beta0 exp[E_beta(1/T-1/T0)]

alpha time:
  tau_alpha(k,T) = tau_d(T) F^{-1}[1/(Gamma_k lambda(T) tau_d(T))]

Stokes-Einstein diagnostic:
  D(T) tau_alpha(k,T)
    = lambda(T) q(T) tau_d(T) F^{-1}[1/(Gamma_k lambda(T) tau_d(T))] / 2
  xi_SE(T) = -d log D(T) / d log tau_alpha(k,T)
  ordinary SE: xi_SE = 1
  fractional SE: 0 < xi_SE < 1
  E_app(T) = d log tau_alpha(k,T) / d(1/T)
  local fragility proxy: m_loc(T) = E_app(T)/(T log 10)

activated barrier interpretation:
  lambda(T) tau_d(T) = lambda0 tau_d0 exp[(E_d-E_lambda)(1/T-1/T0)]
  E_d > E_lambda makes delayed renewal increasingly control relaxation on cooling

renewal-count susceptibility:
  chi_R(k,t) = Var_N[exp(-k^2 L(t)/2) exp(-k^2 q N(t)/2)]
  chi_R/F_s^2 = exp[R(t)(exp(-k^2 q/2)-1)^2]-1
  correlated renewal domain:
    chi_4^R(k,t) = N_corr chi_R(k,t)
    N_corr = chi_4,peak^obs / max_t chi_R(k,t)

observable inversion:
  A = -2 log(f_k)/k^2
  [1-exp(-k^2 q/2)]/q
    = -log(h) / [2 D tau_d F(tau_alpha/tau_d)]
  existence margin:
    D tau_d F(tau_alpha/tau_d) k^2 / [-log(h)] > 1
  then lambda = 2D/q and the NGP peak is predicted, not fit

full observable inversion without supplied tau_d:
  q = 4 A alpha_2(t*)
  lambda = 2D/q
  solve F(s)/s = (A/q)/(lambda t*) with s=t*/tau_d
  tau_alpha is then a held-out residual
```

Competing mechanisms should be asked to reproduce the same four observables:
the MSD plateau, the early-time NGP exponent, peak/late-time renewal-rate
consistency, and the long-time `1/t` NGP decay. A stronger glass-transition
comparison can additionally ask whether they reproduce the same growth of
`D tau_alpha` when cooling increases the delayed-renewal control parameter
`lambda tau_d`, whether they predict a fractional exponent `xi_SE<1`, and
whether they produce growing apparent alpha-activation/fragility without an
extra fit parameter. They should also explain whether their dynamic-heterogeneity
measure has the same renewal-count susceptibility peak and whether an observed
`chi_4` amplitude maps to the same cooperative renewal-domain size. The strongest
falsifiability test is now the
scattering-transport inversion: once `F_s` plateau, `D`, `tau_alpha`, and
`tau_d` are specified, the model either has no positive-`q` solution or predicts
the NGP peak time and height without another free parameter.
The stronger inference protocol uses the NGP peak itself to infer `tau_d`, so
the measured `tau_alpha` becomes an independent consistency check.

## What Still Needs Strengthening

Before arXiv submission, the work should add:

```text
1. Final prose polish and narrower claims.
2. Author and affiliation confirmation.
3. A final rendered-PDF read-through.
```
