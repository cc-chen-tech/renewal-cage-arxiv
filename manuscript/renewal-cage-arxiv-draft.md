# A Delayed Renewal Cage Model for Non-Gaussian Dynamics Near Glass Transition

## Abstract

Glass-forming liquids and polymer melts often display a transient non-Gaussian
parameter (NGP) peak during the crossover from localized cage motion to long-time
transport. A harmonic Langevin cage explains the mean-squared-displacement (MSD)
plateau but remains Gaussian and therefore cannot generate a nonzero NGP. Here we
introduce a minimal delayed renewal cage model in which local Ornstein-Uhlenbeck
cage motion is interrupted by delayed cage-center renewal events. The model yields
closed-form expressions for the MSD and NGP:

```text
MSD(t) = L(t) + q R(t)
alpha_2(t) = q^2 R(t) / [L(t) + q R(t)]^2
```

where `L(t)` is the local cage variance, `q` is the jump variance per renewal event,
and `R(t)` is the mean delayed renewal count. With a renewal intensity
`r(t)=lambda[1-exp(-t/tau_d)]^2`, the NGP starts from zero, develops a finite-time
maximum, and decays as `1/(lambda t)` at long times. In the plateau regime the peak
condition reduces to `q R(t*) = A`, giving `alpha_2(t*) = q/(4A)`. The model therefore
separates the roles of cage size, cage-breaking delay, renewal rate, and jump variance,
and provides a compact analytic mechanism for dynamic heterogeneity without
requiring non-Gaussian microscopic noise.

## 1. Motivation

The thesis `Zijun_Lu_thesis-merged.pdf` shows that a Langevin equation with a
harmonic potential can reproduce low-temperature MSD plateau behavior, but that
this Gaussian description does not explain the NGP signal observed near glass
transition. This gap is also visible in the broader literature on Fickian yet
non-Gaussian diffusion and dynamic heterogeneity: non-Gaussian displacement
statistics can coexist with ordinary long-time transport, but a minimal link between
local caging, cage renewal, and the NGP peak remains useful.

The present model targets exactly that gap. It does not attempt to fit a particular
polymer. Instead, it asks for the smallest analytically tractable stochastic model with
all three qualitative features:

```text
1. MSD plateau from local caging
2. finite-time NGP peak from heterogeneous renewal
3. long-time Gaussian recovery
```

## 2. Model

The displacement is decomposed into local cage motion plus cage-center jumps:

```text
Delta x(t) = xi_cage(t) + sum_{j=1}^{N(t)} eta_j.
```

The local cage contribution is Gaussian with variance

```text
L(t) = A [1 - exp(-t/tau_c)].
```

The cage-center jumps are independent Gaussian increments with variance `q`.
The renewal count `N(t)` is an inhomogeneous Poisson process with intensity

```text
r(t) = lambda [1 - exp(-t/tau_d)]^2.
```

The mean count is

```text
R(t)
  = lambda [t - 2 tau_d(1 - exp(-t/tau_d))
            + (tau_d/2)(1 - exp(-2t/tau_d))].
```

The delayed intensity suppresses unphysical instantaneous cage jumps:

```text
R(t) ~ lambda t^3/(3 tau_d^2),  t -> 0.
```

## 3. Moments

Conditional on `N(t)=n`, the displacement is Gaussian with variance

```text
V(t|n) = L(t) + n q.
```

Since `N(t)` is Poisson,

```text
E[V] = L(t) + qR(t)
Var[V] = q^2 R(t).
```

Thus

```text
M2(t) = E[Delta x(t)^2] = L(t) + qR(t)
```

and

```text
M4(t) = E[Delta x(t)^4]
      = 3E[V^2]
      = 3[(L(t)+qR(t))^2 + q^2 R(t)].
```

The one-dimensional NGP is

```text
alpha_2(t) = M4(t)/(3M2(t)^2) - 1,
```

so

```text
alpha_2(t) = q^2 R(t) / [L(t) + qR(t)]^2.
```

For an isotropic three-dimensional displacement, the same scalar variance `V` applies
to each Cartesian coordinate. Conditional on `V`,

```text
<r^2|V> = 3V,
<r^4|V> = 15V^2.
```

The standard three-dimensional NGP,

```text
alpha_2^3D(t) = 3<r^4>/(5<r^2>^2) - 1,
```

therefore reduces to the same variance-mixture expression:

```text
alpha_2^3D(t) = Var[V]/E[V]^2
              = q^2 R(t) / [L(t) + qR(t)]^2.
```

Thus the model's central prediction is independent of whether the NGP is computed in
one dimension or with the standard three-dimensional definition.

## 4. Asymptotics and Peak Condition

At short times,

```text
L(t) ~ (A/tau_c)t
R(t) ~ lambda t^3/(3 tau_d^2),
```

therefore

```text
alpha_2(t)
  ~ [q^2 lambda tau_c^2/(3A^2 tau_d^2)] t.
```

The NGP starts from zero. At long times,

```text
L(t) -> A
R(t) ~ lambda t,
```

and

```text
alpha_2(t) ~ 1/(lambda t).
```

The peak condition follows by differentiating the exact NGP:

```text
R'(t)[L(t)-qR(t)] - 2R(t)L'(t) = 0.
```

Once local cage relaxation has reached the plateau, `L'(t)≈0` and `L(t)≈A`, giving

```text
qR(t*) = A,
alpha_2(t*) = q/(4A).
```

This predicts that the renewal delay primarily shifts the peak time, whereas the
jump variance primarily controls the peak height.

A finite-time consistency check follows from `beta=q/A≈4 alpha_2(t*)` and
`y(t)=beta R(t)`. On the late branch, `alpha_l=beta y_l/(1+y_l)^2`, so a late
NGP value gives `y_l` and therefore
`lambda_l≈y_l/[beta tau_d F(t_l/tau_d)]`. Agreement between this value and the
peak-inferred renewal rate is a falsifiable check using only observable
peak/late-NGP quantities plus `tau_d`.
The inverse is admissible only for `alpha_l<=beta/4`; its two roots obey
`y_- y_+=1`, and the late-time branch is the `y_+>1` root.

Alternative mechanisms can still be viable, but to fall into the same diagnostic
class they must reproduce the MSD plateau, the early-time NGP exponent, the
peak/late-time rate consistency, and the long-time `1/t` NGP decay.

## 4.5. Self-intermediate scattering function

The same delayed renewal count gives a closed-form glass-literature observable:

```text
F_s(k,t) = exp[-k^2 L(t)/2 + R(t)(exp(-k^2 q/2)-1)].
```

Equivalently,

```text
F_s(k,t) = exp[-k^2 L(t)/2] Phi_alpha(k,t)
Phi_alpha(k,t) = exp[-Gamma_k R(t)]
Gamma_k = 1 - exp(-k^2 q/2).
```

This gives a cage Debye-Waller plateau `f_k=exp(-k^2 A/2)` and a renewal
controlled long-time alpha rate:

```text
tau_alpha(k)^-1 ~= lambda [1 - exp(-k^2 q/2)].
```

This is the main reason to position the note around delayed cage renewal rather
than generic random diffusivity.

## 4.6. Temperature dependence and Stokes-Einstein decoupling

A minimal temperature extension assigns reduced-temperature dependence to the
effective renewal parameters:

```text
Delta_T = 1/T - 1/T0
lambda(T) = lambda0 exp[-E_lambda Delta_T]
tau_d(T)  = tau_d0 exp[ E_d Delta_T]
A(T)      = A0 exp[-E_A Delta_T]
q(T)/A(T) = beta0 exp[E_beta Delta_T]
```

The long-time diffusion coefficient is

```text
D(T) = lambda(T) q(T) / 2.
```

The cage-normalized alpha time is defined by

```text
Gamma_k(T) R(tau_alpha;T) = 1
Gamma_k(T) = 1 - exp[-k^2 q(T)/2].
```

Since `R(t;T)=lambda(T) tau_d(T) F[t/tau_d(T)]`,

```text
tau_alpha(k,T)
  = tau_d(T) F^{-1}[1/(Gamma_k lambda(T) tau_d(T))]
```

and therefore

```text
D(T) tau_alpha(k,T)
  = lambda(T) q(T) tau_d(T)
    F^{-1}[1/(Gamma_k lambda(T) tau_d(T))] / 2.
```

This is the model's Stokes-Einstein diagnostic. Changing `lambda(T)` alone mostly
rescales diffusion and relaxation together. Decoupling appears when cooling makes
the delayed-onset product `lambda(T) tau_d(T)` grow, so structural relaxation is
controlled by delayed cage renewal more strongly than long-time diffusion.

## 5. Reproducible Results

The repository contains the exact implementation:

```text
src/renewal_cage.py
tests/test_renewal_cage.py
scripts/generate_renewal_cage_results.py
```

The script produces:

```text
data/renewal_cage_main.csv
data/renewal_cage_sweeps.csv
data/renewal_cage_consistency.csv
data/renewal_cage_scattering.csv
data/renewal_cage_temperature.csv
figures/renewal_cage_results.svg
figures/renewal_cage_dimensionless.svg
figures/renewal_cage_scattering.svg
figures/renewal_cage_temperature.svg
```

The current parameter set gives:

```text
NGP peak time  = 11.306
NGP peak value = 0.200
NGP final      = 0.0293 at t=180
MSD final      = 26.272
lambda_*       = 0.180010 from peak diagnostics
lambda_l       = 0.180000 from finite-time late-NGP inversion
```

For the illustrative temperature law, cooling from `T=1.00` to `T=0.62`
decreases `D` by a factor of `3.30`, increases `tau_alpha(k=1.1)` by a factor
of `6.72`, and raises the normalized `D tau_alpha` product to `2.03`.

The parameter sweeps show:

```text
Increasing tau_d shifts the NGP peak to later times with nearly unchanged peak height.
Increasing q raises the peak and shifts it earlier, consistent with qR(t*)≈A.
```

The dimensionless output file `data/renewal_cage_dimensionless.csv` rescales time by
the predicted peak time and rescales NGP by the predicted peak height. In the
plateau-dominated regime the curves collapse because the leading peak condition
depends only on `qR/A`.

The van Hove output file `data/renewal_cage_van_hove.csv` gives the three-dimensional
radial distribution

```text
G_s(r,t) = sum_n Pr[N(t)=n]
           sqrt(2/pi) r^2 / V_n(t)^(3/2) exp[-r^2/(2V_n(t))]
```

with

```text
V_n(t) = L(t) + nq.
```

This distribution is normalized and displays a transient broad tail near the NGP peak,
then becomes closer to a Gaussian radial form as the renewal count grows.

## 6. Relation to Existing Literature

This work is adjacent to Fickian yet non-Gaussian diffusion, diffusing-diffusivity
models, continuous-time random walks, and cage-jump descriptions of glassy dynamics.
The specific contribution here is the closed-form combination of:

```text
local OU cage plateau
delayed renewal count
Gaussian jump increments
exact NGP peak and asymptotic formulas
```

The delayed renewal intensity is the key technical device that avoids the short-time
NGP singularity of a memoryless Poisson jump process while preserving analytic
tractability.

## 7. Next Checks Before Submission

1. Add a Gaussian-baseline overlay for the van Hove distribution.
2. Add a small set of dimensionless plots using `q/A`, `lambda tau_c`, and
   `tau_d/tau_c` in final publication styling.
3. Add a table positioning the model against diffusing diffusivity, CTRW, and
   cage-jump models.
4. Convert this draft to LaTeX with BibTeX references.
