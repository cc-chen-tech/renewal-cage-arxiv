# Variance-Mixture Shape Quotient Design

## Scientific question

After matching the calibration surrogate to the held-out MSD, can the remaining
multi-wave-number self-scattering residual be determined by the held-out NGP
residual without a fitted macroscopic parameter?

The diagnostic is deliberately conditional.  It uses held-out MSD and NGP as
inputs and therefore cannot support a blind prediction or microscopic closure
claim.

## Frozen quotient

For an isotropic three-dimensional displacement let

\[
x = k^2\langle r^2\rangle/6,
\qquad
\alpha = \alpha_2.
\]

A unit-mean positive variance multiplier with variance \(\alpha\) gives a
Laplace scattering law \(L(x;\alpha)\).  Two analytic, parameter-free families
are tested:

\[
L_\Gamma=(1+\alpha x)^{-1/\alpha},
\qquad
L_{IG}=\exp\!\left[\frac{1-\sqrt{1+2\alpha x}}{\alpha}\right].
\]

Both reduce continuously to \(e^{-x}\) at \(\alpha=0\), and both have

\[
\log L=-x+\alpha x^2/2+O(x^3),
\]

so the existing fourth-cumulant prediction is the common low-\(k\) limit.
For each supported fixed-MSD calibration/held-out pair, family \(f\) predicts

\[
F_s^{(f)}(k)=F_{s,\mathrm{cal}}(k)
\frac{L_f(x;\alpha_{\mathrm{held}})}
     {L_f(x;\alpha_{\mathrm{cal}})}.
\]

No extrapolated row is admitted.  The input grid remains the committed
transport-clock quotient grid at \(T=0.45,0.58\) and \(k=2,4,7.25\).

## Gate

The primary error is the absolute self-scattering error divided by the frozen
0.03 tolerance.  A temperature-wave-number cell passes family-robust closure
only when both analytic families have maximum normalized error at most one.

The low-temperature exploratory result is allowed only when:

- the source transport-clock gate remains valid and stationary at \(T=0.45\);
- clock-only or fourth-cumulant truncation fails at least one supported cell;
- both nonlinear resummations pass every supported row at all three wave numbers;
- no family is selected as unique.

The \(T=0.58\) result remains a canary because its source stationarity control
is unresolved.  Every output keeps blind prediction, static-environment,
finite-exchange, microscopic, spatial-facilitation, and thermodynamic claim
flags at zero.

## Artifacts

- `scripts/summarize_ka_variance_mixture_shape_quotient.py`
- `data/renewal_cage_ka_variance_mixture_shape_quotient_rows.csv`
- `data/renewal_cage_ka_variance_mixture_shape_quotient_gate.csv`
- `figures/renewal_cage_ka_variance_mixture_shape_quotient.svg`
- focused unit and arXiv-package tests

The figure compares clock-only, fourth-order, gamma-resummed, and
inverse-Gaussian-resummed maximum errors.  It must state that held-out MSD and
NGP are diagnostic inputs and that the variance-mixture family is unresolved.
