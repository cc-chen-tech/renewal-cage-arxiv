# arXiv Readiness Checklist

This checklist tracks the repository state for submitting the delayed renewal
cage note as a short arXiv manuscript.

## Ready

- Standalone repository separate from the liquid-crystal theory work.
- Closed-form MSD, NGP, radial van Hove distribution, self-intermediate
  scattering function, temperature-dependent alpha relaxation, Stokes-Einstein
  decoupling diagnostics, fractional Stokes-Einstein exponents,
  apparent alpha-activation/fragility diagnostics, activated-barrier gap
  diagnostic, alpha-shape time-temperature-superposition residuals,
  renewal-count susceptibility, renewal-domain chi4/cooperative-size
  estimator, peak diagnostics, NGP peak/alpha-relaxation coupling,
  finite-exchange heterogeneity diagnostics with stretched-like alpha decay and
  Gaussian recovery, a temperature-dependent facilitated-exchange law linking
  cooling to stronger exchange heterogeneity, a static-gamma mobility-disorder
  null model proving that non-exchanging disorder leaves a residual NGP plateau, finite-time
  peak/late-NGP consistency diagnostics,
  late-time heterogeneity consistency diagnostics linking NGP amplitude and
  alpha slope, a finite-exchange diagnostic map for the jointly observable
  window, a late-observable residual `Delta_c` that compares NGP-inferred and
  alpha-slope-inferred exchange ratios, uncertainty propagation for its
  statistical `z` score, a multi-wave-number collapse test for `c_alpha(k)`,
  and scattering-transport observable inversion including a full protocol that
  infers the renewal delay from NGP peak timing.
- Generalized delay-exponent argument explaining the square-delay choice.
- Reproducible CSV outputs, SVG figures, PDF manuscript figures, and arXiv
  source zip.
- GitHub Actions workflow that runs tests, regenerates outputs, builds the
  source zip, installs TeX Live, and compiles the manuscript.
- Literature positioning against diffusing diffusivity, Fickian non-Gaussian
  diffusion, and cage-jump observations.

## Before Submission

- Confirm author name and affiliation exactly as they should appear on arXiv.
- Do one final read-through of the rendered PDF for wording, figure placement,
  and reference formatting.
- Decide whether to submit as an independent research note or add an
  institutional affiliation/acknowledgment.
