# Microscopic Projected Ito Innovation Audit

## Question

The decomposed cage-drift bath failed as an autonomous model. This audit asks
a more basic question: before coarse-graining the drift, do the exact smooth
cage coordinates obey the stochastic increments predicted by projecting the
full many-particle Langevin equation?

## Exact Local Equation

For `Y_i=(w_i,p_i)` and `b_i=(b_C,b_u)`,

```text
dY_i = b_i(R,V) dt + B_i(R) dW,
B_i B_i^T = Q_i(R).
```

The `6 x 6` rate matrix `Q_i` is assembled from the exact center, relative,
and cross blocks:

```text
Q = [[Q_C,  Q_Cu],
     [Q_Cu^T, Q_u]].
```

All quantities are evaluated from the full 4096-particle state. This equation
is microscopic and local, but not autonomous in the retained single-particle
variables.

## Protocol

- four independent 10 tau KA Langevin clones at `T=0.58`;
- 64 fixed A particles and `0.01 tau` saved-frame spacing;
- exact cached `w,p,b_C,b_u` paths;
- independently reconstructed Jacobian geometry and joint `Q` paths;
- nonoverlapping block strides 1, 2, 4, and 8;
- covariance whitening in the full six-dimensional center-relative space.

The preregistered primary estimator is left-point Ito:

```text
e_n = Y_(n+1) - Y_n - dt b_n,
Sigma_n = dt Q_n.
```

The endpoint trapezoid was preregistered as a discretization sensitivity. An
adapted Adams-Bashforth-2 estimator, using only `b_n` and `b_(n-1)`, was added
after the left/trapezoid discrepancy was observed. Its status is therefore a
post-primary diagnostic, not a replacement for the primary gate.

## Numerical Alignment

Recomputing the cage Jacobian geometry reproduces the cached relative
positions within `2.07e-6` and relative velocities within `9.62e-6`. Every
joint covariance is positive definite. At stride 1 the minimum integrated
eigenvalue is `6.53e-4`.

## Pooled Results

| Estimator | stride | trace ratio | Mahalanobis / 6 | covariance error | max lag | max state corr. |
|---|---:|---:|---:|---:|---:|---:|
| left Ito | 1 | 1.02340 | 1.01210 | 0.03774 | 0.08674 | 0.22274 |
| adapted AB2 | 1 | 0.93279 | 0.93805 | 0.06832 | 0.03043 | 0.03238 |
| adapted AB2 | 2 | 0.95740 | 0.96269 | 0.04290 | 0.02989 | 0.04394 |
| adapted AB2 | 4 | 0.98329 | 0.98722 | 0.02685 | 0.01799 | 0.06372 |
| trapezoid | 1 | 0.96109 | 0.95995 | 0.04354 | 0.02862 | 0.00753 |
| trapezoid | 4 | 0.99555 | 0.99569 | 0.01810 | 0.01063 | 0.02245 |

The left estimator passes variance scale, Mahalanobis, mean, covariance, and
kurtosis checks but fails the preregistered lag and state-correlation gates.
Its dominant state correlation is the `p_y` innovation against starting
`p_y`, `-0.22274`; its dominant lag is the `p_y` autocorrelation, `0.08674`.

The adapted AB2 estimator removes both signatures without using future state.
At stride 1 its largest state correlation is only `0.03238` and its largest
lag is `0.03043`; both improve further under short block aggregation. The
trapezoid estimator gives the same qualitative conclusion across every
stride. Whitened component excess kurtosis is also small: `0.12057` for AB2
and `0.12228` for trapezoid at stride 1.

## Physical Verdict

The preregistered left-point gate remains failed. The failure is nevertheless
localized to first-order drift quadrature at the finite `0.01 tau` saved-frame
spacing: a fully past-adapted second-order estimator and the independent
trapezoid sensitivity both recover the predicted configuration-dependent
noise covariance with weak residual memory. This numerically supports the
exact local projected SDE.

It does not solve autonomous coarse-graining. Computing `b_C`, `b_u`, and `Q`
still requires the full many-particle `R,V,F`. Combined with the failed linear
split bath, the result narrows the missing physics to closure of these
configuration-dependent coefficients, not an error in the cage coordinate or
thermostat projection. The next microscopic closure must predict the drift
and multiplicative covariance from measured local collective state.

The claim boundary is

```text
projected_ito_local_gate_pass = 0
adapted_second_order_consistency_gate_pass = 1
trapezoid_sensitivity_gate_pass = 1
projected_sde_numerically_supported = 1
microscopic_projected_sde_allowed = 0
autonomous_single_particle_gle_allowed = 0
complete_event_clock_closure_allowed = 0
kramers_escape_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

## Reproduction

```bash
python scripts/analyze_ka_projected_ito_innovations.py \
  tmp/isoconfigurational_force_velocity_long_T058 \
  --drift-cache-directory tmp/decomposed_cage_drift_reduced_T058 \
  --covariance-cache-directory tmp/projected_ito_covariance_T058 \
  --output-prefix data/renewal_cage_ka_projected_ito_innovations_T058
```
