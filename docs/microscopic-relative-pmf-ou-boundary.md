# Microscopic Relative PMF And Markov-OU Boundary

## Question

Can the harmonic cage term in the effective theory be derived from
microscopic state counting, and does that derivation already imply complete
Markovian cage dynamics?

## Projected Equilibrium Identity

After the additive correlated-noise closure, the relative marginal has

```text
du = p dt,
dp = [a(u) - gamma p + unresolved force] dt + sqrt(d) dW.
```

For a stationary constant-metric process, cancellation in the Fokker-Planck
equation gives

```text
sigma_p^2 = d / (2 gamma),
E[b_u + gamma p | u] = sigma_p^2 grad_u log rho(u).
```

This is a microscopic state-counting relation. If

```text
rho(u) proportional exp[-|u|^2 / (2 sigma_u^2)],
```

then the conditional acceleration is harmonic:

```text
E[b_u + gamma p | u] = -kappa u,
kappa = sigma_p^2 / sigma_u^2.
```

The effective projected mass is `M_eff=T/sigma_p^2`. Consequently the
acceleration stiffness is not the temperature-naive `T/sigma_u^2`; that would
omit `M_eff`.

## Held-Clone Protocol

For each of four held clones, the other three determine `sigma_u^2`, the
constant relative covariance rate `d`, and therefore `sigma_p^2` and `kappa`.
No force samples from the held clone enter the prediction. The held force is
used only to evaluate radial conditional means in 87 to 90 populated bins.

## Static Results

| Quantity | Four-fold result |
|---|---:|
| `sigma_u^2` | `9.0300e-4` |
| observed `sigma_p^2` | `0.21937` |
| FDT `d/(2 gamma)` | `0.21956` |
| maximum FDT relative error | `0.00585` |
| state-counting quadratic coefficient | `-0.49809` |
| `kappa` from FDT/state counting | `243.146` |
| minimum held mean-force correlation | `0.99056` |
| maximum held mean-force NRMSE | `0.06961` |
| minimum temperature-naive NRMSE | `1.58284` |
| maximum normalized `u-p` covariance | `0.00729` |

The measured radial vector density has the Gaussian coefficient `-0.49809`,
close to the parameter-free value `-1/2`. The FDT velocity variance agrees
with the observed variance in every fold. Together these quantities predict
the held conditional mean force without fitting that force.

The temperature-naive null fails by more than an order of magnitude relative
to the PMF closure. This confirms that the projected effective mass is a
necessary part of the microscopic bridge.

## Dynamic Test

The parameter-free Markov model would be

```text
du = p dt,
dp = [-kappa u - gamma p] dt + sqrt(d) dW.
```

Its exact `2 x 2` propagator was compared with held `C_uu`, `C_pp`, and
`C_up` at 1, 2, 4, 8, 16, 32, 64, and 100 saved frames. The maximum
correlation error is `1.09340`. At 16 frames the observed `C_uu` remains
roughly `0.34-0.37`, whereas the Markov OU predicts approximately `-0.71`.

The microscopic mean-force residual is also strongly colored: its maximum
lag-one correlation is `0.94227`. Thus the equilibrium conditional mean is
harmonic, but instantaneous relative force fluctuations are a structured
many-body bath rather than white thermostat noise.

## Physical Verdict

The harmonic cage plateau now has a direct microscopic derivation:

```text
many-particle Langevin projection
-> constant relative FDT metric
-> microscopic rho(u)
-> Gaussian PMF
-> kappa = [d/(2 gamma)] / sigma_u^2.
```

This validates the static cage term in the effective theory. It does not
validate a complete Markovian OU cage process. A non-Markovian relative-force
kernel or additional local collective coordinates are still required to
reproduce cage-interior time correlations.

The claim boundary is

```text
relative_pmf_static_closure_allowed = 1
markovian_relative_ou_allowed = 0
relative_force_memory_required = 1
autonomous_relative_dynamics_allowed = 0
autonomous_single_particle_gle_allowed = 0
complete_event_clock_closure_allowed = 0
kramers_escape_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

## Reproduction

```bash
python scripts/analyze_ka_relative_pmf_ou_boundary.py \
  --drift-cache-directory tmp/decomposed_cage_drift_reduced_T058 \
  --covariance-cache-directory tmp/projected_ito_covariance_T058 \
  --output-prefix data/renewal_cage_ka_relative_pmf_ou_boundary_T058
```
