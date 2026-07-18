# Microscopic Transient-Periodic Langevin Canary

## Question

The empirical activated-jump quotient showed that calibration jump geometry
alone cannot reconstruct the T=0.45 cage-scale scattering curve.  This step
separates two missing contributions:

```text
Var(N) != <N>        finite rate/count disorder,
Cov(J_i,J_j) != 0    ordered recoil/path memory.
```

It asks whether both effects can emerge from one continuous equilibrium
Langevin system rather than from an imposed renewal clock or event-level recoil
rule.

## Count-overdispersion diagnostic

For a count `N` with mean `n` and Fano factor `F`, independent isotropic jump
marks with radial second moment `m2` and component fourth moment `m4x` give

```text
kappa_4,x = n m4x + 3 n(F-1)(m2/3)^2,
kappa_4,x = alpha_2 M^2/3.
```

Therefore

```text
n = alpha_2 M^2 / {3[m4x + 3(F-1)(m2/3)^2]},
a = (M-n m2)/6.
```

Using a gamma-Poisson marginal with the same moments,

```text
G_N(s) = [1+(F-1)(1-s)]^[-n/(F-1)],
F_s(k) = exp(-k^2 a) G_N(phi_J(k)).
```

The Fano factor comes from the existing calibration two-clock count model.
Heldout MSD and NGP are diagnostic inputs; this is not a blind prediction.

At T=0.45, count overdispersion raises physical support from the earlier
Poisson result `8/21` to `20/21`.  The supported-row errors are

| wave number | maximum absolute error |
|---:|---:|
| 2 | 0.002402856 |
| 4 | 0.025613101 |
| 7.25 | 0.041088459 |

The high-k error remains above the unchanged `0.03` tolerance.  T=0.58 has
`25/25` support but maximum high-k error `0.046642857`; its source stationarity
gate fails, so it remains a canary and cannot establish cooling growth.

Count overdispersion fixes the fourth-cumulant support problem but not the
cage-scale displacement shape.

## Continuous equilibrium model

For tagged position `x in R^d`, slow elastic environment `q in R^d`, and one
shared barrier coordinate `z`, use

```text
V(z) = V0 + g z^2,

U(x,q,z) = sum_i {V(z)[1-cos(2 pi x_i/L)]/2
                  + K(x_i-q_i)^2/2}
           + k_z z^2/2.
```

The overdamped Ito equations are

```text
gamma_x dx = -partial_x U dt + sqrt(2 gamma_x T) dW_x,
gamma_q dq = -partial_q U dt + sqrt(2 gamma_q T) dW_q,
gamma_z dz = -partial_z U dt + sqrt(2 gamma_z T) dW_z.
```

Every coordinate uses the same temperature and its own fluctuation-dissipation
pair.  The potential is invariant under the joint shift
`(x,q) -> (x+nL,q+nL)`, so the common translation remains diffusive.  The
event index `floor(x/L+1/2)` is not part of the SDE; it is extracted afterward
with a five-frame non-recrossing dwell.

Near a well minimum at frozen `q,z`,

```text
kappa_x(z) = 2 pi^2 V(z)/L^2 + K,
tau_c(z) = gamma_x/kappa_x(z),
<delta x^2 | z> approximately T/kappa_x(z).
```

For weak elastic bias relative to the periodic barrier, frozen-environment
Eyring-Kramers rates have the form

```text
lambda_+/- (delta,z)
  approximately A_+/- exp[-Delta U_+/-(delta,z)/T],
delta = x_well-q.
```

After a forward crossing, `q` still lies near the old cage.  The elastic force
temporarily favors return; `q` then follows on `tau_q=gamma_q/K`.  Recoil and
finite ordered-path memory therefore arise from a retained continuous
coordinate.  Inside a well, `z` is approximately OU with
`tau_z=gamma_z/k_z`.  Its square changes both the Kramers barrier and local
cage curvature, so rate and cage channels are not independent.

Adiabatic elimination of the fast intra-well coordinate yields a cage-index
process conditioned on `(q,z)`.  Ordinary renewal is recovered only after the
additional approximation that `(q,z)` fully re-equilibrate between escapes.

## Numerical checks

Central finite differences verify all force components against `-grad U`.
Joint integer-period translation leaves energy and forces unchanged.  In the
high-barrier uncoupled limit, the simulated wrapped variance agrees with local
equipartition within the frozen 15% tolerance.  A model with `K>0` retains
unbounded common translation rather than pinning the tagged particle.

Production uses 384 independent trajectories in three dimensions,
`dt=0.002`, 10,000 burn-in steps, 40,000 production steps, record stride 10,
and seed `20260718`.  No design parameter was changed.  The maximum Euler
position increment is below `0.373 L` in every model, and every state remains
finite.

The seed fixes trajectories within one numerical runtime, and the quick
ablation test replays that contract.  Exact event rows are not required to be
bitwise reproducible across operating systems or NumPy builds: tiny
platform-level differences in transcendental forces can move a long stochastic
path across a nearest-well boundary.  Cross-platform package validation instead
recomputes every gate from the committed production rows, verifies the frozen
parameters and zero claim flags, and regenerates the figure exactly from those
rows.  This is an artifact-audit boundary, not an additional physical
tolerance.

## Frozen ablation result

| model | events | count Fano | successive step correlation | persistence/exchange | max NGP |
|---|---:|---:|---:|---:|---:|
| `static_periodic` | 66,790 | 0.879901 | -0.004782 | 0.935988 | 0.461009 |
| `rate_only` | 50,009 | 1.260693 | -0.003019 | 1.139325 | 0.651285 |
| `elastic_only` | 64,712 | 0.888939 | -0.068824 | 0.949379 | 0.455246 |
| `full_transient` | 48,372 | 1.247995 | -0.068495 | 1.332826 | 0.643281 |

Relative to `static_periodic`:

```text
rate_only Fano change                 = +0.380792,
elastic_only step-correlation change = -0.064042,
full Fano change                      = +0.368094,
full step-correlation change         = -0.063713.
```

All frozen directional margins are `0.02`, so the rate-only, elastic-only, and
full joint synthetic gates pass.  This is a constructive result: a continuous
thermal model can generate finite count disorder and ordered recoil separately
and together.  It is not evidence that its hidden coordinates are the ones
present in KA.

## Relation to prior work

[Uneyama, Phys. Rev. E 101, 032106
(2020)](https://doi.org/10.1103/PhysRevE.101.032106) derives Langevin dynamics
with a transient potential from microscopic overdamped Langevin path
probabilities and applies it to a tagged particle in a supercooled liquid.
[Uneyama, Phys. Rev. E 105, 044117
(2022)](https://doi.org/10.1103/PhysRevE.105.044117) derives transient-potential
dynamics through Hamiltonian projection and shows how Markovian potential
coordinates arise after approximation.  These works justify the extended
coordinate structure; they do not establish the specific `q,z` potential or
the KA verdict here.

[Chechkin et al., Phys. Rev. X 7, 021002
(2017)](https://doi.org/10.1103/PhysRevX.7.021002) use squared-OU diffusing
diffusivity to obtain continuous non-Gaussian transport.  Here the squared OU
coordinate modulates an activated barrier and cage curvature instead of only
the Gaussian noise amplitude.

[Hasyim and Mandadapu, PNAS 121, e2322592121
(2024)](https://doi.org/10.1073/pnas.2322592121) retain elastic stresses from
localized rearrangements and obtain facilitation.  The present `q` is a local
single-particle memory analogue.  It contains no spatial field and cannot
support a spatial-facilitation claim.

## What remains unresolved

The synthetic capability does not identify microscopic proxies for `q` and
`z` in real configurations.  It also does not show that KA escape rates
conditioned on those proxies follow the frozen-environment Kramers expression.
The next real-data gate must:

1. infer calibration-only local elastic-offset and barrier/softness proxies;
2. test whether they predict event rate, recoil, and finite exchange;
3. calibrate the continuous model without heldout `k=7.25`;
4. predict heldout MSD, NGP, and multi-k `F_s` for each independent replicate.

Until that gate passes, the result is a synthetic capability plus a real-data
diagnostic rejection of count-only closure.

```text
blind_prediction_claim_allowed = 0
finite_exchange_resolved = 0
static_environment_resolved = 0
spatial_facilitation_resolved = 0
activated_cage_geometry_resolved = 0
transient_potential_identified_in_ka = 0
microdynamic_closure_claim_allowed = 0
thermodynamic_claim_allowed = 0
```
