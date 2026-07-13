# C3-Switched KA Tangent Control Design

## Objective

Remove the hard-cutoff differentiability failure from the 4096-particle
Kob-Andersen common-noise response experiment without replacing the
many-particle Langevin dynamics by an effective stochastic model. The control
must test the full microscopic tangent-noise covariance over the complete
saved horizon and must preserve `thermodynamic_claim_allowed = 0`.

This is a mechanism control, not the final event-clock derivation. Its purpose
is to decide whether the current long-horizon failure comes from the
non-smooth `lj/cut` force or from a missing term in the microscopic tangent
equation.

## Potential

For each KA pair type, retain the usual Lennard-Jones energy

```text
V(r) = 4 epsilon [(sigma/r)^12 - (sigma/r)^6]
```

up to `r_on = 2.0 sigma`. Between `r_on` and `r_c = 2.5 sigma`, use

```text
U(r) = V(r) S(x)
x = (r - r_on) / (r_c - r_on)
S(x) = 1 - 35 x^4 + 84 x^5 - 70 x^6 + 20 x^7.
```

Set `U(r)=0` for `r >= r_c`. At both switch endpoints, the first three
derivatives of `S` vanish. Consequently `U`, `U'`, `U''`, and `U'''` match the
unswitched LJ branch at `r_on` and vanish continuously at `r_c`. This is the
minimum practical smoothness needed here: `F` uses `U'`, the pair Hessian uses
`U''`, and `L2F` contains `U'''`.

The radial derivatives used by the Python generator are

```text
U'   = V' S + V S'
U''  = V'' S + 2 V' S' + V S''
U''' = V''' S + 3 V'' S' + 3 V' S'' + V S'''.
```

Derivatives of `S` with respect to `r` include the corresponding powers of
`1 / (r_c-r_on)`.

## LAMMPS Realization

Build a second serial LAMMPS executable from the pinned
`stable_22Jul2025_update4` source with `PKG_LEPTON=ON`. Use `pair_style lepton`
with one explicit expression and pair cutoff for each KA pair:

| pair | epsilon | sigma | r_on | r_c |
|---|---:|---:|---:|---:|
| AA | 1.0 | 1.00 | 2.00 | 2.50 |
| AB | 1.5 | 0.80 | 1.60 | 2.00 |
| BB | 0.5 | 0.88 | 1.76 | 2.20 |

Lepton analytically differentiates the energy expression to obtain the force.
The expression uses the septic switch directly; no independently interpolated
force table is permitted. A tabulated potential is rejected for this control
because separate interpolation of energy and force would weaken exact
LAMMPS/Python derivative parity.

## Equilibrated Parent

Read the existing KA `T=0.58` restart, replace the pair style and coefficients,
and run a dedicated switched-potential equilibration before any response
displacement. Use unit mass, `dt=0.001`, `gamma=1`, and the same temperature.
The equilibration manifest records the source restart SHA-256, Lepton binary
SHA-256, pair expressions, seed, duration, and output restart SHA-256.

The first canary may use `10 tau`. This duration is accepted only for the
tangent-equation control; it does not establish equilibrium glass observables.
Energy, temperature, and force finiteness are mandatory. A later physical
comparison must extend equilibration and compare structural and dynamical
observables before the switched system is used for event-clock claims.

## Python Microscopic Generator

Keep existing hard-cutoff APIs behaviorally unchanged. Add an explicit
potential protocol to the generator-response extraction boundary and implement
the switched radial derivatives in `src/ka_local_cage.py`.

For central pair potentials, construct

```text
F_ij = -U'(r) n
H_ij = (U'' - U'/r) n n^T + (U'/r) I.
```

Use the same microscopic identities as the hard-cutoff experiment:

```text
LF_i = -sum_j H_ij (v_i-v_j)
d(delta LF) = delta(L2F) dt - sqrt(2 gamma T) delta H dW.
```

`L2F` remains a deterministic generator drift evaluated from the full particle
configuration and velocities. Its directional finite difference must use the
same switched potential on both displaced configurations.

## Validation Ladder

1. Analytic switch endpoint test: values and first three derivatives match at
   `r_on` and vanish at `r_c`.
2. Radial finite-difference test: analytic `U'`, `U''`, and `U'''` agree with
   derivatives of the preceding order away from the two endpoints.
3. Tiny-system force parity: LAMMPS Lepton forces and Python forces agree for
   AA, AB, and BB separations in the LJ, switching, and near-cutoff regions.
4. Generator parity: finite trajectory derivatives agree with `LF`; the
   deterministic drift and tangent covariance use the switched Hessians.
5. Full 8-member response: two epsilons, common seeds, `dt=0.001`, saved every
   `0.001`, duration `0.2 tau`, and no right censoring.

## Decision Gates

The smooth control passes only if all of the following hold:

- no non-finite values and no potential-protocol mismatch;
- all `320` intervals per epsilon remain eligible;
- memberwise covariance responses are cross-epsilon stable;
- pooled covariance trace-variance ratio lies in `[0.8, 1.2]`;
- pooled mean squared Mahalanobis lies in `[2.4, 3.6]`;
- absolute whitened lag-1 correlation is below `0.1`;
- the four tangent kinematic identities improve or converge with frame
  refinement rather than relying on a single saved interval.

If these gates pass, the multiplicative tangent-noise term is validated over
the complete `0.2 tau` horizon for the smooth many-particle control. If they
fail, the failure is retained and localized by force parity, derivative order,
epsilon, member, and frame resolution. Neither outcome permits a
thermodynamic claim or by itself derives the renewal hazard.

## Follow-On Connection to the Effective Theory

After this control, the next reduction is not another free stochastic fit. It
is a projection of the validated microscopic tagged-particle response onto
cage coordinates and precursor observables. The eventual single-particle
Langevin representation must inherit its drift, multiplicative noise, memory,
and escape statistics from that projection and must still predict held-out
MSD, NGP, multi-k scattering, persistence/exchange, and transport.
