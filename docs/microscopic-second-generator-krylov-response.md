# Microscopic Second-Generator Krylov Response

## Question

Can one more observable from the exact many-particle Langevin generator turn
the tagged-particle response into a predictive autonomous closure?  The test
extends the resolved state from

```text
Z1 = (delta x, delta v, delta F, delta L F)
```

to

```text
Z2 = (delta x, delta v, delta F, delta L F, delta L2 F).
```

This is a microscopic response test, not a fit to diffusion, NGP, scattering,
event statistics, or thermodynamics.

## Microscopic Hierarchy

For the C3-switched Kob-Andersen Langevin system with unit mass,

```text
d r_i = v_i dt
d v_i = [F_i(R) - gamma v_i] dt + sqrt(2 gamma T) dW_i.
```

Writing `G_i = L F_i` and `H_i = L2 F_i`, the matched common-noise tangent
response obeys

```text
d(delta x) = delta v dt
d(delta v) = [delta F - gamma delta v] dt
d(delta F) = delta G dt
d(delta G) = delta H dt + dM_G.
```

The force and first two generator observables are evaluated from the full
4096-particle configuration.  Only the next unresolved drift is projected:

```text
d(delta H)/dt = B [delta x, delta v, delta F, delta G, delta H] + R3.
```

The `3 x 15` block `B` is fitted in integrated weak form,

```text
delta H_(n+1) - delta H_n
  = dt B [Z2_(n+1) + Z2_n] / 2 + rho_n.
```

Thus the first four rows remain fixed by microscopic dynamics.  This test asks
whether a finite local generator Krylov chain is the missing bath state.

## Data And Holdout Protocol

The fresh panel contains 32 matched paths: eight independent seed pairs,
two perturbation amplitudes (`0.001`, `0.002`), and matched plus/minus paths.
All paths use `T=0.58`, `gamma=1`, target id 821, integration step `0.001 tau`,
saved interval `0.005 tau`, duration `1.0 tau`, and the smooth
`ka_lj_c3_switch` potential.  Every manifest hash passed and all paths contain
201 frames.  No macroscopic observable is fitted.

Each leave-one-member-out fold fits the other seven members at `epsilon=0.001`.
Fits at `0.05`, `0.10`, and `0.20 tau` are reported, but the preregistered
claim gate uses only the `0.20 tau` fit.  Both epsilon values are held-out
evaluations.  The maximum cross-epsilon mismatch at the primary `0.20 tau`
horizon is `0.00023`, well below the `0.02` linearity limit, so all eight folds
are identified at both amplitudes.

## Held-Out Result

At `epsilon=0.001`, the primary aggregate position-response errors are:

| Model | `0.20 tau` | `0.50 tau` | `1.00 tau` |
|---|---:|---:|---:|
| first-generator constrained | 0.17569 | 0.83409 | 6.19303 |
| second-generator constrained | 0.23290 | 1.28756 | 20.39776 |
| free 15-state transition control | 0.18291 | 0.63607 | 2.29121 |

The second-generator model therefore worsens the primary mean error by
61.87 percent instead of delivering the required 20 percent improvement.  Its
maximum fold error is 0.38104 at `0.20 tau`, above the 0.20 gate.  The maximum
held residual-state correlation reaches 0.86643, far above the 0.20 gate.  The
same verdict is reproduced at `epsilon=0.002` (`0.23291` mean error), so the
failure is not caused by leaving the linear-response regime.

The fitted transitions are also unstable under autonomous propagation.  The
free transition control reduces the longer-horizon error relative to the
generator-constrained models but still fails every preregistered threshold.
It therefore does not supply evidence for a physical autonomous closure.

## Verdict

The exact `L2 F` observable is measurable and the weak design is full rank,
but closing its next action by a time-local linear projection does not extend
the held-out response horizon.  A short finite-order generator chain is not
the missing microscopic state in this protocol.  The next justified branch
is an explicit slow bath or state-dependent memory, such as time-lagged
collective-force modes or a position-dependent nonlinear GLE.  Blindly adding
`L3 F`, `L4 F`, and higher local derivatives is not supported by these data.

```text
second_generator_response_allowed = 0
one_tau_generator_response_allowed = 0
autonomous_stochastic_single_particle_gle_allowed = 0
event_clock_claim_allowed = 0
kramers_escape_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

The result is a microscopic exclusion boundary.  It does not reject the
cage-event physical picture; it rejects this particular finite, linear,
time-local route from many-particle Langevin dynamics to an autonomous tagged
particle model.
