# Microscopic Relative Generator Parity Audit

## Why This Audit Is Needed

The generator-augmented matrix Mori model passed its projection-consistent
discrete noise/operator identity on two independent validation clones. That
identity is not automatically the equilibrium second fluctuation-dissipation
theorem for a stochastic Langevin thermostat. A thermal interpretation also
requires the correct invariant inner product, adjoint dynamics, and momentum
time-reversal operation.

For the underdamped parent process, define

```text
Theta(R,V) = (R,-V).
```

The previously used generator coordinate

```text
Lp = J F + Hess(u):(V tensor V) - gamma p
```

mixes an even reversible acceleration and the odd irreversible drag. Since
`p` is already resolved, the same linear subspace can be written in the
parity-definite basis

```text
a_rev = Lp + gamma p = J F + Hess(u):(V tensor V),
g = [u-u0, p, a_rev],
E = diag(+1,-1,+1).
```

Here `u-u0` and `a_rev` are even, while `p` is odd.

## Detailed-Balance Necessary Condition

For an equilibrium process satisfying generalized detailed balance,

```text
P_t^dagger = Theta P_t Theta.
```

Parity-definite observables must therefore obey

```text
C(t) = E C(t)^T E,
C_ij(t) = epsilon_i epsilon_j C_ji(t).
```

This condition is tested directly through `8 tau` on the two independent
validation clones. The cage bias, mean, and coordinate scales are fixed from
the original four training clones. The validation trajectories are not used
for fitting. An all-even matrix `E=I` is evaluated as a wrong-parity null.

## Results

| Metric | validation 1 | validation 2 | gate |
|---|---:|---:|---:|
| parity defect NRMSE | 0.01866 | 0.02192 | <= 0.03 |
| maximum absolute defect | 0.00765 | 0.00871 | <= 0.01 |
| equal-time forbidden even/odd correlation | 0.00074 | 0.00078 | <= 0.01 |
| wrong all-even parity NRMSE | 1.07727 | 1.09790 | >= 0.50 |

The correct `[+,-,+]` parity passes every gate. The all-even null fails by
more than a factor of 35 in normalized defect, so the pass is not caused by
all cross-correlations being small. In particular, the large antisymmetric
position-momentum correlations are assigned the correct odd sign.

This provides independent numerical support that the resolved microscopic
subspace respects the time-reversal symmetry required by equilibrium
underdamped Langevin dynamics. It also identifies the proper parity basis for
the next adjoint Mori calculation.

## Remaining Thermal-FDT Step

This checkpoint is a necessary-condition result, not yet a complete thermal
FDT derivation. For a general semigroup the memory operator involves the
adjoint generator acting on the resolved observable. Replacing that adjoint
force by an ordinary forward-noise autocorrelation requires an additional
skew-adjoint or generalized-detailed-balance argument. The stochastic
thermostat also contributes an instantaneous dissipative/noise sector that
must be separated from the conservative orthogonal memory.

The next calculation must therefore construct forward and parity-adjoint
discrete Mori noises and test the corresponding cross-correlation identity.
Only after that passes can `thermal_fdt_adjoint_audit_pass` change to one.

The caution is consistent with the general semigroup derivation of Widder,
Zimmer, and Schilling,
[J. Phys. A 58, 405001 (2025)](https://doi.org/10.1088/1751-8121/ae02cc),
and with the non-Hamiltonian projection analysis of Xing and Kim,
[J. Chem. Phys. 134, 044132 (2011)](https://doi.org/10.1063/1.3530071).

The claim boundary is

```text
parity_definite_generator_basis_allowed = 1
resolved_generalized_detailed_balance_supported = 1
wrong_all_even_parity_rejected = 1
thermal_fdt_adjoint_audit_pass = 0
physical_relative_generator_gle_allowed = 0
orthogonal_noise_generation_closed = 0
autonomous_single_particle_gle_allowed = 0
complete_event_clock_closure_allowed = 0
kramers_escape_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

## Reproduction

```bash
python scripts/analyze_ka_relative_generator_parity.py \
  --training-drift-cache-directory tmp/decomposed_cage_drift_reduced_T058 \
  --validation-drift-cache-directory tmp/relative_generator_mori_validation_drift_T058 \
  --output-prefix data/renewal_cage_ka_relative_generator_parity_validation_T058
```
