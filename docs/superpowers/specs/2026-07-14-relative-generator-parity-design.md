# Relative Generator Parity Design

## Objective

Test the generalized-detailed-balance symmetry required before interpreting
the validated discrete Mori identity as a thermal fluctuation-dissipation
relation for the underdamped Langevin parent process.

## Parity Basis

Replace `Lp` by the reversible generator component

```text
a_rev = Lp + gamma p = J F + Hess(u):(V tensor V).
g = [delta u, p, a_rev],
E = diag(+1,-1,+1).
```

This is an invertible linear change of the validated resolved subspace. Test
the equilibrium time-reversal condition `C(t)=E C(t)^T E` on the two
independent validation clones. Use `E=I` as a wrong-parity null.

The necessary-condition gate requires parity defect NRMSE below `0.03`,
maximum absolute defect below `0.01`, and equal-time forbidden even/odd
correlation below `0.01`. It does not by itself close the adjoint random-force
identity or thermal FDT.
