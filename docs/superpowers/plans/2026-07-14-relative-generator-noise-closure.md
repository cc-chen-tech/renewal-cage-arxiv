# Relative Generator Noise Closure Plan

1. Add an exact finite-memory residual helper and synthetic AR test.
2. Implement moving-block innovation propagation with `B=1` as the iid null.
3. Select the shortest passing block by discovery-clone leave-one-out scans.
4. Freeze the selected block before using independent validation clones.
5. Repeat high-statistics validation with disjoint Monte Carlo seeds.
6. Package code, data, tests, reproduction commands, and strict claim flags.
7. Keep thermal FDT and complete single-particle Langevin claims closed.
