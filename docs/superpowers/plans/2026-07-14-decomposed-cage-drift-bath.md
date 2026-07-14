# Decomposed Cage-Drift Bath Implementation Plan

1. Validate batched `b_u`, `b_C`, and joint multiplicative-noise identities
   against the scalar smooth-cage projection.
2. Add fixed-step sensitivity diagnostics for the geometric drift.
3. Cache `u,p,w,b_u,b_C` and covariance geometry from all four hash-validated
   long trajectories.
4. Fit raw-H16, split-8, and split-16 leave-one-clone-out baths using only
   microscopic histories.
5. Reconstruct autonomous tagged displacement from `w+p`, apply residual and
   macro gates, document the verdict, and checkpoint exact artifacts.
