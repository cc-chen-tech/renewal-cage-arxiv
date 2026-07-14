# Bilinear State-Dependent Memory Plan

1. Add synthetic tests where a known state-dependent bath transition beats a
   stationary linear model on held data.
2. Implement invariant fitting, bilinear ridge propagation, and held residual
   state/lag diagnostics.
3. Reuse the verified reduced exact-force caches and repeat leave-one-clone-out
   rank-16 basis fitting without touching macro observables.
4. Apply the preregistered residual-whitening promotion gate.
5. If the gate passes, derive a covariance-stable autonomous mixture and test
   `D`, NGP, multi-k `F_s`, and p_hop.  If it fails, document the rejected
   invariant class and move to learned cage-relative features.

