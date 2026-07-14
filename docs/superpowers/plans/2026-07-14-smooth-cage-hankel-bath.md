# Smooth-Cage Hankel Bath Implementation Plan

1. Add a batched smooth-cage extractor and compare it against the existing
   scalar analytic Jacobian implementation.
2. Add tests for translation/rotation covariance, positive noise Gram
   matrices, and the cage-relative velocity identity.
3. Build a cache-validated long-trajectory CLI using the same four clones and
   64 fixed A particles as the exact-force Hankel experiment.
4. Compare `H16`, `H16+u`, and `H16+up` with leave-one-clone-out residual and
   autonomous macro/event diagnostics.
5. Apply preregistered gates, document the result and claim boundary, run the
   full test suite, and checkpoint only the exact new files.
