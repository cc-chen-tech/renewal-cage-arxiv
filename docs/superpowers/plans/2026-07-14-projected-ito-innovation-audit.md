# Projected Ito Innovation Audit Plan

1. Add dimension-agnostic covariance whitening and nonoverlapping Ito-block
   construction with synthetic calibration tests.
2. Reconstruct `Q_C`, `Q_u`, and `Q_Cu` from the smooth-cage Jacobian geometry
   on all four trajectories.
3. Combine those covariance paths with cached exact `w,p,b_C,b_u` paths.
4. Evaluate left-point and trapezoid estimators at strides 1, 2, 4, and 8.
5. Apply the preregistered local gate, document the claim boundary, and commit
   exact artifacts.
