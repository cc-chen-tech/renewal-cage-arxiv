# PRL acquisition completion and ingestion implementation plan

1. Add failing tests for passive completion snapshots: unknown exit, explicit
   success, explicit failure, hashes, frames, restarts, and error signatures.
2. Implement the completion core and credential-free CLI; run focused tests.
3. Deploy the watcher through the authenticated SSH control connection and
   commit the resulting non-sensitive completion artifact.
4. Add failing tests for completion validation, parent-keyed trajectory split,
   held-out targets, per-parent stationarity, and the exact six-family/64-grid
   orchestration contract.
5. Implement the one-click acquisition import CLI by reusing the frozen model,
   stationarity, trajectory, and gate functions.  Exercise its blocker path on
   the completed remote runs and its full path on synthetic fixtures.
6. Run focused tests, the full Python 3.12 suite, artifact byte rebuild, package
   build, and git diff checks; commit and push while PR #20 remains draft.
