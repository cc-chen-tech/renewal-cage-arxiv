# PRL Memory Closure Independent-Parent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build a parent-first, fail-closed validation pipeline for the particle-conditioned finite-exchange ordered-path kernel and its frozen nested ablations.

**Architecture:** A focused core module validates parent provenance, generates calibration-only block-path surrogates, aggregates restart predictions inside parent trajectories, and classifies the frozen gate. A CLI loads explicit manifests and committed target tables, writes deterministic CSV/SVG artifacts, and leaves the positive claim closed when independent parents or stationarity are missing.

**Tech Stack:** Python 3.12, NumPy, standard-library CSV/JSON/argparse/unittest, and existing ka_replicates trajectory/spectral utilities.

## Global Constraints

- The parent trajectory is the statistical unit; restart, realization, and time-origin counts never increase independent sample count.
- T=0.45 is primary; T=0.58 is canary-only unless all frozen stationarity rows pass.
- Calibration times are 5000 at T=0.45 and 750 at T=0.58, with equal held-out windows and block size 20.
- Score MSD, NGP, and Fs at k={2,4,7.25} on the existing lag grids.
- Curve tolerances are MSD 0.10, NGP 0.30, Fs 0.03; Monte Carlo SE tolerances are MSD 0.01, NGP 0.03, Fs 0.003.
- Use nested realization indices 0..15, extending every evaluated stochastic cell to 0..63 if any precision gate fails.
- Use base seed 20260718; retain spectral-ablation seed 211003.
- No held-out path, event, MSD, NGP, or Fs may enter a model parameter or sampling decision.
- Keep microscopic-closure, spatial-facilitation, and thermodynamic-glass-transition flags zero for every outcome.
- Do not edit or write beneath /Users/luicy/AI/renewal-cage-arxiv; that raw trajectory directory is read-only input.

---

### Task 1: Parent provenance ledger and blocker gate

**Files:**
- Create: src/ka_prl_memory_closure.py
- Create: tests/test_ka_prl_memory_closure.py

**Interfaces:**
- Consumes: rows from data/renewal_cage_ka_replicates_T058_T045_provenance.csv and both frozen stationarity tables.
- Produces: audit_parent_provenance(provenance_rows, stationarity_by_temperature, protocol_by_temperature) returning ledger and blocker rows.

- [ ] **Step 1: Write the failing parent-unit tests**

    def test_parent_audit_counts_shared_restart_parent_once(self):
        ledger, blockers = module.audit_parent_provenance(
            provenance_rows=self.shared_parent_rows(),
            stationarity_by_temperature={
                0.45: self.stationarity(True),
                0.58: self.stationarity(False),
            },
            protocol_by_temperature=module.FROZEN_PROTOCOLS,
        )
        low = next(row for row in blockers if row["temperature"] == 0.45)
        self.assertEqual(len({row["parent_id"] for row in ledger if row["temperature"] == 0.45}), 1)
        self.assertEqual(low["available_parent_count"], 1)
        self.assertEqual(low["missing_parent_count"], 2)
        self.assertEqual(low["blocker_state"], "missing_independent_parents")

    def test_parent_audit_keeps_failed_warm_stationarity_as_canary(self):
        _, blockers = module.audit_parent_provenance(
            provenance_rows=self.shared_parent_rows(),
            stationarity_by_temperature={
                0.45: self.stationarity(True),
                0.58: self.stationarity(False),
            },
            protocol_by_temperature=module.FROZEN_PROTOCOLS,
        )
        warm = next(row for row in blockers if row["temperature"] == 0.58)
        self.assertEqual(warm["blocker_state"], "stationarity_and_independent_parents")
        self.assertEqual(warm["evidence_role"], "canary_only")

- [ ] **Step 2: Run the focused tests and verify RED**

Run: /tmp/renewal-cage-prl-memory-closure.ocGhcz/venv312/bin/python -m unittest tests.test_ka_prl_memory_closure.ParentProvenanceTests -v

Expected: import failure because src/ka_prl_memory_closure.py does not exist.

- [ ] **Step 3: Implement frozen protocols and fail-closed provenance audit**

    FROZEN_PROTOCOLS = {
        0.45: {"required_parent_count": 3, "calibration_time": 5000.0, "heldout_time": 5000.0, "evidence_role": "primary"},
        0.58: {"required_parent_count": 5, "calibration_time": 750.0, "heldout_time": 750.0, "evidence_role": "canary_only"},
    }

    def parent_identifier(row):
        doi = str(row["source_doi"]).strip()
        digest = str(row["source_sha256"]).strip()
        if not doi or len(digest) != 64:
            raise ValueError("parent provenance requires DOI and SHA256")
        return f"{doi}:{digest}"

    def audit_parent_provenance(*, provenance_rows, stationarity_by_temperature, protocol_by_temperature):
        ledger, blockers = [], []
        for temperature, protocol in sorted(protocol_by_temperature.items()):
            selected = [row for row in provenance_rows if float(row["temperature"]) == temperature]
            parents = {parent_identifier(row) for row in selected}
            stationarity_pass = all(float(row["curve_transfer_pass"]) == 1.0 for row in stationarity_by_temperature[temperature])
            ledger.extend(build_parent_ledger_row(row, parent_id=parent_identifier(row), protocol=protocol, stationarity_pass=stationarity_pass) for row in selected)
            missing = max(int(protocol["required_parent_count"]) - len(parents), 0)
            blockers.append(build_parent_blocker_row(temperature, protocol, len(parents), missing, stationarity_pass))
        return ledger, blockers

- [ ] **Step 4: Run the parent-unit tests and verify GREEN**

Run the same ParentProvenanceTests command. Expected: all tests pass.

- [ ] **Step 5: Commit the parent audit**

    git add src/ka_prl_memory_closure.py tests/test_ka_prl_memory_closure.py
    git commit -m "audit PRL parent provenance"

### Task 2: Finite-exchange ordered-path kernel and nested ablations

**Files:**
- Modify: src/ka_prl_memory_closure.py
- Modify: tests/test_ka_prl_memory_closure.py

**Interfaces:**
- Consumes: finite calibration particle x block x 3 arrays and a calibration-only environment e-fold time.
- Produces: generate_ablation_path(blocks, model, environment_time, block_size, rng) returning a path and exact provenance audit.

- [ ] **Step 1: Write failing information-contract tests**

    def test_full_candidate_uses_contiguous_blocks_until_exchange(self):
        path, audit = module.generate_ablation_path(
            self.labelled_blocks(4, 8),
            model="full_candidate",
            environment_time=1.0e12,
            block_size=20.0,
            rng=np.random.default_rng(7),
        )
        self.assertEqual(audit["ordered_path_memory_retained"], 1.0)
        self.assertEqual(audit["finite_exchange_environment_retained"], 1.0)
        self.assertTrue(self.contiguous_within_source_runs(path, audit))

    def test_finite_exchange_ablation_retains_identity_but_destroys_order(self):
        _, audit = module.generate_ablation_path(
            self.labelled_blocks(4, 12),
            model="finite_exchange_environment",
            environment_time=40.0,
            block_size=20.0,
            rng=np.random.default_rng(11),
        )
        self.assertEqual(audit["finite_exchange_environment_retained"], 1.0)
        self.assertEqual(audit["ordered_path_memory_retained"], 0.0)
        self.assertGreater(audit["environment_exchange_count"], 0.0)

- [ ] **Step 2: Run MemoryKernelTests and verify RED**

Run: /tmp/renewal-cage-prl-memory-closure.ocGhcz/venv312/bin/python -m unittest tests.test_ka_prl_memory_closure.MemoryKernelTests -v

Expected: generate_ablation_path is missing.

- [ ] **Step 3: Implement validation, exchange probability, and non-spectral ablations**

    ABLATION_MODELS = (
        "mean_rate_null",
        "one_step_jump_law",
        "static_particle_environment",
        "finite_exchange_environment",
        "full_candidate",
    )

    def exchange_probability(*, block_size, environment_time):
        if min(block_size, environment_time) <= 0.0 or not all(map(math.isfinite, (block_size, environment_time))):
            raise ValueError("block size and environment time must be positive and finite")
        return -math.expm1(-block_size / environment_time)

    def generate_ablation_path(blocks, *, model, environment_time, block_size, rng):
        values = validate_block_paths(blocks)
        if model == "mean_rate_null":
            return gaussian_mean_rate_path(values, rng)
        if model == "one_step_jump_law":
            return pooled_iid_path(values, rng)
        if model == "static_particle_environment":
            return particle_iid_path(values, rng)
        if model in {"finite_exchange_environment", "full_candidate"}:
            return exchange_path(
                values,
                ordered=model == "full_candidate",
                p_exchange=exchange_probability(block_size=block_size, environment_time=environment_time),
                rng=rng,
            )
        raise ValueError("unknown PRL memory ablation")

- [ ] **Step 4: Add terminal-exchange and schedule-independence tests**

    def test_terminal_source_block_forces_recorded_exchange_without_wrap(self):
        _, audit = module.generate_ablation_path(
            self.labelled_blocks(3, 3),
            model="full_candidate",
            environment_time=1.0e12,
            block_size=20.0,
            rng=np.random.default_rng(5),
        )
        self.assertEqual(audit["source_wrap_count"], 0.0)
        self.assertGreater(audit["forced_terminal_exchange_count"], 0.0)

    def test_full_candidate_has_no_shared_global_source_schedule(self):
        _, audit = module.generate_ablation_path(
            self.labelled_blocks(8, 20),
            model="full_candidate",
            environment_time=60.0,
            block_size=20.0,
            rng=np.random.default_rng(19),
        )
        self.assertEqual(audit["global_source_segment_schedule_preserved"], 0.0)

- [ ] **Step 5: Run MemoryKernelTests and verify GREEN**

Run: /tmp/renewal-cage-prl-memory-closure.ocGhcz/venv312/bin/python -m unittest tests.test_ka_prl_memory_closure.MemoryKernelTests -v

Expected: all information-contract tests pass.

- [ ] **Step 6: Commit the model kernel**

    git add src/ka_prl_memory_closure.py tests/test_ka_prl_memory_closure.py
    git commit -m "implement finite exchange ordered path kernel"

### Task 3: Restart, parent, and claim-gate aggregation

**Files:**
- Modify: src/ka_prl_memory_closure.py
- Modify: tests/test_ka_prl_memory_closure.py

**Interfaces:**
- Consumes: per-realization prediction rows and parent-ledger rows.
- Produces: summarize_restarts, summarize_parents, classify_memory_closure_gate, and build_claim_ledger.

- [ ] **Step 1: Write failing parent-first aggregation tests**

    def test_parent_summary_averages_children_before_error(self):
        restarts = module.summarize_restarts(self.two_restart_rows_same_parent())
        parents = module.summarize_parents(restarts, self.parent_ledger())
        self.assertEqual(len(parents), 1)
        self.assertAlmostEqual(parents[0]["predicted_msd"], 2.0)
        self.assertAlmostEqual(parents[0]["msd_relative_error"], 0.0)
        self.assertEqual(parents[0]["child_restart_count"], 2)

    def test_ensemble_average_cannot_rescue_failed_parent(self):
        gate = module.classify_memory_closure_gate(
            parent_summaries=self.parent_rows(one_full_candidate_parent_fails=True),
            blockers=self.no_blockers(),
            upper_control_parents=self.upper_control_rows(pass_all=True),
        )
        self.assertEqual(gate["mechanism_state"], "candidate_rejected")
        self.assertEqual(gate["positive_memory_closure_claim_allowed"], 0.0)

- [ ] **Step 2: Run ParentAggregationTests and verify RED**

Run: /tmp/renewal-cage-prl-memory-closure.ocGhcz/venv312/bin/python -m unittest tests.test_ka_prl_memory_closure.ParentAggregationTests -v

Expected: missing aggregation and gate functions.

- [ ] **Step 3: Implement exact curve and higher-order scoring**

    CURVE_LIMITS = {"msd": 0.10, "ngp": 0.30, "fs": 0.03}
    MC_LIMITS = {"msd": 0.01, "ngp": 0.03, "fs": 0.003}

    def higher_order_score(row):
        fs_errors = [float(value) for key, value in row.items() if key.startswith("absolute_error_fs_k")]
        return max(float(row["ngp_absolute_error"]) / 0.30, max(fs_errors) / 0.03)

    def curve_pass(rows):
        return all(
            float(row["msd_relative_error"]) <= 0.10
            and float(row["ngp_absolute_error"]) <= 0.30
            and max(float(value) for key, value in row.items() if key.startswith("absolute_error_fs_k")) <= 0.03
            for row in rows
        )

- [ ] **Step 4: Implement fail-closed gate ordering**

    def classify_memory_closure_gate(*, parent_summaries, blockers, upper_control_parents):
        result = closed_gate_defaults()
        if any(int(row["missing_parent_count"]) > 0 for row in blockers):
            result["mechanism_state"] = "blocked_independent_parent_validation"
            return result
        if any(row["evidence_role"] != "canary_only" and float(row["stationarity_pass"]) == 0.0 for row in blockers):
            result["mechanism_state"] = "blocked_stationarity_control"
            return result
        if not full_candidate_passes_every_parent(parent_summaries):
            result["mechanism_state"] = "candidate_rejected"
            result["failure_localization"] = localize_candidate_failure(parent_summaries, upper_control_parents)
            return result
        if not required_ablation_pattern_holds(parent_summaries):
            result["mechanism_state"] = "ablation_pattern_unresolved"
            return result
        result["mechanism_state"] = "positive_memory_closure_supported_within_tested_family"
        result["positive_memory_closure_claim_allowed"] = 1.0
        return result

- [ ] **Step 5: Add truth-table tests for all five primary states and all closed claim fields**

    def test_gate_truth_table_is_fail_closed(self):
        for expected, arguments in self.gate_cases():
            with self.subTest(expected=expected):
                gate = module.classify_memory_closure_gate(**arguments)
                self.assertEqual(gate["mechanism_state"], expected)
                self.assertEqual(gate["microdynamic_closure_claim_allowed"], 0.0)
                self.assertEqual(gate["spatial_facilitation_claim_allowed"], 0.0)
                self.assertEqual(gate["thermodynamic_claim_allowed"], 0.0)

Run: /tmp/renewal-cage-prl-memory-closure.ocGhcz/venv312/bin/python -m unittest tests.test_ka_prl_memory_closure.ParentAggregationTests tests.test_ka_prl_memory_closure.MemoryClosureGateTests -v

Expected: all aggregation and truth-table cases pass.

- [ ] **Step 6: Commit parent-first classification**

    git add src/ka_prl_memory_closure.py tests/test_ka_prl_memory_closure.py
    git commit -m "gate PRL closure at parent level"

### Task 4: Recomputable CLI and deterministic artifacts

**Files:**
- Create: scripts/analyze_ka_prl_memory_closure.py
- Modify: tests/test_ka_prl_memory_closure.py
- Create: data/renewal_cage_ka_prl_parent_provenance.csv
- Create: data/renewal_cage_ka_prl_parent_blockers.csv
- Create: data/renewal_cage_ka_prl_memory_closure_restart_rows.csv
- Create: data/renewal_cage_ka_prl_memory_closure_restart_summary.csv
- Create: data/renewal_cage_ka_prl_memory_closure_parent_summary.csv
- Create: data/renewal_cage_ka_prl_memory_closure_model_verdicts.csv
- Create: data/renewal_cage_ka_prl_memory_closure_gate.csv
- Create: data/renewal_cage_ka_prl_memory_closure_claim_ledger.csv
- Create: figures/renewal_cage_ka_prl_memory_closure.svg

**Interfaces:**
- Consumes explicit provenance, stationarity, ensemble-manifest, environment-crossing, and held-out-target paths.
- Produces byte-deterministic CSV/SVG artifacts; supports audit-only and full correlated-parent diagnostic modes.

- [ ] **Step 1: Write failing CLI validation and determinism tests**

    def test_cli_audit_only_writes_exact_parent_blocker(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            cli.main(self.audit_only_arguments(output))
            blocker = next(csv.DictReader((output / "blockers.csv").open()))
            self.assertEqual(blocker["blocker_state"], "missing_independent_parents")
            self.assertEqual(blocker["shared_parent_resampling_can_satisfy"], "0")

    def test_cli_outputs_are_byte_deterministic(self):
        first, second = self.run_synthetic_cli_twice()
        self.assertEqual(
            {path.name: path.read_bytes() for path in first.iterdir()},
            {path.name: path.read_bytes() for path in second.iterdir()},
        )

- [ ] **Step 2: Run MemoryClosureCliTests and verify RED**

Run: /tmp/renewal-cage-prl-memory-closure.ocGhcz/venv312/bin/python -m unittest tests.test_ka_prl_memory_closure.MemoryClosureCliTests -v

Expected: scripts/analyze_ka_prl_memory_closure.py is missing.

- [ ] **Step 3: Implement explicit parser and audit-only path**

    def build_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--provenance", type=Path, required=True)
        parser.add_argument("--low-stationarity", type=Path, required=True)
        parser.add_argument("--high-stationarity", type=Path, required=True)
        parser.add_argument("--audit-only", action="store_true")
        parser.add_argument("--run-temperature", type=float, choices=(0.45, 0.58))
        parser.add_argument("--ensemble-directory", type=Path)
        parser.add_argument("--heldout-targets", type=Path)
        parser.add_argument("--environment-crossings", type=Path)
        parser.add_argument("--realizations", type=int, choices=(16, 64), default=16)
        add_output_arguments(parser)
        return parser

Full mode rejects missing ensemble, target, crossing, restart-row, restart-summary, parent-summary, model-verdict, and SVG paths. Audit-only mode writes parent ledger, blockers, gate, and claim ledger without touching trajectories.

- [ ] **Step 4: Implement calibration-only full mode**

    def run_temperature(spec, *, ensemble_directory, heldout_targets, environment_crossings, realizations):
        manifest = load_and_validate_manifest(ensemble_directory, spec)
        rows = []
        for child in manifest:
            blocks = load_calibration_blocks_only(
                ensemble_directory / child["directory"], spec
            )
            tau_environment = lookup_environment_time(
                environment_crossings,
                temperature=spec.temperature,
                restart=int(child["replicate"]),
                block_size=spec.block_size,
            )
            rows.extend(
                predict_all_ablation_rows(
                    blocks=blocks,
                    targets=heldout_targets,
                    tau_environment=tau_environment,
                    child=child,
                    spec=spec,
                    realizations=realizations,
                )
            )
        return rows

Reuse radial_multivariate_surrogate for two_point_path_spectrum. Join held-out factorization rows only after every calibration prediction exists.

- [ ] **Step 5: Run synthetic CLI tests and verify GREEN**

Run: /tmp/renewal-cage-prl-memory-closure.ocGhcz/venv312/bin/python -m unittest tests.test_ka_prl_memory_closure.MemoryClosureCliTests -v

Expected: manifest validation, held-out exclusion, exact schemas, parent grouping, and byte determinism pass.

- [ ] **Step 6: Write committed audit-only artifacts**

Run:

    /tmp/renewal-cage-prl-memory-closure.ocGhcz/venv312/bin/python scripts/analyze_ka_prl_memory_closure.py --audit-only --provenance data/renewal_cage_ka_replicates_T058_T045_provenance.csv --low-stationarity data/renewal_cage_ka_replicates_T045_nonlinear_path_stationarity.csv --high-stationarity data/renewal_cage_ka_replicates_T058_block20_nonlinear_path_stationarity.csv --output-parent-ledger data/renewal_cage_ka_prl_parent_provenance.csv --output-blockers data/renewal_cage_ka_prl_parent_blockers.csv --output-gate data/renewal_cage_ka_prl_memory_closure_gate.csv --output-claim-ledger data/renewal_cage_ka_prl_memory_closure_claim_ledger.csv

Expected: T=0.45 one available/three required parents; T=0.58 one available/five required parents and failed stationarity; positive claim zero.

- [ ] **Step 7: Run the T=0.45 correlated-parent diagnostic**

Run:

    /tmp/renewal-cage-prl-memory-closure.ocGhcz/venv312/bin/python scripts/analyze_ka_prl_memory_closure.py --provenance data/renewal_cage_ka_replicates_T058_T045_provenance.csv --low-stationarity data/renewal_cage_ka_replicates_T045_nonlinear_path_stationarity.csv --high-stationarity data/renewal_cage_ka_replicates_T058_block20_nonlinear_path_stationarity.csv --run-temperature 0.45 --ensemble-directory /Users/luicy/AI/renewal-cage-arxiv/tmp/ka_replicates/T045 --heldout-targets data/renewal_cage_ka_replicates_T045_event_oracle_factorization_rows.csv --environment-crossings data/renewal_cage_ka_debye_waller_environment_crossover_crossings.csv --realizations 16 --output-parent-ledger data/renewal_cage_ka_prl_parent_provenance.csv --output-blockers data/renewal_cage_ka_prl_parent_blockers.csv --output-restart-rows data/renewal_cage_ka_prl_memory_closure_restart_rows.csv --output-restart-summary data/renewal_cage_ka_prl_memory_closure_restart_summary.csv --output-parent-summary data/renewal_cage_ka_prl_memory_closure_parent_summary.csv --output-model-verdicts data/renewal_cage_ka_prl_memory_closure_model_verdicts.csv --output-gate data/renewal_cage_ka_prl_memory_closure_gate.csv --output-claim-ledger data/renewal_cage_ka_prl_memory_closure_claim_ledger.csv --output-svg figures/renewal_cage_ka_prl_memory_closure.svg

If any precision field fails, rerun the identical command with --realizations 64. The gate remains blocked regardless of curves.

- [ ] **Step 8: Commit CLI and artifacts**

    git add scripts/analyze_ka_prl_memory_closure.py tests/test_ka_prl_memory_closure.py data/renewal_cage_ka_prl_*.csv figures/renewal_cage_ka_prl_memory_closure.svg
    git commit -m "record parent blocked PRL memory closure diagnostic"

### Task 5: Result note and package recomputation test

**Files:**
- Create: docs/prl-memory-closure-independent-parent-result.md
- Modify: README.md
- Modify: tests/test_arxiv_package.py

**Interfaces:**
- Consumes committed parent ledger, blockers, parent summaries, gate, and claim ledger.
- Produces manuscript-safe result wording and a package test that recomputes the gate instead of trusting pass flags.

- [ ] **Step 1: Write the failing package recomputation test**

    def test_prl_memory_closure_is_parent_gated_and_claim_limited(self):
        data = ROOT / "data"
        parent_rows = list(csv.DictReader((data / "renewal_cage_ka_prl_parent_provenance.csv").open()))
        stored = next(csv.DictReader((data / "renewal_cage_ka_prl_memory_closure_gate.csv").open()))
        recomputed = recompute_committed_memory_closure_gate(data)
        self.assertCsvGateMatchesComputedGate(stored, recomputed)
        self.assertEqual(len({row["parent_id"] for row in parent_rows if float(row["temperature"]) == 0.45}), 1)
        self.assertEqual(stored["mechanism_state"], "blocked_independent_parent_validation")
        self.assertEqual(float(stored["positive_memory_closure_claim_allowed"]), 0.0)
        self.assertTrue(all(float(stored[key]) == 0.0 for key in CLOSED_CLAIM_FIELDS))

- [ ] **Step 2: Run that one package test and verify RED**

Run: /tmp/renewal-cage-prl-memory-closure.ocGhcz/venv312/bin/python -m unittest tests.test_arxiv_package.ArxivPackageTests.test_prl_memory_closure_is_parent_gated_and_claim_limited -v

Expected: recomputation imports and assertions are absent.

- [ ] **Step 3: Add recomputation import and exact stored-versus-computed assertions**

    from analyze_ka_prl_memory_closure import recompute_committed_memory_closure_gate
    from ka_prl_memory_closure import CLOSED_CLAIM_FIELDS

Add the test from Step 1. Also copy the gate row, set
positive_memory_closure_claim_allowed to 1, and assert that it differs from the
recomputed gate.

- [ ] **Step 4: Write result note and README boundary**

Add one README paragraph linking docs/prl-memory-closure-independent-parent-result.md.
The result note reports historical evidence, diagnostic model result,
independent-parent blocker, T=0.58 stationarity canary, focused tests, full local
tests, package rebuild, and remote CI separately. It states the exact missing
parent counts and never enables the candidate claim.

- [ ] **Step 5: Run focused integration tests and verify GREEN**

Run:

    /tmp/renewal-cage-prl-memory-closure.ocGhcz/venv312/bin/python -m unittest tests.test_ka_prl_memory_closure -v
    /tmp/renewal-cage-prl-memory-closure.ocGhcz/venv312/bin/python -m unittest tests.test_arxiv_package.ArxivPackageTests.test_prl_memory_closure_is_parent_gated_and_claim_limited -v

Expected: all focused tests pass.

- [ ] **Step 6: Commit documentation and integration**

    git add docs/prl-memory-closure-independent-parent-result.md README.md tests/test_arxiv_package.py
    git commit -m "document blocked independent parent closure result"

### Task 6: Full verification and draft PR

**Files:**
- Verify every file changed by Tasks 1-5.

**Interfaces:**
- Produces fresh local validation evidence, a clean diff, pushed branch, and a draft PR separating scientific, engineering, and CI states.

- [ ] **Step 1: Run fresh focused tests**

Run: /tmp/renewal-cage-prl-memory-closure.ocGhcz/venv312/bin/python -m unittest tests.test_ka_prl_memory_closure -v

Expected: zero failures/errors.

- [ ] **Step 2: Run the complete local suite**

Run: /tmp/renewal-cage-prl-memory-closure.ocGhcz/venv312/bin/python -m unittest discover -s tests -v

Expected: zero failures/errors and record the exact test count.

- [ ] **Step 3: Rebuild the arXiv package**

Run: /tmp/renewal-cage-prl-memory-closure.ocGhcz/venv312/bin/python scripts/build_arxiv_package.py

Expected: exit zero.

- [ ] **Step 4: Run syntax and diff checks**

    python -m py_compile src/ka_prl_memory_closure.py scripts/analyze_ka_prl_memory_closure.py tests/test_ka_prl_memory_closure.py
    git diff --check
    git status -sb
    git diff origin/main...HEAD --stat

Expected: only intended changes; no syntax or whitespace error.

- [ ] **Step 5: Push and create a draft PR**

Push codex/prl-memory-closure-independent-parents and open a draft PR against main. The body states two missing T=0.45 parents, four missing T=0.58 parents, failed T=0.58 stationarity, all model/ablation failures, exact local checks, and remote CI as a separate pending/current state.

- [ ] **Step 6: Inspect remote CI and keep the PR draft**

Even if Actions passes, the scientific gate remains blocked and the PR stays draft.
