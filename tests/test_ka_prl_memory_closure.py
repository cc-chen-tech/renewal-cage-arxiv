import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ka_prl_memory_closure as closure


class ParentProvenanceTests(unittest.TestCase):
    def provenance_rows(self):
        rows = []
        for temperature, digest, frames in (
            (0.45, "a" * 64, (5000, 35000, 65000)),
            (0.58, "b" * 64, (3000, 8000, 13000, 18000, 23000)),
        ):
            for replicate, frame in enumerate(frames, start=1):
                rows.append(
                    {
                        "temperature": str(temperature),
                        "replicate": str(replicate),
                        "source_doi": "10.5281/zenodo.7469766",
                        "source_sha256": digest,
                        "source_frame_index": str(frame),
                        "velocity_seed": str(45000 + replicate),
                        "production_time_tau": "10000" if temperature == 0.45 else "1500",
                        "independence_class": "decorrelated_parent_frames_plus_velocity_seeds",
                        "independently_prepared_parent_samples": "False",
                    }
                )
        return rows

    @staticmethod
    def stationarity(passing):
        return [
            {"comparison": comparison, "curve_transfer_pass": str(float(passing))}
            for comparison in ("early_late", "early_heldout", "late_heldout")
        ]

    def test_parent_audit_counts_shared_restart_parent_once(self):
        ledger, blockers = closure.audit_parent_provenance(
            provenance_rows=self.provenance_rows(),
            stationarity_by_temperature={
                0.45: self.stationarity(True),
                0.58: self.stationarity(False),
            },
        )

        low_ledger = [row for row in ledger if row["temperature"] == 0.45]
        low_blocker = next(row for row in blockers if row["temperature"] == 0.45)
        self.assertEqual({row["parent_id"] for row in low_ledger}, {f"10.5281/zenodo.7469766:{'a' * 64}"})
        self.assertEqual(low_blocker["available_parent_count"], 1)
        self.assertEqual(low_blocker["missing_parent_count"], 2)
        self.assertEqual(low_blocker["blocker_state"], "missing_independent_parents")
        self.assertEqual(sum(row["parent_unit_contribution"] for row in low_ledger), 1)
        self.assertTrue(all(row["restart_is_independent_sample"] == 0.0 for row in low_ledger))

    def test_parent_audit_keeps_failed_warm_stationarity_as_canary(self):
        _, blockers = closure.audit_parent_provenance(
            provenance_rows=self.provenance_rows(),
            stationarity_by_temperature={
                0.45: self.stationarity(True),
                0.58: self.stationarity(False),
            },
        )

        warm = next(row for row in blockers if row["temperature"] == 0.58)
        self.assertEqual(warm["blocker_state"], "stationarity_and_independent_parents")
        self.assertEqual(warm["evidence_role"], "canary_only")
        self.assertEqual(warm["available_parent_count"], 1)
        self.assertEqual(warm["missing_parent_count"], 4)
        self.assertEqual(warm["stationarity_pass"], 0.0)

    def test_parent_audit_rejects_malformed_parent_hash_and_missing_comparison(self):
        rows = self.provenance_rows()
        rows[0]["source_sha256"] = "not-a-sha"
        with self.assertRaisesRegex(ValueError, "SHA256"):
            closure.audit_parent_provenance(
                provenance_rows=rows,
                stationarity_by_temperature={
                    0.45: self.stationarity(True),
                    0.58: self.stationarity(False),
                },
            )

        with self.assertRaisesRegex(ValueError, "stationarity comparisons"):
            closure.audit_parent_provenance(
                provenance_rows=self.provenance_rows(),
                stationarity_by_temperature={
                    0.45: self.stationarity(True)[:2],
                    0.58: self.stationarity(False),
                },
            )


class MemoryKernelTests(unittest.TestCase):
    @staticmethod
    def labelled_blocks(particles=4, block_count=12):
        blocks = np.zeros((particles, block_count, 3), dtype=float)
        for particle in range(particles):
            for block in range(block_count):
                blocks[particle, block] = (
                    1000.0 * particle + block,
                    10.0 * particle + block,
                    particle - block,
                )
        return blocks

    def test_full_candidate_uses_contiguous_blocks_until_exchange(self):
        _, audit = closure.generate_ablation_path(
            self.labelled_blocks(),
            model="full_candidate",
            environment_time=1.0e12,
            block_size=20.0,
            rng=np.random.default_rng(7),
        )

        sources = audit["source_particle"]
        source_blocks = audit["source_block"]
        for particle in range(sources.shape[0]):
            for index in range(1, sources.shape[1]):
                if sources[particle, index] == sources[particle, index - 1]:
                    self.assertEqual(
                        source_blocks[particle, index],
                        source_blocks[particle, index - 1] + 1,
                    )
        self.assertEqual(audit["ordered_path_memory_retained"], 1.0)
        self.assertEqual(audit["finite_exchange_environment_retained"], 1.0)
        self.assertEqual(audit["source_wrap_count"], 0.0)

    def test_finite_exchange_ablation_retains_identity_but_destroys_order(self):
        _, audit = closure.generate_ablation_path(
            self.labelled_blocks(block_count=40),
            model="finite_exchange_environment",
            environment_time=20.0,
            block_size=20.0,
            rng=np.random.default_rng(11),
        )

        self.assertEqual(audit["finite_exchange_environment_retained"], 1.0)
        self.assertEqual(audit["ordered_path_memory_retained"], 0.0)
        self.assertGreater(audit["environment_exchange_count"], 0.0)

    def test_static_environment_never_changes_source_particle(self):
        _, audit = closure.generate_ablation_path(
            self.labelled_blocks(),
            model="static_particle_environment",
            environment_time=40.0,
            block_size=20.0,
            rng=np.random.default_rng(13),
        )

        expected = np.broadcast_to(
            np.arange(4, dtype=int)[:, None], audit["source_particle"].shape
        )
        np.testing.assert_array_equal(audit["source_particle"], expected)
        self.assertEqual(audit["environment_exchange_count"], 0.0)
        self.assertEqual(audit["static_particle_environment_retained"], 1.0)
        self.assertEqual(audit["ordered_path_memory_retained"], 0.0)

    def test_terminal_source_block_forces_recorded_exchange_without_wrap(self):
        _, audit = closure.generate_ablation_path(
            self.labelled_blocks(particles=3, block_count=8),
            model="full_candidate",
            environment_time=1.0e12,
            block_size=20.0,
            rng=np.random.default_rng(5),
        )

        self.assertEqual(audit["source_wrap_count"], 0.0)
        self.assertGreater(audit["forced_terminal_exchange_count"], 0.0)

    def test_full_candidate_has_no_shared_global_source_schedule(self):
        _, audit = closure.generate_ablation_path(
            self.labelled_blocks(particles=8, block_count=20),
            model="full_candidate",
            environment_time=60.0,
            block_size=20.0,
            rng=np.random.default_rng(19),
        )

        self.assertEqual(audit["global_source_segment_schedule_preserved"], 0.0)
        self.assertTrue(
            any(
                not np.array_equal(
                    audit["source_particle"][0],
                    audit["source_particle"][particle],
                )
                for particle in range(1, 8)
            )
        )

    def test_mean_rate_and_one_step_ablation_information_flags_are_exact(self):
        blocks = self.labelled_blocks()
        for model, one_step in (("mean_rate_null", 0.0), ("one_step_jump_law", 1.0)):
            with self.subTest(model=model):
                generated, audit = closure.generate_ablation_path(
                    blocks,
                    model=model,
                    environment_time=40.0,
                    block_size=20.0,
                    rng=np.random.default_rng(23),
                )
                self.assertEqual(generated.shape, blocks.shape)
                self.assertTrue(np.all(np.isfinite(generated)))
                self.assertEqual(audit["one_step_jump_law_retained"], one_step)
                self.assertEqual(audit["particle_identity_retained"], 0.0)
                self.assertEqual(audit["ordered_path_memory_retained"], 0.0)

    def test_invalid_kernel_controls_fail_closed(self):
        blocks = self.labelled_blocks()
        with self.assertRaisesRegex(ValueError, "unknown"):
            closure.generate_ablation_path(
                blocks,
                model="posthoc_model",
                environment_time=40.0,
                block_size=20.0,
                rng=np.random.default_rng(1),
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            closure.generate_ablation_path(
                blocks,
                model="full_candidate",
                environment_time=0.0,
                block_size=20.0,
                rng=np.random.default_rng(1),
            )


class ParentAggregationTests(unittest.TestCase):
    @staticmethod
    def realization_row(*, restart, realization, predicted_msd):
        return {
            "temperature": 0.45,
            "restart": restart,
            "model": "full_candidate",
            "realization": realization,
            "lag": 20.0,
            "predicted_msd": predicted_msd,
            "predicted_ngp": 0.25,
            "predicted_fs_k2": 0.8,
            "predicted_fs_k4": 0.6,
            "predicted_fs_k7p25": 0.4,
            "target_msd": 2.0,
            "target_ngp": 0.25,
            "target_fs_k2": 0.8,
            "target_fs_k4": 0.6,
            "target_fs_k7p25": 0.4,
        }

    def test_parent_summary_averages_children_before_error(self):
        rows = [
            self.realization_row(restart=1, realization=0, predicted_msd=1.0),
            self.realization_row(restart=2, realization=0, predicted_msd=3.0),
        ]
        restarts = closure.summarize_restarts(rows)
        parents = closure.summarize_parents(
            restarts,
            [
                {"temperature": 0.45, "replicate": 1, "parent_id": "parent-a"},
                {"temperature": 0.45, "replicate": 2, "parent_id": "parent-a"},
            ],
        )

        self.assertEqual(len(parents), 1)
        self.assertAlmostEqual(parents[0]["predicted_msd"], 2.0)
        self.assertAlmostEqual(parents[0]["msd_relative_error"], 0.0)
        self.assertEqual(parents[0]["child_restart_count"], 2)
        self.assertEqual(parents[0]["all_child_restart_curve_gate_pass"], 0.0)
        self.assertFalse(closure.curve_pass(parents))

    def test_restart_summary_computes_monte_carlo_error_before_parent_pooling(self):
        rows = [
            self.realization_row(restart=1, realization=0, predicted_msd=1.0),
            self.realization_row(restart=1, realization=1, predicted_msd=3.0),
        ]
        summary = closure.summarize_restarts(rows)[0]

        self.assertAlmostEqual(summary["predicted_msd"], 2.0)
        self.assertAlmostEqual(summary["mc_se_msd"], 1.0)
        self.assertEqual(summary["realization_count"], 2)

    def test_target_drift_within_restart_fails_closed(self):
        rows = [
            self.realization_row(restart=1, realization=0, predicted_msd=1.0),
            self.realization_row(restart=1, realization=1, predicted_msd=1.0),
        ]
        rows[1]["target_msd"] = 2.1
        with self.assertRaisesRegex(ValueError, "target"):
            closure.summarize_restarts(rows)


class MemoryClosureGateTests(unittest.TestCase):
    REQUIRED_ABLATIONS = (
        "mean_rate_null",
        "one_step_jump_law",
        "two_point_path_spectrum",
        "static_particle_environment",
        "finite_exchange_environment",
    )

    @staticmethod
    def blocker(*, missing=0, stationary=1.0):
        return {
            "temperature": 0.45,
            "evidence_role": "primary",
            "missing_parent_count": missing,
            "stationarity_pass": stationary,
        }

    @staticmethod
    def parent_row(*, parent, model, ngp_error=0.0, msd_error=0.0):
        return {
            "temperature": 0.45,
            "parent_id": parent,
            "model": model,
            "lag": 100.0,
            "target_msd": 1.0,
            "msd_relative_error": msd_error,
            "ngp_absolute_error": ngp_error,
            "absolute_error_fs_k2": 0.0,
            "absolute_error_fs_k4": 0.0,
            "absolute_error_fs_k7p25": 0.0,
            "mc_se_msd": 0.0,
            "mc_se_ngp": 0.0,
            "mc_se_fs_k2": 0.0,
            "mc_se_fs_k4": 0.0,
            "mc_se_fs_k7p25": 0.0,
            "support_pass": 1.0,
        }

    def parent_rows(self, *, full_fails=False, ablations_fail=True):
        rows = []
        for parent_index in range(3):
            parent = f"parent-{parent_index}"
            rows.append(
                self.parent_row(
                    parent=parent,
                    model="full_candidate",
                    msd_error=0.2 if full_fails and parent_index == 0 else 0.0,
                )
            )
            for model in self.REQUIRED_ABLATIONS:
                rows.append(
                    self.parent_row(
                        parent=parent,
                        model=model,
                        ngp_error=0.31
                        if ablations_fail and parent_index < 2
                        else 0.0,
                    )
                )
        return rows

    def upper_controls(self, *, pass_all=True):
        return [
            self.parent_row(
                parent=f"parent-{index}",
                model="contiguous_empirical_upper_control",
                msd_error=0.0 if pass_all else 0.2,
            )
            for index in range(3)
        ]

    def test_ensemble_average_cannot_rescue_failed_parent(self):
        gate = closure.classify_memory_closure_gate(
            parent_summaries=self.parent_rows(full_fails=True),
            blockers=[self.blocker()],
            upper_control_parents=self.upper_controls(),
        )

        self.assertEqual(gate["mechanism_state"], "candidate_rejected")
        self.assertEqual(gate["positive_memory_closure_claim_allowed"], 0.0)

    def test_gate_truth_table_is_fail_closed(self):
        cases = (
            (
                "blocked_independent_parent_validation",
                self.parent_rows(),
                [self.blocker(missing=1)],
                self.upper_controls(),
            ),
            (
                "blocked_stationarity_control",
                self.parent_rows(),
                [self.blocker(stationary=0.0)],
                self.upper_controls(),
            ),
            (
                "candidate_rejected",
                self.parent_rows(full_fails=True),
                [self.blocker()],
                self.upper_controls(),
            ),
            (
                "ablation_pattern_unresolved",
                self.parent_rows(ablations_fail=False),
                [self.blocker()],
                self.upper_controls(),
            ),
            (
                "positive_memory_closure_supported_within_tested_family",
                self.parent_rows(),
                [self.blocker()],
                self.upper_controls(),
            ),
        )
        for expected, parent_rows, blockers, controls in cases:
            with self.subTest(expected=expected):
                gate = closure.classify_memory_closure_gate(
                    parent_summaries=parent_rows,
                    blockers=blockers,
                    upper_control_parents=controls,
                )
                self.assertEqual(gate["mechanism_state"], expected)
                self.assertEqual(gate["microdynamic_closure_claim_allowed"], 0.0)
                self.assertEqual(gate["complete_microscopic_closure_claim_allowed"], 0.0)
                self.assertEqual(gate["spatial_facilitation_claim_allowed"], 0.0)
                self.assertEqual(gate["thermodynamic_claim_allowed"], 0.0)
                self.assertEqual(
                    gate["thermodynamic_glass_transition_claim_allowed"], 0.0
                )
                if expected.startswith("positive_memory"):
                    self.assertEqual(
                        gate["positive_memory_closure_claim_allowed"], 1.0
                    )

    def test_higher_order_score_and_precision_are_frozen(self):
        row = self.parent_row(parent="p", model="m", ngp_error=0.15)
        row["absolute_error_fs_k7p25"] = 0.024
        self.assertAlmostEqual(closure.higher_order_score(row), 0.8)
        self.assertTrue(closure.curve_pass([row]))
        row["mc_se_fs_k7p25"] = 0.0031
        self.assertFalse(closure.curve_pass([row]))

    def test_msd_monte_carlo_limit_is_relative_to_target(self):
        row = self.parent_row(parent="p", model="m")
        row["target_msd"] = 0.1
        row["mc_se_msd"] = 0.002
        self.assertFalse(closure.curve_pass([row]))


class MemoryClosureCliTests(unittest.TestCase):
    @staticmethod
    def load_cli():
        path = ROOT / "scripts" / "analyze_ka_prl_memory_closure.py"
        spec = importlib.util.spec_from_file_location("analyze_ka_prl_memory_closure", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def write_rows(path, rows):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def audit_inputs(self, directory):
        base = Path(directory)
        base.mkdir(parents=True, exist_ok=True)
        provenance = base / "provenance.csv"
        low_stationarity = base / "low_stationarity.csv"
        high_stationarity = base / "high_stationarity.csv"
        self.write_rows(provenance, ParentProvenanceTests().provenance_rows())
        self.write_rows(low_stationarity, ParentProvenanceTests.stationarity(True))
        self.write_rows(high_stationarity, ParentProvenanceTests.stationarity(False))
        return provenance, low_stationarity, high_stationarity

    @staticmethod
    def output_arguments(output):
        return [
            "--output-parent-ledger",
            str(output / "parents.csv"),
            "--output-blockers",
            str(output / "blockers.csv"),
            "--output-gate",
            str(output / "gate.csv"),
            "--output-claim-ledger",
            str(output / "claims.csv"),
        ]

    def run_audit(self, source, output):
        provenance, low_stationarity, high_stationarity = self.audit_inputs(source)
        arguments = [
            "--audit-only",
            "--provenance",
            str(provenance),
            "--low-stationarity",
            str(low_stationarity),
            "--high-stationarity",
            str(high_stationarity),
            *self.output_arguments(output),
        ]
        self.load_cli().main(arguments)

    def test_cli_audit_only_writes_exact_parent_blocker_and_closed_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            output = base / "out"
            output.mkdir()
            self.run_audit(base / "source", output)
            with (output / "blockers.csv").open() as handle:
                blockers = list(csv.DictReader(handle))
            low = next(row for row in blockers if float(row["temperature"]) == 0.45)
            warm = next(row for row in blockers if float(row["temperature"]) == 0.58)
            with (output / "gate.csv").open() as handle:
                gate = next(csv.DictReader(handle))

            self.assertEqual(low["blocker_state"], "missing_independent_parents")
            self.assertEqual(low["available_parent_count"], "1")
            self.assertEqual(low["missing_parent_count"], "2")
            self.assertEqual(low["shared_parent_resampling_can_satisfy"], "0")
            self.assertEqual(
                warm["blocker_state"], "stationarity_and_independent_parents"
            )
            self.assertEqual(
                gate["mechanism_state"], "blocked_independent_parent_validation"
            )
            self.assertEqual(gate["positive_memory_closure_claim_allowed"], "0")

    def test_cli_audit_outputs_are_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            first_source = base / "first_source"
            second_source = base / "second_source"
            first_source.mkdir()
            second_source.mkdir()
            first = base / "first"
            second = base / "second"
            first.mkdir()
            second.mkdir()
            self.run_audit(first_source, first)
            self.run_audit(second_source, second)

            self.assertEqual(
                {path.name: path.read_bytes() for path in first.iterdir()},
                {path.name: path.read_bytes() for path in second.iterdir()},
            )

    def test_correlated_parent_diagnostic_is_deterministic_and_never_reads_targets_as_inputs(self):
        cli = self.load_cli()
        generator = np.random.default_rng(91)
        blocks_by_restart = {
            restart: generator.normal(size=(4, 160, 3))
            for restart in (1, 2, 3)
        }
        targets = []
        spectral = []
        lags = (20, 100, 200, 500, 1000, 2000, 3000)
        for restart, blocks in blocks_by_restart.items():
            observed = cli.cumulative_observables_many_lags(
                blocks,
                block_counts=tuple(lag // 20 for lag in lags),
                wave_numbers=np.asarray((2.0, 4.0, 7.25)),
            )
            for lag in lags:
                row = observed[lag // 20]
                target = {
                    "replicate": restart,
                    "temperature": 0.45,
                    "lag": lag,
                    "observed_msd": row["msd"],
                    "observed_ngp": row["ngp"],
                    "observed_fs_k2": row["characteristic_k2"],
                    "observed_fs_k4": row["characteristic_k4"],
                    "observed_fs_k7p25": row["characteristic_k7p25"],
                }
                targets.append(target)
                for realization in range(8):
                    spectral.append(
                        {
                            "model": "radial_multivariate_surrogate",
                            "replicate": restart,
                            "realization": realization,
                            "temperature": 0.45,
                            "lag": lag,
                            "predicted_msd": row["msd"],
                            "predicted_ngp": row["ngp"],
                            "predicted_fs_k2": row["characteristic_k2"],
                            "predicted_fs_k4": row["characteristic_k4"],
                            "predicted_fs_k7p25": row["characteristic_k7p25"],
                            **{key: value for key, value in target.items() if key.startswith("observed_")},
                        }
                    )
        crossings = [
            {
                "temperature_group": "low",
                "replicate": restart,
                "block_size": 20,
                "efold_crossing_time": 200 + restart,
            }
            for restart in (1, 2, 3)
        ]
        arguments = {
            "blocks_by_restart": blocks_by_restart,
            "target_rows": targets,
            "crossing_rows": crossings,
            "spectral_source_rows": spectral,
            "temperature": 0.45,
            "realizations": 16,
        }

        first = cli.predict_correlated_parent_diagnostic(**arguments)
        second = cli.predict_correlated_parent_diagnostic(**arguments)

        self.assertEqual(first, second)
        self.assertEqual({row["model"] for row in first}, {
            "mean_rate_null",
            "one_step_jump_law",
            "two_point_path_spectrum",
            "static_particle_environment",
            "finite_exchange_environment",
            "full_candidate",
            "contiguous_empirical_upper_control",
        })
        self.assertTrue(all(row["heldout_path_used_in_prediction"] == 0.0 for row in first))
        self.assertTrue(
            all(row["heldout_observables_used_as_model_inputs"] == 0.0 for row in first)
        )


if __name__ == "__main__":
    unittest.main()
