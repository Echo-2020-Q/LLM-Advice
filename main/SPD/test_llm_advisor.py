"""离线经验建议测试：所有批次和仿真输出均写入临时目录，不调用 API。"""

import contextlib
import csv
from dataclasses import asdict
import io
import json
from pathlib import Path
import random
import socket
import tempfile
import unittest
from unittest import mock

from main.SPD.llm_advisor import AdviceContext, EmpiricalAdvisor, NeighborObservation
from main.SPD.simulation import SimulationConfig, SpatialPrisonersDilemma


def observation(strategy="C", cooperators=2, b=1.5, neighbors=4):
    return AdviceContext(
        round=1, position=(1, 1), strategy=strategy, payoff=0, b=b, k=0.1,
        neighbors=tuple(
            NeighborObservation((0, index), "C" if index < cooperators else "D", 0)
            for index in range(neighbors)
        ),
    )


class BatchFixtureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="spd_advisor_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.batch = self.root / "batch"
        self.batch.mkdir()
        # 任何意外网络访问立即失败，测试只允许本地合成数据。
        guard = mock.patch.object(socket, "socket", side_effect=AssertionError("禁止联网"))
        guard.start()
        self.addCleanup(guard.stop)
        self.cases = [
            {"b": 1.5, "current_action": action,
             "cooperating_neighbors": number, "defecting_neighbors": 4 - number}
            for action in ("C", "D") for number in range(5)
        ]
        self.config = {
            "batch_id": "synthetic_batch", "model": "gpt-5.5", "b": 1.5,
            "n": 10, "planned_requests": 100, "cases": self.cases,
        }
        self.rows = []
        for repetition in range(10):
            for case_index, case in enumerate(self.cases):
                # 相同邻居状态下，当前 C 有 20% 建议 C，当前 D 有 70% 建议 C。
                threshold = 2 if case["current_action"] == "C" else 7
                action = "C" if repetition < threshold else "D"
                self.rows.append({
                    "request_index": len(self.rows) + 1, "repetition": repetition + 1,
                    "combination_index": case_index + 1, **case,
                    "status": "success", "recommendation": action, "output_text": action,
                })
        self.write_batch()

    def write_batch(self):
        (self.batch / "config.json").write_text(
            json.dumps(self.config), encoding="utf-8"
        )
        with (self.batch / "results.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)


class EmpiricalAdvisorTests(BatchFixtureTests):
    def test_table_preserves_observed_action_frequencies_for_all_ten_cases(self):
        advisor = EmpiricalAdvisor(self.batch, seed=42)
        table = advisor.table_rows()
        self.assertEqual(len(table), 10)
        self.assertEqual(
            [(row["current_action"], row["cooperating_neighbors"]) for row in table],
            [(action, number) for action in ("C", "D") for number in range(5)],
        )
        for row in table:
            expected = 2 if row["current_action"] == "C" else 7
            self.assertEqual(row["samples"], 10)
            self.assertEqual(row["C"], expected)
            self.assertEqual(row["D"], 10 - expected)
            self.assertEqual(row["p_C"], expected / 10)
            self.assertEqual(row["cooperating_neighbors"] + row["defecting_neighbors"], 4)
        json.dumps(advisor.metadata())

    def test_sampling_with_replacement_matches_both_empirical_distributions(self):
        advisor = EmpiricalAdvisor(self.batch, seed=95)
        for strategy, probability in (("C", 0.2), ("D", 0.7)):
            with self.subTest(strategy=strategy):
                samples = [advisor(observation(strategy)) for _ in range(4000)]
                self.assertEqual(set(samples), {"C", "D"})
                self.assertAlmostEqual(samples.count("C") / len(samples), probability, delta=0.04)
        self.assertEqual(sum(row["draws"] for row in advisor.usage_rows()), 8000)
        self.assertEqual(sum(row["samples"] for row in advisor.table_rows()), 100)
        for row in advisor.usage_rows():
            self.assertEqual(row["draws"], row["selected_C"] + row["selected_D"])

    def test_same_seed_reproduces_sequence_and_other_seed_changes_it(self):
        first = EmpiricalAdvisor(self.batch, seed=11)
        second = EmpiricalAdvisor(self.batch, seed=11)
        third = EmpiricalAdvisor(self.batch, seed=12)
        context = observation("D", 3)
        sequence = [first(context) for _ in range(100)]
        self.assertEqual(sequence, [second(context) for _ in range(100)])
        self.assertNotEqual(sequence, [third(context) for _ in range(100)])

    def test_advisor_does_not_consume_global_or_fermi_random_stream(self):
        state = random.getstate()
        advisor = EmpiricalAdvisor(self.batch, seed=42)
        model = SpatialPrisonersDilemma(SimulationConfig(size=3, seed=42), advisor=advisor)
        fermi_state = model._rng.getstate()
        trigger_state = model._advice_rng.getstate()
        for _ in range(100):
            advisor(observation())
        self.assertEqual(state, random.getstate())
        self.assertEqual(fermi_state, model._rng.getstate())
        self.assertEqual(trigger_state, model._advice_rng.getstate())

    def test_failed_or_invalid_samples_are_rejected_instead_of_becoming_advice(self):
        invalid_fields = (
            ("status", "error"), ("status", "running"),
            ("recommendation", ""), ("recommendation", "cooperate"),
            ("output_text", "D"),  # 原始文本必须与已解析的 C 一致。
            ("b", 1.6), ("cooperating_neighbors", 5),
            ("defecting_neighbors", 3), ("current_action", "X"),
        )
        for field, value in invalid_fields:
            with self.subTest(field=field, value=value):
                old = self.rows[0][field]
                self.rows[0][field] = value
                self.write_batch()
                with self.assertRaises(ValueError):
                    EmpiricalAdvisor(self.batch)
                self.rows[0][field] = old

    def test_duplicate_request_index_is_rejected(self):
        self.rows[1]["request_index"] = self.rows[0]["request_index"]
        self.write_batch()
        with self.assertRaises(ValueError):
            EmpiricalAdvisor(self.batch)

    def test_missing_sample_is_rejected(self):
        self.rows.pop()
        self.write_batch()
        with self.assertRaises(ValueError):
            EmpiricalAdvisor(self.batch)

    def test_missing_combination_is_rejected(self):
        self.rows = [row for row in self.rows if row["combination_index"] != 10]
        self.write_batch()
        with self.assertRaises(ValueError):
            EmpiricalAdvisor(self.batch)

    def test_inconsistent_optional_source_b_or_sample_count_is_rejected(self):
        original = dict(self.config)
        for replacement in ({**original, "b": 1.6}, {**original, "n": 9}):
            with self.subTest(config=replacement):
                self.config = replacement
                self.write_batch()
                with self.assertRaises(ValueError):
                    EmpiricalAdvisor(self.batch)

    def test_advice_accepts_other_game_b_but_still_requires_four_neighbor_boundary(self):
        advisor = EmpiricalAdvisor(self.batch)
        advisor.validate_config(1.5, "periodic")
        advisor.validate_config(1.6, "periodic")
        self.assertIn(advisor(observation(b=1.6)), ("C", "D"))
        with self.assertRaises(ValueError):
            advisor.validate_config(1.5, "open")
        with self.assertRaises(ValueError):
            advisor(observation(neighbors=3))
        with self.assertRaises(ValueError):
            advisor(observation(strategy="X"))

    def test_same_local_states_and_seed_give_same_advice_across_different_game_b(self):
        first = EmpiricalAdvisor(self.batch, seed=29)
        second = EmpiricalAdvisor(self.batch, seed=29)
        contexts = [(strategy, cooperators) for _ in range(20)
                    for strategy in ("C", "D") for cooperators in range(5)]
        first_sequence = [first(observation(strategy, cooperators, b=1.01))
                          for strategy, cooperators in contexts]
        second_sequence = [second(observation(strategy, cooperators, b=1.8))
                           for strategy, cooperators in contexts]
        self.assertEqual(first_sequence, second_sequence)
        self.assertEqual(first.usage_rows(), second.usage_rows())
        self.assertEqual(first.metadata()["b"], 1.5)
        self.assertEqual(first.metadata()["matching_fields"],
                         ["current_action", "cooperating_neighbors", "defecting_neighbors"])

    def test_source_b_is_optional_in_config_cases_and_csv(self):
        original_config = json.loads(json.dumps(self.config))
        original_rows = [dict(row) for row in self.rows]
        for removed in (("config",), ("cases",), ("csv",), ("config", "cases", "csv")):
            with self.subTest(removed=removed):
                self.config = json.loads(json.dumps(original_config))
                self.rows = [dict(row) for row in original_rows]
                if "config" in removed:
                    self.config.pop("b")
                if "cases" in removed:
                    for case in self.config["cases"]:
                        case.pop("b")
                if "csv" in removed:
                    for row in self.rows:
                        row.pop("b")
                self.write_batch()
                advisor = EmpiricalAdvisor(self.batch, seed=42)
                advisor.validate_config(1.8, "periodic")
                self.assertEqual(advisor.metadata()["b"], None if "config" in removed else 1.5)
                self.assertNotIn("b", advisor.metadata()["matching_fields"])
                table = advisor.table_rows()
                self.assertEqual(len(table), 10)
                self.assertEqual(sum(row["samples"] for row in table), 100)
                self.assertIn(advisor(observation(b=1.8)), ("C", "D"))

    def test_lambda_zero_never_draws_and_preserves_fermi_trajectory(self):
        advisor = EmpiricalAdvisor(self.batch)
        config = SimulationConfig(size=4, lambda_prob=0, seed=29)
        baseline = SpatialPrisonersDilemma(config)
        empirical = SpatialPrisonersDilemma(config, advisor=advisor)
        self.assertEqual(baseline.run(10), empirical.run(10))
        self.assertEqual(baseline.grid, empirical.grid)
        self.assertEqual(sum(row["draws"] for row in advisor.usage_rows()), 0)

    def test_empirical_advice_receives_correct_synchronous_and_asynchronous_context(self):
        # 构造必定翻转自身策略的经验表，显式检查整个更新期间的局部观测。
        for row in self.rows:
            row["recommendation"] = row["output_text"] = (
                "D" if row["current_action"] == "C" else "C"
            )
        self.write_batch()
        for mode in ("synchronous", "asynchronous"):
            with self.subTest(mode=mode):
                advisor = EmpiricalAdvisor(self.batch)
                initial = [list("CDC"), list("DCD"), list("CCD")]
                expected = [row.copy() for row in initial]
                visited = []

                def observing_advisor(context):
                    row, column = context.position
                    old = initial if mode == "synchronous" else expected
                    self.assertEqual(context.strategy, old[row][column])
                    self.assertEqual(context.cooperating_neighbors, sum(
                        old[neighbor.position[0]][neighbor.position[1]] == "C"
                        for neighbor in context.neighbors
                    ))
                    selected = advisor(context)
                    expected[row][column] = selected
                    visited.append(context.position)
                    return selected

                model = SpatialPrisonersDilemma(
                    SimulationConfig(size=3, lambda_prob=1, update_mode=mode, seed=42),
                    advisor=observing_advisor, initial_grid=initial,
                )
                result = model.step()
                self.assertEqual([list(row) for row in model.grid], expected)
                self.assertEqual(len(visited), 9)
                self.assertEqual(result.advice_requests, 9)
                self.assertEqual(result.advice_used, 9)
                self.assertEqual(result.fermi_updates, 0)
                self.assertEqual(sum(row["draws"] for row in advisor.usage_rows()), 9)

    def test_simulation_with_empirical_advice_is_reproducible(self):
        for mode in ("synchronous", "asynchronous"):
            with self.subTest(mode=mode):
                config = SimulationConfig(size=4, lambda_prob=0.5, seed=7, update_mode=mode)
                first = SpatialPrisonersDilemma(config, advisor=EmpiricalAdvisor(self.batch, seed=7))
                second = SpatialPrisonersDilemma(config, advisor=EmpiricalAdvisor(self.batch, seed=7))
                self.assertEqual(first.run(5), second.run(5))
                self.assertEqual(first.grid, second.grid)


class ExperimentIntegrationTests(BatchFixtureTests):
    def run_cli(self, *arguments):
        from main.SPD import run_experiment

        argv = ["--size", "3", "--steps", "2", "--report-every", "0", "--repeats", "1",
                "--advisor-batch", str(self.batch), "--output-dir", str(self.root / "outputs")]
        with mock.patch.multiple(run_experiment, P=1.0, BOUNDARY="periodic", UPDATE_MODE="synchronous",
                                 INITIAL_COOPERATION=0.5, SEED=42, SCAN_SEEDS=None, SCAN_SAVE_DETAILS=True,
                                 PAIRED_FERMI_CONTROL=False, NUM_WORKERS=1, ADVICE_MODE="fixed"), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return run_experiment.main(argv + list(arguments))

    def test_single_run_uses_empirical_advice_and_saves_source_statistics(self):
        self.assertEqual(self.run_cli("--single", "--b", "1.5", "--lambda", "1", "--p", "1"), 0)
        histories = list((self.root / "outputs").rglob("history.csv"))
        self.assertEqual(len(histories), 1)
        with histories[0].open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 3)
        for row in rows[1:]:
            self.assertEqual(int(row["advice_used"]), 9)
            self.assertEqual(int(row["fermi_updates"]), 0)
        output = histories[0].parent
        with (output / "advisor_table.csv").open(encoding="utf-8-sig", newline="") as file:
            table = list(csv.DictReader(file))
        self.assertEqual(len(table), 10)
        self.assertEqual(sum(int(row["samples"]) for row in table), 100)
        with (output / "advisor_usage.csv").open(encoding="utf-8-sig", newline="") as file:
            usage = list(csv.DictReader(file))
        self.assertEqual(sum(int(row["draws"]) for row in usage), 18)
        metadata = json.loads((output / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["advisor_source"]["batch_id"], "synthetic_batch")
        self.assertEqual(metadata["advisor_source"]["sampling"], "uniform_with_replacement")
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["llm_access"]["eligible_count"], 9)
        self.assertEqual(metadata["llm_access"]["actual_fraction"], 1.0)

    def test_fixed_subset_uses_advice_only_for_eligible_agents_and_exports_mask(self):
        self.assertEqual(self.run_cli(
            "--single", "--b", "1.5", "--lambda", "1", "--p", "0.5",
            "--seed", "42", "--update-mode", "synchronous",
        ), 0)
        output = next((self.root / "outputs").iterdir())
        with (output / "history.csv").open(encoding="utf-8", newline="") as file:
            history = list(csv.DictReader(file))
        for row in history[1:]:
            self.assertEqual(int(row["advice_requests"]), 4)
            self.assertEqual(int(row["advice_used"]), 4)
            self.assertEqual(int(row["fermi_updates"]), 5)
        with (output / "llm_access_grid.csv").open(encoding="utf-8", newline="") as file:
            mask = list(csv.reader(file))
        self.assertEqual([len(row) for row in mask], [3, 3, 3])
        self.assertTrue(all(value in ("0", "1") for row in mask for value in row))
        self.assertEqual(sum(int(value) for row in mask for value in row), 4)
        metadata = json.loads((output / "config.json").read_text(encoding="utf-8"))
        expected_access = {
            "rule": "fixed_random_subset", "requested_fraction": 0.5,
            "eligible_count": 4, "actual_fraction": 4 / 9, "grid_file": "llm_access_grid.csv",
        }
        for name, value in expected_access.items():
            self.assertEqual(metadata["llm_access"][name], value)
        self.assertEqual(metadata["config"]["llm_access_fraction"], 0.5)
        self.assertEqual(metadata["summary"]["llm_access_fraction"], 0.5)
        self.assertEqual(metadata["summary"]["llm_access_count"], 4)
        self.assertEqual(metadata["summary"]["actual_llm_access_fraction"], 4 / 9)
        self.assertEqual(metadata["summary"]["total_advice_used"], 8)
        self.assertEqual(metadata["advisor_source"]["batch_id"], "synthetic_batch")
        independent = SpatialPrisonersDilemma(SimulationConfig(**metadata["config"]))
        self.assertEqual(mask, [[str(int(value)) for value in row]
                                for row in independent.llm_access_grid])
        with (output / "advisor_usage.csv").open(encoding="utf-8-sig", newline="") as file:
            self.assertEqual(sum(int(row["draws"]) for row in csv.DictReader(file)), 8)

    def test_invalid_game_b_or_unsupported_boundary_fails_before_creating_results(self):
        for parameters in (("--b", "1.0"), ("--b", "1.5", "--boundary", "open")):
            with self.subTest(parameters=parameters), self.assertRaises(SystemExit):
                self.run_cli("--single", "--lambda", "1", *parameters)
            self.assertFalse((self.root / "outputs").exists())

    def test_lambda_zero_does_not_require_an_advisor_batch(self):
        self.assertEqual(self.run_cli("--single", "--b", "1.6", "--lambda", "0",
                                      "--boundary", "open",
                                      "--advisor-batch", str(self.root / "missing")), 0)

    def test_zero_eligible_count_skips_advisor_loading_for_single_and_scan(self):
        from main.SPD import run_experiment

        # 正 P 也可能因人数向下取整而没有任何可听取建议的个体。
        with mock.patch.object(run_experiment, "EmpiricalAdvisor",
                               side_effect=AssertionError("不应加载本地批次")):
            for fraction in ("0", "0.05"):
                with self.subTest(fraction=fraction):
                    self.assertEqual(self.run_cli(
                        "--single", "--b", "1.7", "--lambda", "1", "--p", fraction,
                        "--boundary", "open", "--advisor-batch", str(self.root / "missing"),
                    ), 0)
            self.assertEqual(self.run_cli(
                "--scan", "--b-values", "1.1", "1.7", "--k-values", "0.1",
                "--lambda-values", "1", "--p-values", "0", "0.05", "--seed-values", "42",
                "--boundary", "open", "--advisor-batch", str(self.root / "missing"),
            ), 0)
        configs = list((self.root / "outputs").rglob("config.json"))
        self.assertEqual(len(configs), 6)
        for path in configs:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["llm_access"]["eligible_count"], 0)
            self.assertEqual(metadata["summary"]["total_advice_requests"], 0)
            self.assertEqual(metadata["summary"]["total_advice_used"], 0)
            self.assertEqual(metadata["summary"]["total_fermi_updates"], 18)
            self.assertEqual(metadata["advisor_source"]["source"], "disabled_or_placeholder")
            self.assertFalse((path.parent / "advisor_usage.csv").exists())

    def test_scan_uses_one_independent_advisor_per_run(self):
        self.assertEqual(self.run_cli(
            "--scan", "--b-values", "1.5", "--k-values", "0.1",
            "--lambda-values", "0", "1", "--p-values", "0", "0.5", "1",
            "--seed-values", "42", "43", "--repeats", "2", "--update-mode", "synchronous",
        ), 0)
        histories = list((self.root / "outputs").rglob("history.csv"))
        self.assertEqual(len(histories), 12)
        used = []
        for path in histories:
            with path.open(encoding="utf-8-sig", newline="") as file:
                used.append(sum(int(row["advice_used"]) for row in csv.DictReader(file)))
            metadata = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
            config = SimulationConfig(**metadata["config"])
            expected = 2 * config.llm_access_count if config.lambda_prob == 1 else 0
            self.assertEqual(used[-1], expected)
        self.assertEqual(sorted(used), [0] * 8 + [8, 8, 18, 18])
        usage_files = list((self.root / "outputs").rglob("advisor_usage.csv"))
        self.assertEqual(len(usage_files), 4)
        for path in usage_files:
            metadata = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
            expected = 2 * metadata["llm_access"]["eligible_count"]
            with path.open(encoding="utf-8-sig", newline="") as file:
                self.assertEqual(sum(int(row["draws"]) for row in csv.DictReader(file)), expected)

    def test_scan_preflights_all_combinations_before_running_any(self):
        with self.assertRaises(SystemExit):
            self.run_cli("--scan", "--b-values", "1.5", "1.6", "--k-values", "0.1",
                         "--lambda-values", "0", "1", "--p-values", "0", "0.5",
                         "--seed-values", "42", "--boundary", "open")
        self.assertFalse((self.root / "outputs").exists())

    def test_scan_reuses_advice_for_other_b_and_for_sources_without_b(self):
        for with_source_b in (True, False):
            with self.subTest(with_source_b=with_source_b):
                if not with_source_b:
                    self.config.pop("b")
                    for case in self.config["cases"]:
                        case.pop("b")
                    for row in self.rows:
                        row.pop("b")
                    self.write_batch()
                output_root = self.root / ("scan_with_b" if with_source_b else "scan_without_b")
                self.assertEqual(self.run_cli(
                    "--scan", "--b-values", "1.01", "1.8", "--k-values", "0.1",
                    "--lambda-values", "1", "--p-values", "1", "--seed-values", "42",
                    "--output-dir", str(output_root),
                ), 0)
                batch = next(output_root.glob("scan_*"))
                with (batch / "summary.csv").open(encoding="utf-8", newline="") as file:
                    summaries = list(csv.DictReader(file))
                self.assertEqual([float(row["b"]) for row in summaries], [1.01, 1.8])
                for summary in summaries:
                    output = batch / summary["output_dir"]
                    b = float(summary["b"])
                    metadata = json.loads((output / "config.json").read_text(encoding="utf-8"))
                    self.assertEqual(metadata["config"]["b"], b)
                    self.assertEqual(metadata["payoff_matrix"]["T"], b)
                    self.assertEqual(metadata["advisor_source"]["b"], 1.5 if with_source_b else None)
                    self.assertNotIn("b", metadata["advisor_source"]["matching_fields"])
                    self.assertEqual(int(summary["total_advice_used"]), 18)
                    with (output / "history.csv").open(encoding="utf-8", newline="") as file:
                        history = list(csv.DictReader(file))
                    self.assertEqual([int(row["advice_used"]) for row in history], [0, 9, 9])
                    with (output / "initial_grid.csv").open(encoding="utf-8", newline="") as file:
                        initial = list(csv.reader(file))
                    expected_payoff = 0
                    for r in range(3):
                        for c in range(3):
                            cooperating = sum(initial[nr % 3][nc % 3] == "C" for nr, nc in
                                              ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))
                            expected_payoff += cooperating * (1 if initial[r][c] == "C" else b)
                    self.assertAlmostEqual(float(history[0]["average_payoff"]), expected_payoff / 9)

    def test_repeats_use_independent_empirical_samplers_and_reproduce_single_seed_trajectories(self):
        self.assertEqual(self.run_cli(
            "--single", "--repeats", "3", "--seed", "51", "--b", "1.5",
            "--lambda", "1", "--p", "0.5", "--update-mode", "synchronous",
        ), 0)
        batches = list((self.root / "outputs").glob("repeat_*"))
        self.assertEqual(len(batches), 1)
        batch = batches[0]
        with (batch / "summary.csv").open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual([int(row["seed"]) for row in rows], [51, 52, 53])
        for row in rows:
            output = batch / row["output_dir"]
            metadata = json.loads((output / "config.json").read_text(encoding="utf-8"))
            config = SimulationConfig(**metadata["config"])
            advisor = EmpiricalAdvisor(self.batch, seed=config.seed)
            independent = SpatialPrisonersDilemma(config, advisor=advisor)
            expected_history = independent.run(2)
            with (output / "history.csv").open(encoding="utf-8", newline="") as file:
                history = list(csv.DictReader(file))
            legacy_fields = tuple(asdict(expected_history[0]))
            self.assertEqual([{key: row[key] for key in legacy_fields} for row in history],
                             [{key: str(value) for key, value in asdict(item).items()}
                              for item in expected_history])
            with (output / "advisor_usage.csv").open(encoding="utf-8-sig", newline="") as file:
                usage = list(csv.DictReader(file))
            self.assertEqual(sum(int(item["draws"]) for item in usage), 8)
            self.assertEqual(usage, [{key: str(value) for key, value in item.items()}
                                    for item in advisor.usage_rows()])
            self.assertEqual(int(row["total_advice_used"]), 8)

    def test_compact_scan_uses_empirical_advice_without_extra_run_files(self):
        self.assertEqual(self.run_cli(
            "--scan", "--no-scan-save-details", "--b-values", "1.8", "--k-values", "0.1",
            "--lambda-values", "1", "--p-values", "0.5", "1", "--seed-values", "42",
        ), 0)
        batch = next((self.root / "outputs").glob("scan_*"))
        metadata = json.loads((batch / "scan_config.json").read_text(encoding="utf-8"))
        self.assertFalse(metadata["save_details"])
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["completed_runs"], 2)
        self.assertEqual([source["batch_id"] for source in metadata["advisor_sources"]],
                         ["synthetic_batch", "synthetic_batch"])
        with (batch / "summary.csv").open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 2)
        for row in rows:
            count = 4 if float(row["llm_access_fraction"]) == 0.5 else 9
            self.assertEqual(int(row["total_advice_used"]), 2 * count)
            output = batch / row["output_dir"]
            self.assertEqual({path.name for path in output.iterdir()}, {"history.csv", "final_grid.csv"})
            with (output / "history.csv").open(encoding="utf-8", newline="") as file:
                history = list(csv.DictReader(file))
            self.assertEqual([int(item["advice_used"]) for item in history], [0, count, count])


if __name__ == "__main__":
    unittest.main()
