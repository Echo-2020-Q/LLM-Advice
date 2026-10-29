"""动态信任的 CLI、配对对照和 spawn 输出测试；仅使用临时合成建议表。"""

from contextlib import redirect_stderr, redirect_stdout
import csv
from dataclasses import replace
import io
from itertools import product
import json
from unittest.mock import patch

from main.SPD import run_experiment as runner
from main.SPD.simulation import SimulationConfig, SpatialPrisonersDilemma
from main.SPD.test_llm_advisor import BatchFixtureTests


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class TrustIntegrationTests(BatchFixtureTests):
    def invoke(self, output, extra=()):
        common = [
            "--single", "--advice-mode", "adaptive", "--workers", "1", "--repeats", "1",
            "--size", "3", "--steps", "4", "--b", "1.5", "--k", "0.1", "--seed", "13",
            "--p", "0.5", "--initial-cooperation", "0.5", "--boundary", "periodic",
            "--update-mode", "synchronous", "--initial-trust", "0.6",
            "--trust-learning-rate", "0.2", "--aspiration-learning-rate", "0.3",
            "--paired-fermi-control", "--early-average-steps", "2", "--tail-average-steps", "2",
            "--report-every", "0", "--advisor-batch", str(self.batch), "--output-dir", str(output),
        ]
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return runner.main(common + list(extra))

    def run_single(self, name, extra=()):
        output = self.root / name
        self.assertEqual(self.invoke(output, extra), 0)
        run_dir = next(output.iterdir())
        return run_dir, read_json(run_dir / "config.json"), read_csv(run_dir / "history.csv")

    def assert_number(self, actual, expected):
        if expected is None:
            self.assertEqual(actual, "")
        else:
            self.assertAlmostEqual(float(actual), expected)

    def test_adaptive_history_summary_and_pure_fermi_pair_match_full_time_range(self):
        # lambda 的旧代码默认值为 0 时，动态信任仍应正常加载建议。
        with patch.object(runner, "LAMBDA_PROB", 0):
            run_dir, metadata, history = self.run_single("adaptive")
        config = SimulationConfig(**metadata["config"])
        self.assertEqual(config.advice_mode, "adaptive")
        self.assertEqual(config.initial_trust, 0.6)
        self.assertEqual(config.trust_learning_rate, 0.2)
        self.assertEqual(config.aspiration_learning_rate, 0.3)
        self.assertEqual(config.lambda_prob, 0)
        self.assertEqual(metadata["status"], "completed")
        control = SpatialPrisonersDilemma(replace(config, advice_mode="fixed", lambda_prob=0))
        self.assertEqual(config.llm_access_count, 4)
        self.assertEqual(len(history), 5)
        expected_control = [control.measure()] + [control.step() for _ in range(4)]
        for row, expected in zip(history, expected_control):
            self.assert_number(row["control_cooperation_rate"], expected.cooperation_rate)
            self.assert_number(row["control_average_payoff"], expected.average_payoff)
            mean = float(row["eligible_trust_mean"])
            self.assertLessEqual(float(row["eligible_trust_min"]), mean)
            self.assertLessEqual(mean, float(row["eligible_trust_max"]))
            self.assertTrue(0 <= mean <= 1)
            self.assertAlmostEqual(sum(float(row[f"eligible_trust_bin_{index}"])
                                       for index in range(10)), 1)
            self.assertEqual(sum(int(row[field]) for field in (
                "trust_positive_feedback", "trust_negative_feedback", "trust_zero_feedback",
            )), int(row["advice_used"]))
            if int(row["round"]) == 0:
                self.assertEqual(row["eligible_advice_rate"], "")
            else:
                self.assert_number(row["eligible_advice_rate"], int(row["advice_used"]) / 4)
        summary = metadata["summary"]
        self.assertEqual(summary["initial_eligible_trust_mean"], 0.6)
        self.assert_number(history[-1]["eligible_trust_mean"], summary["final_eligible_trust_mean"])
        self.assertAlmostEqual(summary["time_mean_eligible_trust"],
                               sum(float(row["eligible_trust_mean"]) for row in history[1:]) / 4)
        self.assertAlmostEqual(summary["early_mean_eligible_trust"],
                               sum(float(row["eligible_trust_mean"]) for row in history[1:3]) / 2)
        self.assertAlmostEqual(summary["tail_mean_eligible_trust"],
                               sum(float(row["eligible_trust_mean"]) for row in history[-2:]) / 2)
        self.assertAlmostEqual(summary["time_mean_eligible_advice_rate"],
                               sum(int(row["advice_used"]) for row in history) / 16)
        self.assertGreater(summary["total_advice_used"], 0)
        for direction in ("positive", "negative", "zero"):
            field = f"trust_{direction}_feedback"
            self.assertEqual(summary["total_" + field], sum(int(row[field]) for row in history))

    def test_zero_trust_learning_reproduces_fixed_probability_decisions(self):
        for probability in (0, 0.6, 1):
            with self.subTest(initial_trust=probability):
                adaptive_dir, adaptive, adaptive_history = self.run_single(
                    f"adaptive_{probability}", ["--initial-trust", str(probability),
                                                "--trust-learning-rate", "0"],
                )
                fixed_dir, fixed, fixed_history = self.run_single(
                    f"fixed_{probability}", ["--advice-mode", "fixed", "--lambda", str(probability)],
                )
                self.assertEqual((adaptive_dir / "final_grid.csv").read_bytes(),
                                 (fixed_dir / "final_grid.csv").read_bytes())
                for actual, expected in zip(adaptive_history, fixed_history):
                    for field in ("round", "cooperation_rate", "average_payoff", "advice_used",
                                  "advice_requests", "fermi_updates", "strategy_changes",
                                  "population_cumulative_mean_payoff", "eligible_advice_rate"):
                        self.assertEqual(actual[field], expected[field], field)
                    self.assert_number(actual["eligible_trust_mean"], probability)
                    self.assertEqual(expected["eligible_trust_mean"], "")
                for name in ("final_cooperation_rate", "tail_mean_cooperation_rate",
                             "eligible_cumulative_mean_payoff", "delta_population_cumulative_mean_payoff"):
                    self.assertEqual(adaptive["summary"][name], fixed["summary"][name])

    def scan_arguments(self, output, workers=1):
        return [
            "--scan", "--advice-mode", "adaptive", "--workers", str(workers), "--repeats", "1",
            "--size", "3", "--steps", "3", "--b-values", "1.5", "--k-values", "0.1",
            "--p-values", "0.5", "--seed-values", "13", "--initial-trust-values", "0.2", "0.8",
            "--trust-learning-rate-values", "0", "0.2", "--aspiration-learning-rate-values", "0", "0.3",
            "--initial-cooperation", "0.5", "--boundary", "periodic", "--update-mode", "synchronous",
            "--paired-fermi-control", "--early-average-steps", "2", "--tail-average-steps", "2",
            "--report-every", "0", "--advisor-batch", str(self.batch), "--no-scan-save-details",
            "--output-dir", str(output),
        ]

    def run_scan(self, output, workers=1):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(runner.main(self.scan_arguments(output, workers)), 0)
        return next(output.glob("scan_*"))

    def test_trust_scan_axes_ignore_fixed_lambda_list_and_parallel_outputs_are_identical(self):
        with patch.object(runner, "SCAN_LAMBDA_VALUES", [0, 0.1, 0.5, 1]):
            serial = self.run_scan(self.root / "serial", workers=1)
            parallel = self.run_scan(self.root / "parallel", workers=2)
        self.assertEqual((serial / "summary.csv").read_bytes(), (parallel / "summary.csv").read_bytes())
        metadata = read_json(parallel / "scan_config.json")
        self.assertEqual(metadata["total_runs"], 8)
        self.assertEqual(metadata["completed_run_ids"], list(range(1, 9)))
        self.assertEqual(metadata["actual_workers"], 2)
        self.assertNotIn("lambda_prob", metadata["scan_values"])
        self.assertEqual(metadata["scan_values"]["initial_trust"], [0.2, 0.8])
        self.assertEqual(metadata["scan_values"]["trust_learning_rate"], [0, 0.2])
        self.assertEqual(metadata["scan_values"]["aspiration_learning_rate"], [0, 0.3])
        rows = read_csv(parallel / "summary.csv")
        self.assertEqual(len(rows), 8)
        for row in rows:
            self.assertEqual(row["advice_mode"], "adaptive")
            self.assertEqual({path.name for path in (parallel / row["output_dir"]).iterdir()},
                             {"history.csv", "final_grid.csv"})
            self.assertNotEqual(row["final_eligible_trust_mean"], "")
            for filename in ("history.csv", "final_grid.csv"):
                self.assertEqual((serial / row["output_dir"] / filename).read_bytes(),
                                 (parallel / row["output_dir"] / filename).read_bytes())

    def test_zero_trust_or_no_eligible_people_do_not_require_advisor_files(self):
        missing = self.root / "missing_batch"
        for probability, access in ((0, 1), (0.6, 0)):
            with self.subTest(initial_trust=probability, access=access):
                _, metadata, history = self.run_single(
                    f"no_advisor_{probability}_{access}", ["--advisor-batch", str(missing),
                                                          "--initial-trust", str(probability), "--p", str(access)],
                )
                self.assertEqual(metadata["summary"]["total_advice_used"], 0)
                self.assertEqual(metadata["summary"]["delta_population_cumulative_mean_payoff"], 0)
                for row in history:
                    self.assert_number(row["eligible_trust_mean"], probability if access else None)
                    self.assertEqual(int(row["trust_positive_feedback"]), 0)
                    self.assertEqual(int(row["trust_negative_feedback"]), 0)
                    self.assertEqual(int(row["trust_zero_feedback"]), 0)

    def test_zero_steps_keep_initial_trust_but_do_not_invent_adoption_observations(self):
        _, metadata, history = self.run_single("zero_steps", ["--steps", "0"])
        self.assertEqual(len(history), 1)
        summary = metadata["summary"]
        self.assertEqual(summary["initial_eligible_trust_mean"], 0.6)
        self.assertEqual(summary["final_eligible_trust_mean"], 0.6)
        for name in ("time_mean_eligible_trust", "early_mean_eligible_trust", "tail_mean_eligible_trust",
                     "time_mean_eligible_advice_rate", "early_mean_eligible_advice_rate",
                     "tail_mean_eligible_advice_rate"):
            self.assertIsNone(summary[name], name)
        self.assertEqual(history[0]["eligible_advice_rate"], "")
        self.assertEqual(summary["total_advice_used"], 0)

    def test_current_3x3_grid_table_supports_adaptive_adoption_and_usage_statistics(self):
        # 当前真实批次采用512种完整3×3局面；这里用每局面一条回答的合成版本。
        cases = []
        for symbols in product("CD", repeat=9):
            key = "".join(symbols)
            cooperators = sum(key[index] == "C" for index in (1, 3, 5, 7))
            cases.append({"grid_key": key, "grid_rows": [key[:3], key[3:6], key[6:]],
                          "current_action": key[4], "cooperating_neighbors": cooperators,
                          "defecting_neighbors": 4 - cooperators})
        self.config = {"batch_id": "adaptive_grid_synthetic", "observation_format": "grid_3x3",
                       "b": None, "n": 1, "planned_requests": 512, "cases": cases}
        self.rows = [{**case, "grid_rows": "/".join(case["grid_rows"]),
                      "observation_format": "grid_3x3",
                      "request_index": index, "status": "success",
                      "recommendation": "C", "output_text": "C"}
                     for index, case in enumerate(cases, start=1)]
        self.write_batch()
        run_dir, metadata, history = self.run_single("adaptive_grid", ["--initial-trust", "1"])
        self.assertEqual(len(read_csv(run_dir / "advisor_table.csv")), 512)
        self.assertEqual(int(history[1]["advice_used"]), 4)
        used = sum(int(row["draws"]) for row in read_csv(run_dir / "advisor_usage.csv"))
        self.assertEqual(used, metadata["summary"]["total_advice_used"])
        self.assertGreater(used, 0)

    def test_editable_source_parameters_enable_trust_scan_without_command_line_flags(self):
        output = self.root / "source_defaults"
        with patch.multiple(
            runner, ADVICE_MODE="adaptive", INITIAL_TRUST=0.6, TRUST_LEARNING_RATE=0.25,
            ASPIRATION_LEARNING_RATE=0.4, SCAN_ENABLED=True, SCAN_B_VALUES=[1.5],
            SCAN_K_VALUES=[0.1], SCAN_P_VALUES=[0.5], SCAN_SEEDS=None,
            SCAN_INITIAL_TRUST_VALUES=[0.4, 0.7], SCAN_TRUST_LEARNING_RATES=[0.25],
            SCAN_ASPIRATION_LEARNING_RATES=[0.4], SCAN_LAMBDA_VALUES=[0, 0.1, 0.5, 1],
            SIZE=3, STEPS=3, REPEATS=1, NUM_WORKERS=1, SEED=13, INITIAL_COOPERATION=0.5,
            BOUNDARY="periodic", UPDATE_MODE="synchronous", LLM_BATCH_DIR=self.batch,
            OUTPUT_DIR=output, REPORT_EVERY=0, SCAN_SAVE_DETAILS=False,
        ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(runner.main([]), 0)
        batch = next(output.glob("scan_*"))
        rows = read_csv(batch / "summary.csv")
        self.assertEqual(len(rows), 2)
        self.assertEqual({float(row["initial_trust"]) for row in rows}, {0.4, 0.7})
        self.assertEqual({float(row["trust_learning_rate"]) for row in rows}, {0.25})
        self.assertEqual({float(row["aspiration_learning_rate"]) for row in rows}, {0.4})
        self.assertEqual({row["advice_mode"] for row in rows}, {"adaptive"})

    def test_invalid_or_ambiguous_trust_settings_fail_before_output_creation(self):
        invalid = [
            ["--initial-trust", "1.1"], ["--initial-trust", "nan"],
            ["--trust-learning-rate", "-0.1"], ["--trust-learning-rate", "inf"],
            ["--aspiration-learning-rate", "1.1"], ["--aspiration-learning-rate", "nan"],
            ["--advice-mode", "unknown"], ["--update-mode", "asynchronous"],
            ["--lambda", "0"], ["--lambda-values", "0", "1"],
            ["--initial-trust-values", "0.2", "0.8"],
            ["--trust-learning-rate-values", "0.1"],
            ["--aspiration-learning-rate-values", "0.1"],
        ]
        for index, arguments in enumerate(invalid):
            with self.subTest(arguments=arguments):
                output = self.root / f"invalid_{index}"
                with self.assertRaises(SystemExit) as raised:
                    self.invoke(output, arguments)
                self.assertEqual(raised.exception.code, 2)
                self.assertFalse(output.exists())
        for flag, values in (
            ("--initial-trust-values", ["0.2", "0.2"]),
            ("--trust-learning-rate-values", ["-0.1"]),
            ("--aspiration-learning-rate-values", ["inf"]),
            ("--lambda-values", ["0", "1"]),
        ):
            output = self.root / ("invalid_scan_" + flag[2:])
            with self.subTest(scan_flag=flag), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    runner.main(self.scan_arguments(output) + [flag] + values)
                self.assertEqual(raised.exception.code, 2)
                self.assertFalse(output.exists())
