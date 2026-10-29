"""参数扫描的文件输出、可复现性和失败恢复测试（Python >= 3.9）。"""

import csv
import io
import itertools
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from main.SPD import run_experiment as runner


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_grid(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class ParameterScanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="spd_scan_test_")
        self.addCleanup(temporary.cleanup)
        self.output_root = Path(temporary.name)
        # 可编辑实验默认值属于用户配置；测试固定默认模式，再由各用例显式覆盖。
        defaults = patch.multiple(runner, REPEATS=1, SCAN_ENABLED=False, SCAN_SEEDS=None,
                                  SCAN_SAVE_DETAILS=True, PAIRED_FERMI_CONTROL=False, NUM_WORKERS=1,
                                  ADVICE_MODE="fixed")
        defaults.start()
        self.addCleanup(defaults.stop)
        # 此组测试隔离扫描与汇总逻辑，显式保留空建议；真实本地抽样由 test_llm_advisor 覆盖。
        placeholder = patch.object(runner, "create_advisor", return_value=None)
        placeholder.start()
        self.addCleanup(placeholder.stop)

    def invoke(self, argv):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return runner.main(argv)

    def scan_arguments(self, output=None):
        return [
            "--scan", "--b-values", "1.01", "1.03",
            "--k-values", "0.1", "0.5", "--lambda-values", "0", "0.4",
            "--p-values", "1",
            "--seed-values", "1", "2", "--repeats", "2", "--size", "3", "--steps", "3",
            "--initial-cooperation", "0.5", "--boundary", "open",
            "--update-mode", "asynchronous", "--tail-average-steps", "2",
            "--report-every", "0", "--output-dir", str(output or self.output_root),
        ]

    def only_batch(self, root=None):
        batches = list((root or self.output_root).glob("scan_*"))
        self.assertEqual(len(batches), 1)
        return batches[0]

    def run_scan(self, extra=None, output=None):
        self.assertEqual(self.invoke(self.scan_arguments(output) + (extra or [])), 0)
        return self.only_batch(output)

    def assert_summary_matches_history(self, row, history, tail_count):
        samples = history[1:] if len(history) > 1 else history
        samples = samples[-tail_count:]
        self.assertEqual(int(row["tail_average_steps"]), len(samples))
        self.assertAlmostEqual(
            float(row["tail_mean_cooperation_rate"]),
            sum(float(item["cooperation_rate"]) for item in samples) / len(samples),
        )
        self.assertAlmostEqual(
            float(row["tail_mean_payoff"]),
            sum(float(item["average_payoff"]) for item in samples) / len(samples),
        )
        self.assertEqual(int(row["tail_start_round"]), int(samples[0]["round"]))
        self.assertEqual(int(row["tail_end_round"]), int(samples[-1]["round"]))
        self.assertEqual(float(row["initial_cooperation_rate"]), float(history[0]["cooperation_rate"]))
        self.assertEqual(float(row["final_cooperation_rate"]), float(history[-1]["cooperation_rate"]))
        self.assertEqual(float(row["final_average_payoff"]), float(history[-1]["average_payoff"]))
        for counter in ("advice_requests", "advice_used", "fermi_updates", "strategy_changes"):
            self.assertEqual(int(row["total_" + counter]), sum(int(item[counter]) for item in history))

    def test_cartesian_product_outputs_and_summary_statistics(self):
        batch = self.run_scan()
        metadata = read_json(batch / "scan_config.json")
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["total_runs"], 16)
        self.assertEqual(metadata["completed_runs"], 16)
        self.assertEqual(metadata["tail_average_steps"], 2)
        self.assertEqual(metadata["scan_values"], {
            "b": [1.01, 1.03], "k": [0.1, 0.5], "lambda_prob": [0.0, 0.4],
            "llm_access_fraction": [1.0], "seed": [1, 2],
        })
        rows = read_csv(batch / "summary.csv")
        self.assertEqual(len(rows), 16)
        self.assertEqual([int(row["run_id"]) for row in rows], list(range(1, 17)))
        actual_combinations = {
            (float(row["b"]), float(row["k"]), float(row["lambda_prob"]), int(row["seed"]))
            for row in rows
        }
        self.assertEqual(actual_combinations, set(itertools.product(
            (1.01, 1.03), (0.1, 0.5), (0.0, 0.4), (1, 2),
        )))
        self.assertEqual(len({row["output_dir"] for row in rows}), 16)
        for row in rows:
            with self.subTest(run_id=row["run_id"]):
                relative = Path(row["output_dir"])
                self.assertFalse(relative.is_absolute())
                output = batch / relative
                output.resolve().relative_to(batch.resolve())
                self.assertTrue({"config.json", "history.csv", "initial_grid.csv", "final_grid.csv",
                                 "llm_access_grid.csv"}
                                <= {path.name for path in output.iterdir()})
                run_metadata = read_json(output / "config.json")
                self.assertEqual(run_metadata["status"], "completed")
                self.assertEqual(run_metadata["completed_steps"], 3)
                config = run_metadata["config"]
                for name in ("b", "k", "lambda_prob", "llm_access_fraction", "initial_cooperation"):
                    self.assertEqual(config[name], float(row[name]))
                for name in ("size", "seed"):
                    self.assertEqual(config[name], int(row[name]))
                self.assertEqual(config["size"], 3)
                self.assertEqual(config["llm_access_fraction"], 1.0)
                self.assertEqual(int(row["llm_access_count"]), 9)
                self.assertEqual(float(row["actual_llm_access_fraction"]), 1.0)
                self.assertEqual(read_grid(output / "llm_access_grid.csv"), [["1"] * 3 for _ in range(3)])
                self.assertEqual(config["initial_cooperation"], 0.5)
                self.assertEqual(config["boundary"], "open")
                self.assertEqual(config["update_mode"], "asynchronous")
                self.assertEqual(row["boundary"], config["boundary"])
                self.assertEqual(row["update_mode"], config["update_mode"])
                self.assertEqual(int(row["steps"]), 3)
                self.assertEqual(row["status"], "completed")
                history = read_csv(output / "history.csv")
                self.assertEqual([int(item["round"]) for item in history], [0, 1, 2, 3])
                self.assert_summary_matches_history(row, history, tail_count=2)

    def test_every_combination_reinitializes_and_none_advice_preserves_trajectory(self):
        batch = self.run_scan()
        rows = read_csv(batch / "summary.csv")
        initial_by_seed = {}
        trajectories = {}
        for row in rows:
            output = batch / row["output_dir"]
            config = runner.SimulationConfig(**read_json(output / "config.json")["config"])
            independently_initialized = runner.SpatialPrisonersDilemma(config)
            initial = read_grid(output / "initial_grid.csv")
            self.assertEqual(initial, [list(line) for line in independently_initialized.grid])
            seed = int(row["seed"])
            if seed in initial_by_seed:
                self.assertEqual(initial, initial_by_seed[seed])
            initial_by_seed[seed] = initial
            history = read_csv(output / "history.csv")
            state_fields = ("round", "cooperation_rate", "average_payoff", "fermi_updates", "strategy_changes")
            trajectory = [tuple(item[field] for field in state_fields) for item in history]
            key = (float(row["b"]), float(row["k"]), seed)
            if key in trajectories:
                self.assertEqual(trajectory, trajectories[key])
            trajectories[key] = trajectory
            self.assertTrue(all(int(item["advice_used"]) == 0 for item in history))
            if float(row["lambda_prob"]) == 0:
                self.assertTrue(all(int(item["advice_requests"]) == 0 for item in history))

    def test_single_mode_and_scan_have_identical_complete_trajectories(self):
        scan_root = self.output_root / "scan"
        batch = self.run_scan([
            "--b-values", "1.03", "--k-values", "0.5",
            "--lambda-values", "0.4", "--seed-values", "2", "--repeats", "1",
        ], output=scan_root)
        row = read_csv(batch / "summary.csv")[0]
        scan_output = batch / row["output_dir"]
        single_root = self.output_root / "single"
        self.assertEqual(self.invoke([
            "--single", "--b", "1.03", "--k", "0.5", "--lambda", "0.4", "--seed", "2",
            "--p", "1",
            "--size", "3", "--steps", "3", "--initial-cooperation", "0.5", "--boundary", "open",
            "--update-mode", "asynchronous", "--tail-average-steps", "2", "--report-every", "0",
            "--output-dir", str(single_root),
        ]), 0)
        outputs = list(single_root.iterdir())
        self.assertEqual(len(outputs), 1)
        single_output = outputs[0]
        for filename in ("history.csv", "initial_grid.csv", "final_grid.csv", "llm_access_grid.csv"):
            self.assertEqual((scan_output / filename).read_bytes(), (single_output / filename).read_bytes())
        self.assertEqual(read_json(scan_output / "config.json")["config"],
                         read_json(single_output / "config.json")["config"])

    def test_source_configuration_enables_scan_and_repeated_batches_are_unique(self):
        with patch.multiple(
            runner, SCAN_ENABLED=True, SCAN_B_VALUES=[1.02, 1.04], SCAN_K_VALUES=[0.2],
            SCAN_LAMBDA_VALUES=[0.0], SCAN_P_VALUES=[0.5], SCAN_SEEDS=[11], REPEATS=1,
            TAIL_AVERAGE_STEPS=2,
            SIZE=3, STEPS=2, REPORT_EVERY=0, INITIAL_COOPERATION=0.5,
            BOUNDARY="open", UPDATE_MODE="asynchronous", OUTPUT_DIR=self.output_root,
        ):
            self.assertEqual(self.invoke([]), 0)
            self.assertEqual(self.invoke([]), 0)
        batches = list(self.output_root.glob("scan_*"))
        self.assertEqual(len(batches), 2)
        for batch in batches:
            rows = read_csv(batch / "summary.csv")
            self.assertEqual(len(rows), 2)
            self.assertEqual({float(row["b"]) for row in rows}, {1.02, 1.04})
            self.assertEqual({float(row["k"]) for row in rows}, {0.2})
            self.assertEqual({float(row["llm_access_fraction"]) for row in rows}, {0.5})
            self.assertEqual({int(row["llm_access_count"]) for row in rows}, {4})
            self.assertEqual({int(row["seed"]) for row in rows}, {11})
            self.assertTrue(all(int(row["tail_average_steps"]) == 2 for row in rows))
            self.assertEqual(read_json(batch / "scan_config.json")["status"], "completed")

    def test_invalid_scan_is_rejected_before_creating_outputs(self):
        invalid_arguments = [
            ["--b-values"],
            ["--b-values", "1.01", "1.0"],
            ["--k-values", "0.1", "0"],
            ["--lambda-values", "0", "1.1"],
            ["--p-values"],
            ["--p-values", "0", "-0.1"],
            ["--p-values", "0", "1.1"],
            ["--p-values", "0", "nan"],
            ["--p-values", "0", "inf"],
            ["--b", "1.2"],
            ["--p", "0.5"],
        ]
        for index, extra in enumerate(invalid_arguments):
            with self.subTest(extra=extra):
                output = self.output_root / ("invalid_" + str(index))
                with self.assertRaises(SystemExit) as raised:
                    self.invoke(self.scan_arguments(output) + extra)
                self.assertEqual(raised.exception.code, 2)
                self.assertFalse(output.exists())
        output = self.output_root / "empty_source_axis"
        with patch.object(runner, "SCAN_B_VALUES", []):
            with self.assertRaises(SystemExit) as raised:
                self.invoke(["--scan", "--output-dir", str(output)])
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(output.exists())
        output = self.output_root / "single_with_scan_axis"
        with self.assertRaises(SystemExit) as raised:
            self.invoke(["--single", "--b-values", "1.01", "--output-dir", str(output)])
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(output.exists())

    def test_p_scan_reinitializes_every_grid_and_preserves_placeholder_trajectories(self):
        batch = self.run_scan([
            "--b-values", "1.03", "--k-values", "0.1", "--lambda-values", "1",
            "--p-values", "0", "0.5", "1", "--seed-values", "42", "43",
            "--update-mode", "synchronous",
        ])
        metadata = read_json(batch / "scan_config.json")
        self.assertEqual(metadata["total_runs"], 6)
        self.assertEqual(metadata["completed_runs"], 6)
        self.assertEqual(metadata["scan_values"]["llm_access_fraction"], [0.0, 0.5, 1.0])
        rows = read_csv(batch / "summary.csv")
        self.assertEqual(len(rows), 6)
        self.assertEqual({(float(row["llm_access_fraction"]), int(row["seed"])) for row in rows},
                         set(itertools.product((0.0, 0.5, 1.0), (42, 43))))
        state_by_seed = {}
        for row in rows:
            output = batch / row["output_dir"]
            self.assertIn("_p", output.name)
            config = runner.SimulationConfig(**read_json(output / "config.json")["config"])
            independent = runner.SpatialPrisonersDilemma(config)
            initial = read_grid(output / "initial_grid.csv")
            access = read_grid(output / "llm_access_grid.csv")
            self.assertEqual(initial, [list(line) for line in independent.grid])
            self.assertEqual(access, [[str(int(value)) for value in line]
                                      for line in independent.llm_access_grid])
            count = {0.0: 0, 0.5: 4, 1.0: 9}[config.llm_access_fraction]
            self.assertEqual(sum(int(value) for line in access for value in line), count)
            self.assertEqual(int(row["llm_access_count"]), count)
            self.assertEqual(float(row["actual_llm_access_fraction"]), count / 9)
            self.assertEqual(int(row["total_advice_requests"]), 3 * count)
            history = read_csv(output / "history.csv")
            fields = ("cooperation_rate", "average_payoff", "strategy_changes", "fermi_updates")
            state = (initial, [tuple(item[name] for name in fields) for item in history])
            if config.seed in state_by_seed:
                self.assertEqual(state, state_by_seed[config.seed])
            state_by_seed[config.seed] = state

    def test_source_p_controls_single_run_and_invalid_p_axes_leave_no_outputs(self):
        output = self.output_root / "source_p"
        with patch.multiple(runner, SCAN_ENABLED=False, SCAN_SAVE_DETAILS=False, P=0.5, SIZE=3, STEPS=0,
                            REPORT_EVERY=0, OUTPUT_DIR=output):
            self.assertEqual(self.invoke([]), 0)
        metadata = read_json(next(output.iterdir()) / "config.json")
        self.assertEqual(metadata["config"]["llm_access_fraction"], 0.5)
        self.assertEqual(metadata["llm_access"]["eligible_count"], 4)
        self.assertTrue({"initial_grid.csv", "llm_access_grid.csv", "config.json"}
                        <= {path.name for path in next(output.iterdir()).iterdir()})

        for index, extra in enumerate((
            ["--single", "--p-values", "0.5"],
            ["--single", "--p", "-0.1"],
            ["--single", "--p", "1.1"],
            ["--single", "--p", "nan"],
        )):
            with self.subTest(extra=extra):
                invalid_output = self.output_root / ("invalid_p_" + str(index))
                with self.assertRaises(SystemExit) as raised:
                    self.invoke(extra + ["--output-dir", str(invalid_output)])
                self.assertEqual(raised.exception.code, 2)
                self.assertFalse(invalid_output.exists())
        for index, values in enumerate(([], 0.5, [0.5, -0.1])):
            with self.subTest(values=values):
                invalid_output = self.output_root / ("invalid_p_axis_" + str(index))
                with patch.object(runner, "SCAN_P_VALUES", values):
                    with self.assertRaises(SystemExit) as raised:
                        self.invoke(["--scan", "--output-dir", str(invalid_output)])
                self.assertEqual(raised.exception.code, 2)
                self.assertFalse(invalid_output.exists())

    def test_failed_run_preserves_completed_rows_and_marks_partial_outputs(self):
        real_step = runner.SpatialPrisonersDilemma.step
        expected_error = RuntimeError("test failure in second parameter combination")

        def fail_in_second_run(model):
            if model.config.b == 1.03 and model.round == 1:
                raise expected_error
            return real_step(model)

        with patch.object(runner.SpatialPrisonersDilemma, "step", autospec=True,
                          side_effect=fail_in_second_run):
            with self.assertRaises(RuntimeError) as raised:
                self.invoke(self.scan_arguments() + [
                    "--k-values", "0.1", "--lambda-values", "0", "--seed-values", "1",
                    "--repeats", "1",
                ])
        self.assertIs(raised.exception, expected_error)
        batch = self.only_batch()
        metadata = read_json(batch / "scan_config.json")
        self.assertIn(metadata["status"], ("failed", "failed_or_interrupted"))
        self.assertEqual(metadata["total_runs"], 2)
        self.assertEqual(metadata["completed_runs"], 1)
        self.assertEqual(metadata["failed_run_id"], 2)
        rows = read_csv(batch / "summary.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "completed")
        self.assertEqual(float(rows[0]["b"]), 1.01)
        self.assertTrue((batch / rows[0]["output_dir"] / "final_grid.csv").exists())
        configs = list(batch.glob("*/config.json"))
        self.assertEqual(len(configs), 2)
        failed_configs = [path for path in configs if read_json(path)["status"] != "completed"]
        self.assertEqual(len(failed_configs), 1)
        failed = read_json(failed_configs[0])
        self.assertIn(failed["status"], ("failed", "failed_or_interrupted"))
        self.assertEqual(failed["completed_steps"], 1)
        self.assertEqual([int(item["round"]) for item in read_csv(failed_configs[0].parent / "history.csv")], [0, 1])

    def test_zero_steps_uses_initial_state_for_tail_statistics(self):
        batch = self.run_scan([
            "--b-values", "1.01", "--k-values", "0.1", "--lambda-values", "0",
            "--seed-values", "1", "--repeats", "1", "--steps", "0", "--tail-average-steps", "100",
        ])
        rows = read_csv(batch / "summary.csv")
        self.assertEqual(len(rows), 1)
        output = batch / rows[0]["output_dir"]
        history = read_csv(output / "history.csv")
        self.assertEqual(len(history), 1)
        self.assertEqual(int(history[0]["round"]), 0)
        self.assert_summary_matches_history(rows[0], history, tail_count=100)
        self.assertEqual(int(rows[0]["tail_average_steps"]), 1)
        self.assertEqual((output / "initial_grid.csv").read_bytes(), (output / "final_grid.csv").read_bytes())
        self.assertEqual(read_json(batch / "scan_config.json")["status"], "completed")

    def test_source_repeats_reproduce_each_single_seed_with_independent_initialization(self):
        batches = []
        for name in ("first", "second"):
            output = self.output_root / name
            with patch.multiple(
                runner, SCAN_ENABLED=False, SCAN_SAVE_DETAILS=False, REPEATS=3, SEED=31, SIZE=3, STEPS=3,
                B=1.03, K=0.1, LAMBDA_PROB=1.0, P=0.5, INITIAL_COOPERATION=0.5,
                UPDATE_MODE="synchronous", REPORT_EVERY=0, OUTPUT_DIR=output,
            ):
                self.assertEqual(self.invoke([]), 0)
            candidates = list(output.glob("repeat_*"))
            self.assertEqual(len(candidates), 1)
            batches.append(candidates[0])
        rows = read_csv(batches[0] / "summary.csv")
        metadata = read_json(batches[0] / "repeat_config.json")
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["repeats"], 3)
        self.assertEqual(metadata["parameter_sets"], 1)
        self.assertEqual(metadata["total_runs"], 3)
        self.assertEqual(metadata["completed_runs"], 3)
        self.assertEqual(metadata["seed_values"], [31, 32, 33])
        self.assertIn("seed_policy", metadata)
        self.assertEqual([int(row["seed"]) for row in rows], [31, 32, 33])
        self.assertEqual([int(row["repeat_index"]) for row in rows], [1, 2, 3])
        repeated_rows = read_csv(batches[1] / "summary.csv")
        filenames = ("history.csv", "initial_grid.csv", "final_grid.csv", "llm_access_grid.csv")
        for row, repeated in zip(rows, repeated_rows):
            self.assertEqual(int(row["parameter_set_id"]), 1)
            self.assertEqual(int(row["repeats"]), 3)
            run_output = batches[0] / row["output_dir"]
            run_metadata = read_json(run_output / "config.json")
            for field in ("parameter_set_id", "repeat_index", "repeats"):
                self.assertEqual(run_metadata[field], int(row[field]))
                self.assertEqual(run_metadata["summary"][field], int(row[field]))
            config = runner.SimulationConfig(**run_metadata["config"])
            independent = runner.SpatialPrisonersDilemma(config)
            self.assertEqual(read_grid(run_output / "initial_grid.csv"),
                             [list(line) for line in independent.grid])
            self.assertEqual(read_grid(run_output / "llm_access_grid.csv"),
                             [[str(int(value)) for value in line] for line in independent.llm_access_grid])
            single_root = self.output_root / ("single_" + row["seed"])
            self.assertEqual(self.invoke([
                "--single", "--repeats", "1", "--seed", row["seed"], "--size", "3", "--steps", "3",
                "--b", "1.03", "--k", "0.1", "--lambda", "1", "--p", "0.5",
                "--initial-cooperation", "0.5", "--update-mode", "synchronous",
                "--report-every", "0", "--output-dir", str(single_root),
            ]), 0)
            single_outputs = list(single_root.iterdir())
            self.assertEqual(len(single_outputs), 1)
            self.assertFalse(single_outputs[0].name.startswith("repeat_"))
            for filename in filenames:
                expected = (run_output / filename).read_bytes()
                self.assertEqual(expected, (single_outputs[0] / filename).read_bytes())
                self.assertEqual(expected, (batches[1] / repeated["output_dir"] / filename).read_bytes())

    def test_scan_runs_three_repeats_for_each_parameter_set(self):
        with patch.multiple(runner, SEED=60, SCAN_SEEDS=None):
            self.assertEqual(self.invoke([
                "--scan", "--b-values", "1.01", "1.03", "--k-values", "0.1",
                "--lambda-values", "0", "--p-values", "0.5", "--repeats", "3",
                "--size", "3", "--steps", "2", "--report-every", "0",
                "--output-dir", str(self.output_root),
            ]), 0)
        batch = self.only_batch()
        metadata = read_json(batch / "scan_config.json")
        self.assertEqual(metadata["repeats"], 3)
        self.assertEqual(metadata["parameter_sets"], 2)
        self.assertEqual(metadata["total_runs"], 6)
        self.assertEqual(metadata["completed_runs"], 6)
        self.assertEqual(metadata["seed_values"], [60, 61, 62])
        rows = read_csv(batch / "summary.csv")
        self.assertEqual(len(rows), 6)
        self.assertEqual([(int(row["parameter_set_id"]), int(row["repeat_index"]), int(row["seed"]))
                          for row in rows],
                         [(parameter_id, repeat, 59 + repeat)
                          for parameter_id in (1, 2) for repeat in (1, 2, 3)])
        for row in rows:
            self.assertEqual(float(row["b"]), {1: 1.01, 2: 1.03}[int(row["parameter_set_id"])])
            self.assertEqual(int(row["repeats"]), 3)
            metadata = read_json(batch / row["output_dir"] / "config.json")
            self.assertEqual(metadata["config"]["seed"], int(row["seed"]))
            for field in ("parameter_set_id", "repeat_index", "repeats"):
                self.assertEqual(metadata[field], int(row[field]))
                self.assertEqual(metadata["summary"][field], int(row[field]))

    def test_invalid_repeats_and_explicit_seed_lists_are_rejected_before_outputs(self):
        invalid_cli = [
            ["--single", "--repeats", "0"], ["--single", "--repeats", "-1"],
            ["--single", "--repeats", "1.5"],
            ["--scan", "--repeats", "2", "--seed-values", "41"],
            ["--scan", "--repeats", "1", "--seed-values", "41", "42"],
            ["--scan", "--repeats", "2", "--seed-values", "41", "41"],
            ["--single", "--repeats", "3", "--seed", "-1"],
            ["--scan", "--repeats", "2", "--seed-values", "-42", "42"],
        ]
        for index, arguments in enumerate(invalid_cli):
            with self.subTest(arguments=arguments):
                output = self.output_root / ("invalid_repeat_cli_" + str(index))
                with self.assertRaises(SystemExit) as raised:
                    self.invoke(arguments + ["--output-dir", str(output)])
                self.assertEqual(raised.exception.code, 2)
                self.assertFalse(output.exists())
        for index, repeats in enumerate((0, -1, 1.5, True)):
            with self.subTest(repeats=repeats):
                output = self.output_root / ("invalid_repeat_source_" + str(index))
                with patch.object(runner, "REPEATS", repeats):
                    with self.assertRaises(SystemExit) as raised:
                        self.invoke(["--single", "--output-dir", str(output)])
                self.assertEqual(raised.exception.code, 2)
                self.assertFalse(output.exists())
        for index, seeds in enumerate(([11], [11, 11], [], [11, 12.5])):
            with self.subTest(seeds=seeds):
                output = self.output_root / ("invalid_seed_source_" + str(index))
                with patch.multiple(runner, REPEATS=2, SCAN_SEEDS=seeds):
                    with self.assertRaises(SystemExit) as raised:
                        self.invoke(["--scan", "--output-dir", str(output)])
                self.assertEqual(raised.exception.code, 2)
                self.assertFalse(output.exists())

    def test_single_negative_seed_remains_supported(self):
        self.assertEqual(self.invoke([
            "--single", "--repeats", "1", "--seed", "-42", "--lambda", "0",
            "--size", "3", "--steps", "1", "--report-every", "0",
            "--output-dir", str(self.output_root),
        ]), 0)
        outputs = list(self.output_root.iterdir())
        self.assertEqual(len(outputs), 1)
        metadata = read_json(outputs[0] / "config.json")
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["config"]["seed"], -42)
        self.assertEqual(metadata["summary"]["seed"], -42)
        self.assertEqual(metadata["repeats"], 1)

    def test_failed_second_repeat_preserves_first_row_and_marks_partial_run(self):
        real_step = runner.SpatialPrisonersDilemma.step
        expected_error = RuntimeError("test failure in second repeat")

        def fail_on_second_repeat(model):
            if model.config.seed == 72 and model.round == 1:
                raise expected_error
            return real_step(model)

        with patch.object(runner.SpatialPrisonersDilemma, "step", autospec=True,
                          side_effect=fail_on_second_repeat):
            with self.assertRaises(RuntimeError) as raised:
                self.invoke([
                    "--single", "--repeats", "3", "--seed", "71", "--lambda", "0",
                    "--size", "3", "--steps", "3", "--report-every", "0",
                    "--output-dir", str(self.output_root),
                ])
        self.assertIs(raised.exception, expected_error)
        batches = list(self.output_root.glob("repeat_*"))
        self.assertEqual(len(batches), 1)
        batch = batches[0]
        metadata = read_json(batch / "repeat_config.json")
        self.assertEqual(metadata["total_runs"], 3)
        self.assertEqual(metadata["completed_runs"], 1)
        self.assertEqual(metadata["failed_run_id"], 2)
        self.assertIn(metadata["status"], ("failed", "failed_or_interrupted"))
        rows = read_csv(batch / "summary.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(int(rows[0]["repeat_index"]), 1)
        self.assertEqual(int(rows[0]["seed"]), 71)
        self.assertTrue((batch / rows[0]["output_dir"] / "final_grid.csv").exists())
        configs = list(batch.glob("*/config.json"))
        self.assertEqual(len(configs), 2)
        failed = [path for path in configs if read_json(path)["status"] != "completed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(read_json(failed[0])["repeat_index"], 2)
        self.assertEqual(read_json(failed[0])["completed_steps"], 1)
        self.assertEqual([int(row["round"]) for row in read_csv(failed[0].parent / "history.csv")], [0, 1])

    def test_default_compact_scan_only_saves_two_files_and_matches_full_output(self):
        parameters = [
            "--b-values", "1.03", "--k-values", "0.1", "--lambda-values", "1",
            "--p-values", "0.5", "--seed-values", "42", "--repeats", "1",
        ]
        with patch.object(runner, "SCAN_SAVE_DETAILS", False):
            compact = self.run_scan(parameters, output=self.output_root / "compact")
            full = self.run_scan(parameters + ["--scan-save-details"], output=self.output_root / "full")
        self.assertFalse(read_json(compact / "scan_config.json")["save_details"])
        self.assertTrue(read_json(full / "scan_config.json")["save_details"])
        compact_rows = read_csv(compact / "summary.csv")
        full_rows = read_csv(full / "summary.csv")
        self.assertEqual(len(compact_rows), 1)
        self.assertEqual(compact_rows, full_rows)
        compact_run = compact / compact_rows[0]["output_dir"]
        full_run = full / full_rows[0]["output_dir"]
        self.assertEqual({path.name for path in compact_run.iterdir()}, {"history.csv", "final_grid.csv"})
        self.assertTrue((full_run / "config.json").exists())
        for filename in ("history.csv", "final_grid.csv"):
            self.assertEqual((compact_run / filename).read_bytes(), (full_run / filename).read_bytes())
        self.assert_summary_matches_history(compact_rows[0], read_csv(compact_run / "history.csv"), tail_count=2)

    def test_compact_scan_failure_preserves_history_and_batch_status(self):
        real_step = runner.SpatialPrisonersDilemma.step
        expected_error = RuntimeError("test compact failure")

        def fail_in_second_run(model):
            if model.config.b == 1.03 and model.round == 1:
                raise expected_error
            return real_step(model)

        with patch.object(runner.SpatialPrisonersDilemma, "step", autospec=True,
                          side_effect=fail_in_second_run):
            with self.assertRaises(RuntimeError) as raised:
                self.invoke(self.scan_arguments() + [
                    "--no-scan-save-details", "--k-values", "0.1", "--lambda-values", "0",
                    "--seed-values", "1", "--repeats", "1",
                ])
        self.assertIs(raised.exception, expected_error)
        batch = self.only_batch()
        metadata = read_json(batch / "scan_config.json")
        self.assertFalse(metadata["save_details"])
        self.assertIn(metadata["status"], ("failed", "failed_or_interrupted"))
        self.assertEqual(metadata["completed_runs"], 1)
        self.assertEqual(metadata["failed_run_id"], 2)
        rows = read_csv(batch / "summary.csv")
        self.assertEqual(len(rows), 1)
        completed = batch / rows[0]["output_dir"]
        self.assertEqual({path.name for path in completed.iterdir()}, {"history.csv", "final_grid.csv"})
        failed = [path for path in batch.iterdir() if path.is_dir() and path != completed]
        self.assertEqual(len(failed), 1)
        self.assertEqual({path.name for path in failed[0].iterdir()}, {"history.csv"})
        self.assertEqual([int(row["round"]) for row in read_csv(failed[0] / "history.csv")], [0, 1])


if __name__ == "__main__":
    unittest.main()
