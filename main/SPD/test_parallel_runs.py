"""使用真正 spawn 进程验证批次并行、可复现性和失败恢复；仅使用合成数据。"""

from contextlib import redirect_stderr, redirect_stdout
import csv
import io
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from main.SPD import run_experiment as runner
from main.SPD.simulation import SimulationConfig


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class ConstantAdvisor:
    """可 pickle 的本地测试建议对象。"""

    def __call__(self, context):
        return "C"

    def metadata(self):
        return {"source": "synthetic_constant"}

    def table_rows(self):
        return [{"samples": 1, "C": 1, "D": 0}]

    def usage_rows(self):
        return [{"draws": 0}]


class SlowAdvisor(ConstantAdvisor):
    def __call__(self, context):
        time.sleep(0.01)
        return "C"


class FailAfterCompletedRun(ConstantAdvisor):
    """等待父进程写入第1次结果后，让第2次运行失败，避免依赖进程启动顺序。"""

    def __init__(self, output_root):
        self.output_root = Path(output_root)

    def __call__(self, context):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            for path in self.output_root.glob("repeat_*/repeat_config.json"):
                try:
                    if 1 in read_json(path).get("completed_run_ids", []):
                        raise RuntimeError("synthetic failure in repeat 2")
                except (OSError, ValueError):
                    pass
            time.sleep(0.01)
        raise RuntimeError("timed out waiting for completed first repeat")


class ParallelRunTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="spd_parallel_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.batch = self.root / "synthetic_batch"
        self.batch.mkdir()
        cases = []
        for symbols in product("CD", repeat=9):
            key = "".join(symbols)
            cooperators = sum(key[index] == "C" for index in (1, 3, 5, 7))
            cases.append({
                "grid_key": key, "grid_rows": [key[:3], key[3:6], key[6:]],
                "current_action": key[4], "cooperating_neighbors": cooperators,
                "defecting_neighbors": 4 - cooperators,
            })
        config = {"batch_id": "parallel_synthetic", "observation_format": "grid_3x3",
                  "b": None, "n": 2, "planned_requests": 1024, "cases": cases}
        (self.batch / "config.json").write_text(json.dumps(config), encoding="utf-8")
        rows = []
        for repetition in range(2):
            for case in cases:
                action = "C" if repetition == 0 else "D"
                rows.append({**case, "grid_rows": "/".join(case["grid_rows"]),
                             "observation_format": "grid_3x3", "request_index": len(rows) + 1,
                             "status": "success", "recommendation": action, "output_text": action})
        with (self.batch / "results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def invoke(self, arguments):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return runner.main(["--advice-mode", "fixed"] + arguments)

    def scan_arguments(self, output, workers, details):
        return [
            "--scan", "--workers", str(workers), "--size", "3", "--steps", "3", "--repeats", "2",
            "--b-values", "1.01", "1.5", "--k-values", "0.1", "--lambda-values", "0", "1",
            "--p-values", "0.5", "--seed-values", "41", "42", "--initial-cooperation", "0.5",
            "--boundary", "periodic", "--update-mode", "synchronous", "--report-every", "0",
            "--paired-fermi-control", "--early-average-steps", "2", "--tail-average-steps", "2",
            "--advisor-batch", str(self.batch), "--output-dir", str(output),
            "--scan-save-details" if details else "--no-scan-save-details",
        ]

    def assert_runs_equal(self, reference, actual):
        self.assertEqual((reference / "summary.csv").read_bytes(), (actual / "summary.csv").read_bytes())
        for row in read_csv(reference / "summary.csv"):
            reference_output = reference / row["output_dir"]
            actual_output = actual / row["output_dir"]
            for filename in ("history.csv", "final_grid.csv"):
                self.assertEqual((reference_output / filename).read_bytes(), (actual_output / filename).read_bytes())
            if (reference_output / "config.json").exists() and (actual_output / "config.json").exists():
                for filename in ("initial_grid.csv", "llm_access_grid.csv", "advisor_table.csv", "advisor_usage.csv"):
                    self.assertEqual((reference_output / filename).exists(), (actual_output / filename).exists())
                    if (reference_output / filename).exists():
                        self.assertEqual((reference_output / filename).read_bytes(),
                                         (actual_output / filename).read_bytes())

    def test_spawn_scan_matches_serial_for_paired_grid_advice_and_both_output_modes(self):
        reference = None
        for details in (False, True):
            mode_reference = None
            for workers in (1, 2):
                with self.subTest(details=details, workers=workers):
                    output = self.root / f"scan_{details}_{workers}"
                    self.assertEqual(self.invoke(self.scan_arguments(output, workers, details)), 0)
                    batch = next(output.glob("scan_*"))
                    metadata = read_json(batch / "scan_config.json")
                    self.assertEqual(metadata["requested_workers"], workers)
                    self.assertEqual(metadata["actual_workers"], workers)
                    if workers == 2:
                        self.assertEqual(metadata["parallel_start_method"], "spawn")
                    self.assertEqual(metadata["status"], "completed")
                    self.assertEqual(metadata["completed_run_ids"], list(range(1, 9)))
                    rows = read_csv(batch / "summary.csv")
                    self.assertEqual([int(row["run_id"]) for row in rows], list(range(1, 9)))
                    self.assertEqual({int(row["seed"]) for row in rows}, {41, 42})
                    self.assertEqual({float(row["b"]) for row in rows}, {1.01, 1.5})
                    for row in rows:
                        run_output = batch / row["output_dir"]
                        self.assertNotEqual(row["control_population_cumulative_mean_payoff"], "")
                        if not details:
                            self.assertEqual({path.name for path in run_output.iterdir()},
                                             {"history.csv", "final_grid.csv"})
                        elif float(row["lambda_prob"]) == 1:
                            self.assertEqual(len(read_csv(run_output / "advisor_table.csv")), 512)
                            self.assertEqual(sum(int(item["draws"]) for item in
                                                 read_csv(run_output / "advisor_usage.csv")), 12)
                        if float(row["lambda_prob"]) == 1:
                            self.assertEqual(int(row["total_advice_used"]), 12)
                    if reference is None:
                        reference = batch
                    else:
                        self.assert_runs_equal(reference, batch)
                    if mode_reference is None:
                        mode_reference = batch
                    else:
                        self.assert_runs_equal(mode_reference, batch)

    def test_script_entry_from_spd_directory_spawns_repeats_and_matches_serial(self):
        common = [
            "--advice-mode", "fixed", "--single", "--repeats", "2", "--seed", "42", "--size", "3", "--steps", "3",
            "--lambda", "1", "--p", "0.5", "--b", "1.5", "--k", "0.1",
            "--initial-cooperation", "0.5", "--boundary", "periodic", "--update-mode", "synchronous",
            "--paired-fermi-control", "--early-average-steps", "2", "--tail-average-steps", "2",
            "--report-every", "0", "--advisor-batch", str(self.batch),
        ]
        serial_root = self.root / "serial_repeat"
        parallel_root = self.root / "parallel_repeat"
        self.assertEqual(self.invoke(common + ["--workers", "1", "--output-dir", str(serial_root)]), 0)
        result = subprocess.run(
            [sys.executable, "run_experiment.py"] + common
            + ["--workers", "2", "--output-dir", str(parallel_root)],
            cwd=Path(runner.__file__).resolve().parent,
            capture_output=True, text=True, timeout=40,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        reference = next(serial_root.glob("repeat_*"))
        actual = next(parallel_root.glob("repeat_*"))
        self.assert_runs_equal(reference, actual)
        metadata = read_json(actual / "repeat_config.json")
        self.assertEqual(metadata["actual_workers"], 2)
        self.assertEqual(metadata["parallel_start_method"], "spawn")
        self.assertEqual(metadata["completed_run_ids"], [1, 2])
        self.assertEqual(metadata["seed_values"], [42, 43])
        for row in read_csv(actual / "summary.csv"):
            self.assertTrue((actual / row["output_dir"] / "config.json").exists())

    def test_worker_validation_and_cap_happen_before_execution(self):
        for value in ("0", "-1", "1.5"):
            with self.subTest(workers=value):
                output = self.root / ("invalid_" + value)
                with self.assertRaises(SystemExit) as raised:
                    self.invoke(["--single", "--workers", value, "--output-dir", str(output)])
                self.assertEqual(raised.exception.code, 2)
                self.assertFalse(output.exists())
        windows_output = self.root / "windows_limit"
        with patch.object(runner.sys, "platform", "win32"):
            with self.assertRaises(SystemExit) as raised:
                self.invoke([
                    "--single", "--workers", "62", "--repeats", "62", "--seed", "0",
                    "--size", "3", "--steps", "0", "--lambda", "0",
                    "--output-dir", str(windows_output),
                ])
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(windows_output.exists())
        output = self.root / "capped"
        arguments = self.scan_arguments(output, workers=4, details=False)
        self.assertEqual(self.invoke(arguments + [
            "--b-values", "1.01", "--lambda-values", "0", "--seed-values", "42", "--repeats", "1",
        ]), 0)
        metadata = read_json(next(output.glob("scan_*")) / "scan_config.json")
        self.assertEqual(metadata["requested_workers"], 4)
        self.assertEqual(metadata["actual_workers"], 1)
        self.assertEqual(metadata["completed_run_ids"], [1])

    def test_spawn_failure_preserves_completed_rows_and_does_not_submit_entire_batch(self):
        from main.SPD.parallel_execution import ParallelRunError

        output = self.root / "failed_repeats"

        def make_advisor(config, batch):
            if config.seed == 42:
                return ConstantAdvisor()
            if config.seed == 43:
                return FailAfterCompletedRun(output)
            return SlowAdvisor()

        with patch.object(runner, "create_advisor", side_effect=make_advisor):
            with self.assertRaises(ParallelRunError) as raised:
                self.invoke([
                    "--single", "--workers", "2", "--repeats", "10", "--seed", "42",
                    "--lambda", "1", "--p", "1", "--size", "3", "--steps", "2",
                    "--update-mode", "synchronous", "--report-every", "0", "--no-paired-fermi-control",
                    "--advisor-batch", str(self.batch), "--output-dir", str(output),
                ])
        self.assertEqual(raised.exception.run_id, 2)
        self.assertIn("synthetic failure in repeat 2", str(raised.exception))
        batch = next(output.glob("repeat_*"))
        metadata = read_json(batch / "repeat_config.json")
        self.assertIn(metadata["status"], ("failed", "failed_or_interrupted"))
        self.assertEqual(metadata["failed_run_id"], 2)
        self.assertEqual(metadata["completed_run_ids"], [1])
        self.assertEqual(metadata["completed_runs"], 1)
        self.assertEqual([int(row["run_id"]) for row in read_csv(batch / "summary.csv")], [1])
        run_outputs = [path for path in batch.iterdir() if path.is_dir()]
        # 初始只提交两个；第1次完成后最多再补一个，失败后不得继续提交剩余七次。
        self.assertLessEqual(len(run_outputs), 3)
        complete = [path for path in run_outputs if read_json(path / "config.json")["status"] == "completed"]
        self.assertEqual(len(complete), 1)
        self.assertTrue((complete[0] / "final_grid.csv").exists())

    def test_parent_interruption_stops_bounded_spawn_work_and_preserves_delivered_result(self):
        from main.SPD.parallel_execution import run_parallel

        tasks = []
        for run_id in range(1, 9):
            tasks.append((run_id, dict(
                config=SimulationConfig(size=3, lambda_prob=1, seed=run_id, update_mode="synchronous"),
                steps=1 if run_id == 1 else 100,
                output=self.root / f"interrupt_{run_id}", report_every=0, tail_average_steps=2,
                advisor=ConstantAdvisor() if run_id == 1 else SlowAdvisor(),
                save_details=False, paired_control=False, early_average_steps=2,
            )))
        completed = []

        def interrupt_after_result(run_id, summary):
            completed.append((run_id, summary))
            raise KeyboardInterrupt("synthetic parent interruption")

        with self.assertRaises(KeyboardInterrupt):
            run_parallel(tasks, num_workers=2, on_result=interrupt_after_result)
        self.assertEqual([run_id for run_id, _ in completed], [1])
        self.assertEqual(completed[0][1]["status"], "completed")
        self.assertTrue((self.root / "interrupt_1" / "final_grid.csv").exists())
        self.assertLessEqual(len(list(self.root.glob("interrupt_*"))), 2)

    def test_interruption_while_saving_summary_recovers_collected_success(self):
        output = self.root / "interrupted_summary"
        original_replace = Path.replace
        interrupted = []

        def interrupt_first_result(path, target):
            if path.name == "summary.csv.tmp" and not interrupted and read_csv(path):
                interrupted.append(True)
                raise KeyboardInterrupt("interrupted before replacing summary")
            return original_replace(path, target)

        def make_advisor(config, batch):
            return ConstantAdvisor() if config.seed == 42 else SlowAdvisor()

        with patch.object(Path, "replace", interrupt_first_result), \
                patch.object(runner, "create_advisor", side_effect=make_advisor):
            with self.assertRaises(KeyboardInterrupt):
                self.invoke([
                    "--single", "--workers", "2", "--repeats", "4", "--seed", "42",
                    "--lambda", "1", "--p", "1", "--size", "3", "--steps", "10",
                    "--update-mode", "synchronous", "--report-every", "0", "--no-paired-fermi-control",
                    "--advisor-batch", str(self.batch), "--output-dir", str(output),
                ])
        self.assertTrue(interrupted)
        batch = next(output.glob("repeat_*"))
        metadata = read_json(batch / "repeat_config.json")
        self.assertEqual(metadata["status"], "failed_or_interrupted")
        self.assertEqual(metadata["completed_run_ids"], [1])
        self.assertEqual(metadata["completed_runs"], 1)
        self.assertEqual([int(row["run_id"]) for row in read_csv(batch / "summary.csv")], [1])
        self.assertFalse((batch / "summary.csv.tmp").exists())

    def test_locked_summary_keeps_spawn_batch_running_and_reports_complete_backup(self):
        output = self.root / "locked_summary"
        original_replace = Path.replace
        original_primary = []
        terminal = io.StringIO()

        def deny_primary_updates(path, target):
            target = Path(target)
            if target.name == "summary.csv":
                if target.exists():
                    if not original_primary:
                        original_primary.append(target.read_bytes())
                    raise PermissionError("synthetic Windows summary.csv lock")
            return original_replace(path, target)

        arguments = self.scan_arguments(output, workers=2, details=False)
        with patch.object(Path, "replace", deny_primary_updates), \
                redirect_stdout(terminal), redirect_stderr(terminal):
            self.assertEqual(runner.main(["--advice-mode", "fixed"] + arguments), 0)

        batch = next(output.glob("scan_*"))
        metadata = read_json(batch / "scan_config.json")
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["completed_runs"], 8)
        self.assertEqual(metadata["completed_run_ids"], list(range(1, 9)))
        self.assertEqual(metadata["actual_workers"], 2)
        self.assertEqual(metadata["parallel_start_method"], "spawn")
        self.assertIsNone(metadata["failed_run_id"])
        self.assertTrue(metadata["summary_pending"])
        latest = batch / metadata["latest_summary_file"]
        self.assertEqual(latest.name, "summary.pending.csv")
        self.assertEqual([int(row["run_id"]) for row in read_csv(latest)], list(range(1, 9)))
        self.assertEqual((batch / "summary.csv").read_bytes(), original_primary[0])
        self.assertEqual(read_csv(batch / "summary.csv"), [])
        self.assertFalse((batch / "summary.csv.tmp").exists())
        for row in read_csv(latest):
            self.assertTrue((batch / row["output_dir"] / "final_grid.csv").exists())
        completion_lines = [line for line in terminal.getvalue().splitlines() if "结果见" in line]
        self.assertEqual(len(completion_lines), 1)
        self.assertIn(str(latest), completion_lines[0])


if __name__ == "__main__":
    unittest.main()
