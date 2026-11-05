"""扫描轨迹精简和采样只能影响输出，不能改变逐轮统计、随机流或配对对照。"""

from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from main.SPD import run_experiment as runner


COMPACT_FIELDS = (
    "round", "cooperation_rate", "average_payoff", "eligible_mean_payoff",
    "ineligible_mean_payoff", "control_cooperation_rate", "control_average_payoff",
    "control_eligible_mean_payoff", "control_ineligible_mean_payoff",
    "eligible_trust_mean", "eligible_trust_std", "eligible_advice_rate",
    "eligible_feedback_reference_mean", "eligible_aspiration_mean", "advice_used",
    "trust_positive_feedback", "trust_negative_feedback", "trust_zero_feedback",
)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class PeriodicAdvisor:
    """可 pickle 的确定性建议器，让轨迹包含非平稳收益和信任反馈。"""

    def __call__(self, context):
        return "D" if context.round % 3 == 0 else "C"

    def metadata(self):
        return {"source": "synthetic_periodic"}

    def table_rows(self):
        return [{"source": "synthetic_periodic"}]

    def usage_rows(self):
        return [{"source": "synthetic_periodic"}]


class ScanHistoryIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="spd_scan_history_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def invoke(self, output, extra=()):
        arguments = [
            "--scan", "--advice-mode", "adaptive", "--workers", "1", "--repeats", "2",
            "--size", "3", "--steps", "7", "--b-values", "1.03", "--k-values", "0.1",
            "--p-values", "0.5", "--seed-values", "13", "14", "--initial-trust-values", "0.6",
            "--trust-learning-rate-values", "0.2", "--trust-payoff-window", "3",
            "--trust-feedback-threshold", "0.01", "--initial-cooperation", "0.5",
            "--boundary", "periodic", "--update-mode", "synchronous", "--paired-fermi-control",
            "--early-average-steps", "2", "--tail-average-steps", "4", "--report-every", "0",
            "--no-scan-save-details", "--output-dir", str(output),
        ]
        with patch.object(runner, "create_advisor", side_effect=lambda *_: PeriodicAdvisor()), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return runner.main(arguments + list(extra))

    def run_scan(self, name, extra=()):
        output = self.root / name
        self.assertEqual(self.invoke(output, extra), 0)
        return next(output.glob("scan_*"))

    def assert_scientific_outputs_equal(self, reference, actual):
        self.assertEqual((reference / "summary.csv").read_bytes(),
                         (actual / "summary.csv").read_bytes())
        _, rows = read_csv(reference / "summary.csv")
        for row in rows:
            self.assertEqual((reference / row["output_dir"] / "final_grid.csv").read_bytes(),
                             (actual / row["output_dir"] / "final_grid.csv").read_bytes())

    def test_full_compact_and_none_have_identical_scientific_summary_and_final_grids(self):
        full = self.run_scan("full", ["--scan-history-mode", "full", "--scan-history-every", "1"])
        compact = self.run_scan("compact", ["--scan-history-mode", "compact", "--scan-history-every", "3"])
        disabled = self.run_scan("none", ["--scan-history-mode", "none", "--scan-history-every", "3"])
        self.assert_scientific_outputs_equal(full, compact)
        self.assert_scientific_outputs_equal(full, disabled)
        fields, rows = read_csv(full / "summary.csv")
        self.assertNotIn("history_mode", fields)
        self.assertNotIn("history_every", fields)
        for item in rows:
            full_fields, original = read_csv(full / item["output_dir"] / "history.csv")
            fields, sampled = read_csv(compact / item["output_dir"] / "history.csv")
            self.assertGreater(len(full_fields), len(fields))
            self.assertEqual(tuple(fields), COMPACT_FIELDS)
            self.assertEqual([int(row["round"]) for row in original], list(range(8)))
            self.assertEqual([int(row["round"]) for row in sampled], [0, 3, 6, 7])
            self.assertEqual(sampled, [
                {field: original[number][field] for field in COMPACT_FIELDS}
                for number in (0, 3, 6, 7)
            ])
            self.assertEqual({path.name for path in (disabled / item["output_dir"]).iterdir()},
                             {"final_grid.csv"})
            self.assertEqual({path.name for path in (compact / item["output_dir"]).iterdir()},
                             {"history.csv", "final_grid.csv"})

    def test_full_mode_also_supports_sampling_and_always_saves_successful_final_round(self):
        batch = self.run_scan("full_sampled", [
            "--scan-history-mode", "full", "--scan-history-every", "4", "--scan-save-details",
        ])
        metadata = read_json(batch / "scan_config.json")
        self.assertEqual(metadata["history_mode"], "full")
        self.assertEqual(metadata["history_every"], 4)
        self.assertIn("history_sampling_semantics", metadata)
        _, rows = read_csv(batch / "summary.csv")
        for item in rows:
            output = batch / item["output_dir"]
            fields, history = read_csv(output / "history.csv")
            self.assertEqual([int(row["round"]) for row in history], [0, 4, 7])
            self.assertIn("population_cumulative_mean_payoff", fields)
            self.assertIn("eligible_trust_bin_9", fields)
            saved = read_json(output / "config.json")["history"]
            self.assertEqual(saved["mode"], "full")
            self.assertEqual(saved["every"], 4)
            self.assertEqual(saved["fields"], fields)
            self.assertEqual(saved["rows_written"], 3)

    def test_editable_scan_defaults_choose_compact_with_one_row_every_ten_rounds(self):
        with patch.multiple(runner, SCAN_HISTORY_MODE="compact", SCAN_HISTORY_EVERY=10):
            batch = self.run_scan("defaults")
        metadata = read_json(batch / "scan_config.json")
        self.assertEqual(metadata["history_mode"], "compact")
        self.assertEqual(metadata["history_every"], 10)
        _, rows = read_csv(batch / "summary.csv")
        for item in rows:
            fields, history = read_csv(batch / item["output_dir"] / "history.csv")
            self.assertEqual(tuple(fields), COMPACT_FIELDS)
            self.assertEqual([int(row["round"]) for row in history], [0, 7])

    def test_zero_steps_save_initial_state_once_and_none_creates_no_history(self):
        for mode in ("full", "compact", "none"):
            with self.subTest(mode=mode):
                batch = self.run_scan("zero_" + mode, [
                    "--steps", "0", "--scan-history-mode", mode, "--scan-history-every", "3",
                    "--scan-save-details",
                ])
                _, rows = read_csv(batch / "summary.csv")
                for item in rows:
                    output = batch / item["output_dir"]
                    metadata = read_json(output / "config.json")
                    self.assertEqual(metadata["status"], "completed")
                    self.assertEqual(metadata["completed_steps"], 0)
                    self.assertEqual(metadata["history"]["rows_written"], 0 if mode == "none" else 1)
                    if mode == "none":
                        self.assertFalse((output / "history.csv").exists())
                    else:
                        _, history = read_csv(output / "history.csv")
                        self.assertEqual([row["round"] for row in history], ["0"])

    def test_compact_represents_absent_control_and_empty_eligible_group_as_blank(self):
        batch = self.run_scan("no_group", [
            "--p-values", "0", "--no-paired-fermi-control", "--scan-history-mode", "compact",
            "--scan-history-every", "2",
        ])
        _, rows = read_csv(batch / "summary.csv")
        for item in rows:
            _, history = read_csv(batch / item["output_dir"] / "history.csv")
            for row in history:
                for name in ("eligible_mean_payoff", "eligible_trust_mean", "eligible_trust_std",
                             "eligible_advice_rate", "eligible_feedback_reference_mean",
                             "eligible_aspiration_mean", "control_average_payoff",
                             "control_cooperation_rate", "control_eligible_mean_payoff",
                             "control_ineligible_mean_payoff"):
                    self.assertEqual(row[name], "", name)

    def test_invalid_history_settings_fail_preflight_before_creating_output(self):
        for index, extra in enumerate((
                ["--scan-history-mode", "other"], ["--scan-history-every", "0"],
                ["--scan-history-every", "-1"], ["--scan-history-every", "1.5"])):
            with self.subTest(extra=extra):
                output = self.root / f"invalid_{index}"
                with self.assertRaises(SystemExit) as raised:
                    self.invoke(output, extra)
                self.assertEqual(raised.exception.code, 2)
                self.assertFalse(output.exists())
        for name, value in (("SCAN_HISTORY_MODE", "invalid"), ("SCAN_HISTORY_EVERY", True)):
            with self.subTest(name=name), patch.object(runner, name, value):
                output = self.root / name
                with self.assertRaises(SystemExit) as raised:
                    self.invoke(output)
                self.assertEqual(raised.exception.code, 2)
                self.assertFalse(output.exists())

    def test_single_and_single_repeats_ignore_scan_output_settings(self):
        for repeats in (1, 2):
            with self.subTest(repeats=repeats):
                output = self.root / f"single_{repeats}"
                arguments = [
                    "--single", "--advice-mode", "adaptive", "--workers", "1", "--repeats", str(repeats),
                    "--size", "3", "--steps", "3", "--b", "1.03", "--k", "0.1", "--p", "0.5",
                    "--seed", "13", "--initial-trust", "0.6", "--trust-learning-rate", "0.2",
                    "--trust-payoff-window", "3", "--trust-feedback-threshold", "0.01",
                    "--initial-cooperation", "0.5", "--boundary", "periodic",
                    "--update-mode", "synchronous", "--no-paired-fermi-control", "--report-every", "0",
                    "--scan-history-mode", "none", "--scan-history-every", "99", "--output-dir", str(output),
                ]
                with patch.object(runner, "create_advisor", return_value=PeriodicAdvisor()), \
                        redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(runner.main(arguments), 0)
                histories = list(output.rglob("history.csv"))
                self.assertEqual(len(histories), repeats)
                for path in histories:
                    fields, rows = read_csv(path)
                    self.assertGreater(len(fields), len(COMPACT_FIELDS))
                    self.assertEqual([row["round"] for row in rows], ["0", "1", "2", "3"])
                    self.assertEqual(read_json(path.parent / "config.json")["history"]["mode"], "full")

    def test_compact_and_disabled_scan_match_real_spawn_and_serial(self):
        reference = self.run_scan("spawn_reference", [
            "--scan-history-mode", "full", "--scan-history-every", "1",
        ])
        for mode in ("compact", "none"):
            with self.subTest(mode=mode):
                serial = self.run_scan("serial_" + mode, [
                    "--scan-history-mode", mode, "--scan-history-every", "3", "--workers", "1",
                ])
                parallel = self.run_scan("parallel_" + mode, [
                    "--scan-history-mode", mode, "--scan-history-every", "3", "--workers", "2",
                ])
                self.assert_scientific_outputs_equal(reference, serial)
                self.assert_scientific_outputs_equal(serial, parallel)
                self.assertEqual(read_json(parallel / "scan_config.json")["parallel_start_method"], "spawn")
                if mode == "compact":
                    _, rows = read_csv(serial / "summary.csv")
                    for row in rows:
                        self.assertEqual((serial / row["output_dir"] / "history.csv").read_bytes(),
                                         (parallel / row["output_dir"] / "history.csv").read_bytes())

    def test_failed_scan_flushes_only_completed_sampled_rounds_and_keeps_failure_status(self):
        actual_step = runner.SpatialPrisonersDilemma.step
        failure = RuntimeError("synthetic scan interruption before round 3")

        def failing_step(model):
            if model.round == 2:
                raise failure
            return actual_step(model)

        output = self.root / "failed"
        with patch.object(runner.SpatialPrisonersDilemma, "step", failing_step):
            with self.assertRaises(RuntimeError) as raised:
                self.invoke(output, [
                    "--scan-history-mode", "compact", "--scan-history-every", "2",
                    "--scan-save-details", "--no-paired-fermi-control",
                ])
        self.assertIs(raised.exception, failure)
        batch = next(output.glob("scan_*"))
        metadata = read_json(batch / "scan_config.json")
        self.assertEqual(metadata["status"], "failed_or_interrupted")
        self.assertEqual(metadata["completed_runs"], 0)
        histories = list(batch.rglob("history.csv"))
        self.assertEqual(len(histories), 1)
        _, rows = read_csv(histories[0])
        self.assertEqual([row["round"] for row in rows], ["0", "2"])
        saved = read_json(histories[0].parent / "config.json")
        self.assertEqual(saved["status"], "failed_or_interrupted")
        self.assertEqual(saved["completed_steps"], 2)
        self.assertEqual(saved["history"]["rows_written"], 2)
        self.assertFalse((histories[0].parent / "final_grid.csv").exists())


if __name__ == "__main__":
    unittest.main()
