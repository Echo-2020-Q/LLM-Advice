"""信任反馈容差带的边界、历史时序、CLI 元数据和真正 spawn 回归。"""

from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
import math
import unittest
from unittest.mock import patch

from main.SPD import run_experiment as runner
from main.SPD.simulation import SimulationConfig, SpatialPrisonersDilemma
from main.SPD.test_llm_advisor import BatchFixtureTests


class DecisionDraw:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


class TrustFeedbackThresholdTests(unittest.TestCase):
    def make_model(self, threshold=0.01, adopt=True, reference=1.0, alpha=0.2):
        # 固定收益依赖，使边界值与格点收益矩阵及随机邻域演化无关。
        config = SimulationConfig(
            size=3, seed=29, advice_mode="adaptive", update_mode="synchronous",
            initial_trust=0.5, trust_learning_rate=alpha,
            trust_feedback_threshold=threshold, trust_payoff_window=3,
        )
        with patch.object(SpatialPrisonersDilemma, "_payoff", return_value=reference):
            model = SpatialPrisonersDilemma(
                config, advisor=lambda _: "C", initial_grid=("CCC",) * 3,
            )
        model._advice_rng = DecisionDraw(0.0 if adopt else 0.999)
        return model

    def advance(self, model, payoff):
        with patch.object(model, "_payoff", return_value=payoff):
            return model.step()

    def assert_uniform(self, grid, expected):
        for row in grid:
            for value in row:
                self.assertAlmostEqual(value, expected)

    def assert_feedback(self, model, positive=0, negative=0, zero=0):
        statistics = model.trust_statistics()
        self.assertEqual(statistics["trust_positive_feedback"], positive)
        self.assertEqual(statistics["trust_negative_feedback"], negative)
        self.assertEqual(statistics["trust_zero_feedback"], zero)

    def test_default_is_one_hundredth_and_threshold_must_be_finite_nonnegative(self):
        self.assertEqual(SimulationConfig().trust_feedback_threshold, 0.01)
        for value in (-0.01, math.nan, math.inf, -math.inf, True, False):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                SimulationConfig(trust_feedback_threshold=value)
        for value in (0, 0.01, 100):
            self.assertEqual(SimulationConfig(trust_feedback_threshold=value).trust_feedback_threshold,
                             value)

    def test_exact_boundaries_and_values_inside_band_keep_trust_unchanged(self):
        # 1.01 - 1.0 > .01 在二进制浮点中可能成立，边界也必须视作无变化。
        for payoff in (1.01, 0.99, 1.0, math.nextafter(1.01, 1.0),
                       math.nextafter(0.99, 1.0)):
            with self.subTest(payoff=payoff):
                model = self.make_model()
                statistics = self.advance(model, payoff)
                self.assertEqual(statistics.advice_used, 9)
                self.assert_uniform(model.trust_grid, 0.5)
                self.assert_feedback(model, zero=9)
                self.assert_uniform(model.feedback_reference_grid, 1.0)
                # 即便反馈在容差带内，本轮原始收益依旧进入历史。
                self.assert_uniform(model.aspiration_grid, (1.0 + payoff) / 2)

    def test_immediately_outside_band_updates_in_correct_direction(self):
        for payoff, expected_trust, direction in (
                (math.nextafter(1.01, math.inf), 0.6, "positive"),
                (math.nextafter(0.99, -math.inf), 0.4, "negative")):
            with self.subTest(payoff=payoff):
                model = self.make_model()
                self.advance(model, payoff)
                self.assert_uniform(model.trust_grid, expected_trust)
                self.assert_feedback(model, **{direction: 9})

    def test_zero_threshold_restores_feedback_for_arbitrarily_small_nonzero_change(self):
        for payoff, expected_trust, direction in (
                (math.nextafter(1.0, math.inf), 0.6, "positive"),
                (math.nextafter(1.0, -math.inf), 0.4, "negative"),
                (1.0, 0.5, "zero")):
            with self.subTest(payoff=payoff):
                model = self.make_model(threshold=0)
                self.advance(model, payoff)
                self.assert_uniform(model.trust_grid, expected_trust)
                self.assert_feedback(model, **{direction: 9})

    def test_threshold_only_classifies_feedback_and_alpha_controls_update_size(self):
        for alpha in (0.0, 0.1, 1.0):
            with self.subTest(alpha=alpha):
                positive = self.make_model(alpha=alpha)
                negative = self.make_model(alpha=alpha)
                self.advance(positive, 1.02)
                self.advance(negative, 0.98)
                self.assert_uniform(positive.trust_grid, 0.5 + alpha * 0.5)
                self.assert_uniform(negative.trust_grid, (1 - alpha) * 0.5)
                self.assert_feedback(positive, positive=9)
                self.assert_feedback(negative, negative=9)

    def test_band_does_not_skip_history_and_history_excludes_current_payoff(self):
        model = self.make_model(threshold=0.01)
        self.advance(model, 1.005)
        self.assert_feedback(model, zero=9)
        self.advance(model, 1.02)
        self.assert_uniform(model.feedback_reference_grid, 1.0025)
        self.assert_uniform(model.trust_grid, 0.6)
        self.assert_feedback(model, positive=9)
        self.assert_uniform(model.aspiration_grid, (1 + 1.005 + 1.02) / 3)

    def test_without_adoption_no_feedback_is_counted_and_history_still_rolls(self):
        model = self.make_model(adopt=False)
        history = [1.0]
        for payoff in (1.005, 2.0, 0.0, 3.0):
            reference = math.fsum(history[-3:]) / len(history[-3:])
            statistics = self.advance(model, payoff)
            self.assertEqual(statistics.advice_used, 0)
            self.assertEqual(statistics.fermi_updates, 9)
            self.assert_uniform(model.trust_grid, 0.5)
            self.assert_feedback(model)
            self.assert_uniform(model.feedback_reference_grid, reference)
            history.append(payoff)
            self.assert_uniform(model.aspiration_grid,
                                math.fsum(history[-3:]) / len(history[-3:]))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class TrustFeedbackThresholdIntegrationTests(BatchFixtureTests):
    def invoke(self, output, extra=()):
        arguments = [
            "--single", "--advice-mode", "adaptive", "--workers", "1", "--repeats", "1",
            "--size", "3", "--steps", "5", "--b", "1.5", "--k", "0.1", "--seed", "13",
            "--p", "1", "--initial-cooperation", "0.5", "--boundary", "periodic",
            "--update-mode", "synchronous", "--initial-trust", "0.6",
            "--trust-learning-rate", "0.2", "--trust-payoff-window", "3",
            "--paired-fermi-control", "--early-average-steps", "2", "--tail-average-steps", "2",
            "--report-every", "0", "--advisor-batch", str(self.batch), "--output-dir", str(output),
        ]
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return runner.main(arguments + list(extra))

    def test_source_default_and_cli_override_are_saved_in_per_run_config_and_summary(self):
        self.assertEqual(runner.TRUST_FEEDBACK_THRESHOLD, 0.01)
        with patch.object(runner, "TRUST_FEEDBACK_THRESHOLD", 0.123):
            for name, extra, expected in (
                    ("source", (), 0.123),
                    ("override", ("--trust-feedback-threshold", "0.456"), 0.456)):
                with self.subTest(name=name):
                    output = self.root / name
                    self.assertEqual(self.invoke(output, extra), 0)
                    metadata = read_json(next(output.iterdir()) / "config.json")
                    self.assertEqual(metadata["config"]["trust_feedback_threshold"], expected)
                    self.assertEqual(metadata["summary"]["trust_feedback_threshold"], expected)

    def test_large_threshold_counts_adopted_dead_zone_feedback_and_history_continues(self):
        output = self.root / "wide_band"
        self.assertEqual(self.invoke(output, ["--trust-feedback-threshold", "100"]), 0)
        directory = next(output.iterdir())
        metadata = read_json(directory / "config.json")
        history = read_csv(directory / "history.csv")
        total_adopted = 0
        for row in history:
            adopted = int(row["advice_used"])
            total_adopted += adopted
            self.assertAlmostEqual(float(row["eligible_trust_mean"]), 0.6)
            self.assertEqual(int(row["trust_positive_feedback"]), 0)
            self.assertEqual(int(row["trust_negative_feedback"]), 0)
            self.assertEqual(int(row["trust_zero_feedback"]), adopted)
        self.assertGreater(total_adopted, 0)
        summary = metadata["summary"]
        self.assertEqual(summary["total_trust_zero_feedback"], total_adopted)
        self.assertEqual(summary["total_trust_positive_feedback"], 0)
        self.assertEqual(summary["total_trust_negative_feedback"], 0)
        self.assertGreater(len({row["eligible_aspiration_mean"] for row in history}), 1)

    def test_invalid_threshold_is_rejected_before_creating_any_output(self):
        for value in ("-0.01", "nan", "inf", "-inf"):
            with self.subTest(value=value):
                output = self.root / ("invalid_" + value)
                with self.assertRaises(SystemExit) as raised:
                    self.invoke(output, ["--trust-feedback-threshold=" + value])
                self.assertEqual(raised.exception.code, 2)
                self.assertFalse(output.exists())

    def test_scan_keeps_threshold_fixed_and_spawn_matches_serial(self):
        directories = []
        for workers in (1, 2):
            output = self.root / ("scan_" + str(workers))
            arguments = [
                "--scan", "--scan-history-mode", "full", "--scan-history-every", "1",
                "--advice-mode", "adaptive", "--workers", str(workers),
                "--repeats", "1", "--size", "3", "--steps", "5",
                "--b-values", "1.5", "--k-values", "0.1", "--p-values", "1",
                "--seed-values", "13", "--initial-trust-values", "0.4", "0.8",
                "--trust-learning-rate-values", "0.2", "--trust-payoff-window", "3",
                "--trust-feedback-threshold", "0.2", "--initial-cooperation", "0.5",
                "--boundary", "periodic", "--update-mode", "synchronous",
                "--paired-fermi-control", "--early-average-steps", "2", "--tail-average-steps", "2",
                "--report-every", "0", "--advisor-batch", str(self.batch),
                "--no-scan-save-details", "--output-dir", str(output),
            ]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(runner.main(arguments), 0)
            directory = next(output.glob("scan_*"))
            directories.append(directory)
            metadata = read_json(directory / "scan_config.json")
            self.assertEqual(metadata["fixed_parameters"]["trust_feedback_threshold"], 0.2)
            self.assertNotIn("trust_feedback_threshold", metadata["scan_values"])
            self.assertEqual(metadata["total_runs"], 2)
            self.assertEqual(metadata["actual_workers"], workers)
            if workers == 2:
                self.assertEqual(metadata["parallel_start_method"], "spawn")
            rows = read_csv(directory / "summary.csv")
            self.assertEqual(len(rows), 2)
            self.assertEqual({float(row["trust_feedback_threshold"]) for row in rows}, {0.2})
        serial, parallel = directories
        self.assertEqual((serial / "summary.csv").read_bytes(), (parallel / "summary.csv").read_bytes())
        for row in read_csv(serial / "summary.csv"):
            for filename in ("history.csv", "final_grid.csv"):
                self.assertEqual((serial / row["output_dir"] / filename).read_bytes(),
                                 (parallel / row["output_dir"] / filename).read_bytes())


if __name__ == "__main__":
    unittest.main()
