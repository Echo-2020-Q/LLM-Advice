"""信任学习使用近期已完成回合的均值，博弈与 Fermi 仍用瞬时收益。"""

from contextlib import redirect_stderr, redirect_stdout
import csv
from dataclasses import replace
import io
import json
import math
import unittest
from unittest.mock import patch

from main.SPD import run_experiment as runner
from main.SPD import simulation
from main.SPD.simulation import SimulationConfig, SpatialPrisonersDilemma
from main.SPD.test_llm_advisor import BatchFixtureTests


class AlwaysAdopt:
    def random(self):
        return 0.0


class TrustPayoffWindowTests(unittest.TestCase):
    def model(self, advisor=None, **changes):
        values = dict(size=3, seed=29, update_mode="synchronous", advice_mode="adaptive",
                      initial_trust=0.5, trust_learning_rate=0.2,
                      aspiration_learning_rate=0.2, trust_payoff_window=3)
        values.update(changes)
        advisor = advisor or (lambda context: "D" if context.round % 2 else "C")
        model = SpatialPrisonersDilemma(SimulationConfig(**values), advisor=advisor,
                                       initial_grid=("CCC",) * 3)
        model._advice_rng = AlwaysAdopt()
        return model

    def assert_uniform(self, grid, expected):
        self.assertEqual(len(grid), 3)
        for row in grid:
            for value in row:
                self.assertAlmostEqual(value, expected)

    def test_window_warmup_excludes_initial_payoff_and_discards_oldest_round(self):
        model = self.model()
        self.assert_uniform(model.aspiration_grid, 4.0)
        self.assertIsNone(model.trust_payoff_grid)
        self.assertIsNone(model.trust_statistics()["eligible_feedback_payoff_mean"])
        # 初态收益为 4；完成回合收益依次 0,4,0,4。第 0 轮不加入窗口。
        for expected in (0, 2, 4 / 3, 8 / 3):
            model.step()
            self.assert_uniform(model.trust_payoff_grid, expected)
            self.assertAlmostEqual(model.trust_statistics()["eligible_feedback_payoff_mean"],
                                   expected)
        snapshot = model.trust_payoff_grid
        with self.assertRaises(TypeError):
            snapshot[0][0] = 999
        model.step()
        self.assert_uniform(snapshot, 8 / 3)
        self.assert_uniform(model.trust_payoff_grid, 4 / 3)

    def test_mean_can_reverse_feedback_direction_from_instantaneous_payoff(self):
        model = self.model()
        model.step()  # 收益 0，旧参考 4，信任 0.4，新参考 3.2。
        model.step()  # 当前收益 4，但窗口均值 2 < 旧参考 3.2，仍应负反馈。
        self.assert_uniform(model.trust_grid, 0.32)
        self.assert_uniform(model.aspiration_grid, 2.96)
        self.assertEqual(model.measure().average_payoff, 4.0)
        self.assertEqual(model.payoff(1, 1), 4.0)
        self.assertEqual(model.trust_statistics()["trust_negative_feedback"], 9)
        self.assertEqual(model.trust_statistics()["trust_positive_feedback"], 0)

    def test_beta_one_feedback_compares_with_old_reference_before_replacement(self):
        model = self.model(aspiration_learning_rate=1)
        expected = ((0, 0.4, "negative"), (2, 0.52, "positive"),
                    (4 / 3, 0.416, "negative"), (8 / 3, 0.5328, "positive"))
        for reference, trust, direction in expected:
            model.step()
            self.assert_uniform(model.aspiration_grid, reference)
            self.assert_uniform(model.trust_grid, trust)
            self.assertEqual(model.trust_statistics()[f"trust_{direction}_feedback"], 9)

    def test_fallback_rounds_still_enter_window_and_update_all_references(self):
        def advisor(context):
            if context.round == 1:
                return "D" if context.position == (1, 1) else "C"
            return None

        model = self.model(advisor=advisor, aspiration_learning_rate=0.25)
        past = []
        expected_reference = [[4.0] * 3 for _ in range(3)]
        after_adoption = None
        found_mean_different_from_current = False
        for round_number in range(1, 5):
            stats = model.step()
            past.append([[model.payoff(r, c) for c in range(3)] for r in range(3)])
            window = past[-3:]
            for row in range(3):
                for col in range(3):
                    average = math.fsum(payoffs[row][col] for payoffs in window) / len(window)
                    found_mean_different_from_current |= (
                        abs(average - past[-1][row][col]) > 1e-10
                    )
                    expected_reference[row][col] = (
                        0.75 * expected_reference[row][col] + 0.25 * average
                    )
                    self.assertAlmostEqual(model.trust_payoff_grid[row][col], average)
                    self.assertAlmostEqual(model.aspiration_grid[row][col],
                                           expected_reference[row][col])
            if round_number == 1:
                after_adoption = model.trust_grid
            self.assertEqual(model.trust_grid, after_adoption)
            if round_number > 1:
                self.assertEqual(stats.advice_used, 0)
                self.assertEqual(stats.fermi_updates, 9)
                for direction in ("positive", "negative", "zero"):
                    self.assertEqual(model.trust_statistics()[f"trust_{direction}_feedback"], 0)
        self.assertTrue(found_mean_different_from_current)

    def test_fermi_reads_current_payoffs_even_when_last_feedback_mean_differs(self):
        def advisor(context):
            return {1: "D", 2: "C"}.get(context.round)

        model = self.model(advisor=advisor)
        model.step()
        model.step()
        self.assert_uniform(model.trust_payoff_grid, 2)
        with patch.object(simulation, "fermi_probability", wraps=simulation.fermi_probability) as fermi:
            stats = model.step()
        self.assertEqual(stats.fermi_updates, 9)
        self.assertEqual(fermi.call_count, 9)
        for call in fermi.call_args_list:
            self.assertEqual(call.args, (4.0, 4.0, model.config.k))

    def test_window_one_preserves_original_instantaneous_learning_trajectory(self):
        grid = ("CDC", "CCD", "DDC")
        config = SimulationConfig(size=3, seed=42, update_mode="synchronous",
                                  advice_mode="adaptive", initial_trust=0.7,
                                  trust_learning_rate=0.2, aspiration_learning_rate=0.3)
        self.assertEqual(config.trust_payoff_window, 1)

        def advisor(context):
            return "C" if context.cooperating_neighbors < 2 else "D"

        default = SpatialPrisonersDilemma(config, advisor=advisor, initial_grid=grid)
        explicit = SpatialPrisonersDilemma(replace(config, trust_payoff_window=1),
                                           advisor=advisor, initial_grid=grid)
        reference = [list(row) for row in default.aspiration_grid]
        trust = [list(row) for row in default.trust_grid]
        for _ in range(8):
            self.assertEqual(default.step(), explicit.step())
            self.assertEqual(default.grid, explicit.grid)
            self.assertEqual(default.trust_grid, explicit.trust_grid)
            self.assertEqual(default.aspiration_grid, explicit.aspiration_grid)
            for row in range(3):
                for col in range(3):
                    payoff = default.payoff(row, col)
                    if default.last_advice_grid[row][col]:
                        if payoff > reference[row][col]:
                            trust[row][col] += 0.2 * (1 - trust[row][col])
                        elif payoff < reference[row][col]:
                            trust[row][col] *= 0.8
                    reference[row][col] = 0.7 * reference[row][col] + 0.3 * payoff
                    self.assertEqual(default.trust_payoff_grid[row][col], payoff)
                    self.assertEqual(default.trust_grid[row][col], trust[row][col])
                    self.assertEqual(default.aspiration_grid[row][col], reference[row][col])

    def test_invalid_windows_and_undefined_learning_statistics(self):
        for value in (0, -1, 1.5, "3", True, math.nan, math.inf):
            with self.subTest(window=value), self.assertRaises((ValueError, TypeError)):
                SimulationConfig(trust_payoff_window=value)
        for config in (SimulationConfig(size=3, trust_payoff_window=3),
                       SimulationConfig(size=3, trust_payoff_window=3, advice_mode="adaptive",
                                        update_mode="synchronous", llm_access_fraction=0)):
            model = SpatialPrisonersDilemma(config)
            self.assertIsNone(model.trust_payoff_grid)
            model.step()
            self.assertIsNone(model.trust_statistics()["eligible_feedback_payoff_mean"])
            if config.advice_mode == "fixed":
                self.assertIsNone(model.trust_payoff_grid)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class TrustPayoffWindowIntegrationTests(BatchFixtureTests):
    def invoke(self, output, extra=()):
        arguments = [
            "--single", "--advice-mode", "adaptive", "--workers", "1", "--repeats", "1",
            "--size", "3", "--steps", "5", "--b", "1.5", "--k", "0.1", "--seed", "13",
            "--p", "0.5", "--initial-cooperation", "0.5", "--boundary", "periodic",
            "--update-mode", "synchronous", "--initial-trust", "0.6",
            "--trust-learning-rate", "0.2", "--aspiration-learning-rate", "0.3",
            "--trust-payoff-window", "3", "--paired-fermi-control", "--early-average-steps", "2",
            "--tail-average-steps", "2", "--report-every", "0", "--advisor-batch", str(self.batch),
            "--output-dir", str(output),
        ]
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return runner.main(arguments + list(extra))

    def test_cli_window_and_history_match_actual_included_post_step_rewards(self):
        output = self.root / "single"
        self.assertEqual(self.invoke(output), 0)
        run = next(output.iterdir())
        metadata = json.loads((run / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["config"]["trust_payoff_window"], 3)
        history = read_csv(run / "history.csv")
        self.assertEqual(history[0]["eligible_feedback_payoff_mean"], "")
        recent = []
        for row in history[1:]:
            recent.append(float(row["eligible_mean_payoff"]))
            self.assertAlmostEqual(float(row["eligible_feedback_payoff_mean"]),
                                   math.fsum(recent[-3:]) / len(recent[-3:]))
        self.assertAlmostEqual(metadata["summary"]["final_eligible_feedback_payoff_mean"],
                               float(history[-1]["eligible_feedback_payoff_mean"]))

    def test_cli_invalid_window_fails_before_output_creation(self):
        for value in ("0", "-2", "1.5", "nan"):
            output = self.root / ("invalid_" + value)
            with self.subTest(window=value), self.assertRaises(SystemExit) as raised:
                self.invoke(output, ["--trust-payoff-window", value])
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(output.exists())

    def test_window_three_actual_spawn_matches_serial_with_paired_controls(self):
        batches = []
        for workers in (1, 2):
            output = self.root / f"repeats_{workers}"
            self.assertEqual(self.invoke(output, ["--workers", str(workers), "--repeats", "2"]), 0)
            batch = next(output.glob("repeat_*"))
            batches.append(batch)
            metadata = json.loads((batch / "repeat_config.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["fixed_parameters"]["trust_payoff_window"], 3)
            self.assertEqual(metadata["actual_workers"], workers)
            for row in read_csv(batch / "summary.csv"):
                self.assertEqual(row["trust_payoff_window"], "3")
                self.assertNotEqual(row["final_eligible_feedback_payoff_mean"], "")
        self.assertEqual((batches[0] / "summary.csv").read_bytes(),
                         (batches[1] / "summary.csv").read_bytes())
        for row in read_csv(batches[0] / "summary.csv"):
            for filename in ("history.csv", "final_grid.csv"):
                self.assertEqual((batches[0] / row["output_dir"] / filename).read_bytes(),
                                 (batches[1] / row["output_dir"] / filename).read_bytes())


if __name__ == "__main__":
    unittest.main()
