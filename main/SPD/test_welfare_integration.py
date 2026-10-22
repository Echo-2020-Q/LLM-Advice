"""配对实验的收益统计集成测试；独立按策略网格计算收益，不复用统计实现。"""

import csv
from dataclasses import asdict, replace
import io
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from main.SPD import run_experiment as runner
from main.SPD.simulation import SimulationConfig, SpatialPrisonersDilemma


GROUPS = ("population", "eligible", "ineligible")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class AlternatingAdvisor:
    """奇数轮建议 C，偶数轮建议 D；用于构造可手算的演化，不读取外部数据。"""

    def __init__(self):
        self.calls = 0

    def __call__(self, context):
        self.calls += 1
        return "C" if context.round % 2 else "D"

    def metadata(self):
        return {"source": "synthetic_alternating_advisor"}


def manual_payoffs(model):
    grid = model.grid
    size = model.config.size
    result = {}
    for r in range(size):
        for c in range(size):
            neighbors = ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            if model.config.boundary == "periodic":
                neighbors = [(nr % size, nc % size) for nr, nc in neighbors]
            else:
                neighbors = [(nr, nc) for nr, nc in neighbors if 0 <= nr < size and 0 <= nc < size]
            cooperating = sum(grid[nr][nc] == "C" for nr, nc in neighbors)
            result[r, c] = cooperating * (1 if grid[r][c] == "C" else model.config.b)
    return result


class WelfareIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="spd_welfare_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def run_case(self, name, config, steps, paired=True, early=2, tail=2):
        advisor = AlternatingAdvisor()
        output = self.root / name
        with redirect_stdout(io.StringIO()):
            summary = runner.run_single(
                config, steps, output, report_every=0, tail_average_steps=tail,
                advisor=advisor, save_details=False, paired_control=paired,
                early_average_steps=early,
            )
        self.assertEqual({path.name for path in output.iterdir()}, {"history.csv", "final_grid.csv"})
        return summary, read_csv(output / "history.csv"), advisor

    def assert_csv_number(self, actual, expected):
        if expected is None:
            self.assertEqual(actual, "")
        else:
            self.assertAlmostEqual(float(actual), expected)

    def test_both_update_modes_match_manual_payoffs_for_same_fixed_groups(self):
        for mode in ("synchronous", "asynchronous"):
            with self.subTest(mode=mode):
                config = SimulationConfig(size=4, b=1.7, k=0.3, lambda_prob=0.8,
                                          llm_access_fraction=0.5, seed=23, update_mode=mode)
                summary, history, advisor = self.run_case(mode, config, steps=5)
                treatment = SpatialPrisonersDilemma(config, advisor=AlternatingAdvisor())
                control = SpatialPrisonersDilemma(replace(config, lambda_prob=0))
                self.assertEqual(treatment.grid, control.grid)
                self.assertEqual(treatment.llm_access_grid, control.llm_access_grid)
                all_positions = [(r, c) for r in range(4) for c in range(4)]
                positions = {
                    "population": all_positions,
                    "eligible": [(r, c) for r, c in all_positions if treatment.llm_access_grid[r][c]],
                    "ineligible": [(r, c) for r, c in all_positions if not treatment.llm_access_grid[r][c]],
                }
                means = {prefix + group: [] for prefix in ("", "control_") for group in GROUPS}
                cooperations = {"": [], "control_": []}
                for index, row in enumerate(history):
                    observed_stats = {}
                    for prefix, model in (("", treatment), ("control_", control)):
                        stats = model.measure() if index == 0 else model.step()
                        observed_stats[prefix] = stats
                        payoffs = manual_payoffs(model)
                        if index:
                            cooperations[prefix].append(stats.cooperation_rate)
                        if not prefix:
                            for name, value in asdict(stats).items():
                                self.assert_csv_number(row[name], value)
                        for group in GROUPS:
                            key = prefix + group
                            mean = sum(payoffs[position] for position in positions[group]) / len(positions[group])
                            if index:
                                means[key].append(mean)
                            self.assert_csv_number(row[key + "_mean_payoff"], mean)
                            self.assert_csv_number(row[key + "_cumulative_mean_payoff"], sum(means[key]))
                    for group in GROUPS:
                        for suffix in ("_mean_payoff", "_cumulative_mean_payoff"):
                            self.assert_csv_number(row["delta_" + group + suffix],
                                                   float(row[group + suffix]) - float(row["control_" + group + suffix]))
                    for name in ("cooperation_rate", "average_payoff"):
                        self.assert_csv_number(row["control_" + name], getattr(observed_stats["control_"], name))
                        self.assert_csv_number(row["delta_" + name],
                                               getattr(observed_stats[""], name) - getattr(observed_stats["control_"], name))
                self.assertEqual(advisor.calls, sum(int(row["advice_requests"]) for row in history))
                for prefix in ("", "control_"):
                    self.assertEqual(summary[prefix + "observed_steps"], 5)
                    self.assertEqual(summary[prefix + "early_observed_steps"], 2)
                    self.assertEqual(summary[prefix + "tail_observed_steps"], 2)
                    self.assertAlmostEqual(summary[prefix + "time_mean_cooperation_rate"],
                                           sum(cooperations[prefix]) / 5)
                    for group in GROUPS:
                        key = prefix + group
                        samples = means[key]
                        count = len(positions[group])
                        expected = {
                            "_count": count,
                            "_cumulative_total_payoff": sum(samples) * count,
                            "_cumulative_mean_payoff": sum(samples),
                            "_time_mean_payoff": sum(samples) / 5,
                            "_early_mean_payoff": sum(samples[:2]) / 2,
                            "_tail_mean_payoff": sum(samples[-2:]) / 2,
                        }
                        for suffix, value in expected.items():
                            self.assertAlmostEqual(summary[key + suffix], value)
                for group in GROUPS:
                    for suffix in ("_cumulative_mean_payoff", "_time_mean_payoff",
                                   "_early_mean_payoff", "_tail_mean_payoff"):
                        self.assertAlmostEqual(summary["delta_" + group + suffix],
                                               summary[group + suffix] - summary["control_" + group + suffix])

    def test_lambda_zero_gives_zero_paired_differences_and_never_requests_advice(self):
        config = SimulationConfig(size=4, lambda_prob=0, llm_access_fraction=0.5, seed=19)
        summary, history, advisor = self.run_case("zero_lambda", config, steps=5)
        self.assertEqual(advisor.calls, 0)
        self.assertTrue(any(name.startswith("delta_") for name in summary))
        for name, value in summary.items():
            if name.startswith("delta_") and value is not None:
                self.assertEqual(value, 0)
        for row in history:
            for name, value in row.items():
                if name.startswith("delta_") and value:
                    self.assertEqual(float(value), 0)

    def test_empty_groups_and_zero_steps_use_defined_missing_values(self):
        for fraction in (0.0, 1.0):
            for steps in (0, 3):
                with self.subTest(fraction=fraction, steps=steps):
                    config = SimulationConfig(size=3, lambda_prob=1, llm_access_fraction=fraction,
                                              seed=7, update_mode="synchronous")
                    summary, history, _ = self.run_case(f"empty_{fraction}_{steps}", config, steps)
                    empty_group = "eligible" if fraction == 0 else "ineligible"
                    for prefix in ("", "control_"):
                        self.assertEqual(summary[prefix + empty_group + "_count"], 0)
                        self.assertEqual(summary[prefix + empty_group + "_cumulative_total_payoff"], 0)
                        self.assertEqual(summary[prefix + "observed_steps"], steps)
                        for suffix in ("_cumulative_mean_payoff", "_time_mean_payoff",
                                       "_early_mean_payoff", "_tail_mean_payoff"):
                            self.assertIsNone(summary[prefix + empty_group + suffix])
                    for row in history:
                        for prefix in ("", "control_", "delta_"):
                            for suffix in ("_mean_payoff", "_cumulative_mean_payoff"):
                                self.assertEqual(row[prefix + empty_group + suffix], "")
                    if steps == 0:
                        self.assertEqual(len(history), 1)
                        for prefix in ("", "control_"):
                            self.assertEqual(summary[prefix + "population_cumulative_mean_payoff"], 0)
                            self.assertEqual(summary[prefix + "early_observed_steps"], 0)
                            self.assertEqual(summary[prefix + "tail_observed_steps"], 0)
                            self.assertIsNone(summary[prefix + "time_mean_cooperation_rate"])
                            for suffix in ("_time_mean_payoff", "_early_mean_payoff", "_tail_mean_payoff"):
                                self.assertIsNone(summary[prefix + "population" + suffix])

    def test_initial_extinction_revivals_and_final_extinction_are_distinguished(self):
        config = SimulationConfig(size=3, lambda_prob=1, llm_access_fraction=1,
                                  initial_cooperation=0, update_mode="synchronous")
        summary, history, _ = self.run_case("revivals", config, steps=4)
        self.assertEqual([float(row["cooperation_rate"]) for row in history], [0, 1, 0, 1, 0])
        expected = {
            "first_extinction_round": 0, "extinction_observed": True,
            "extinction_events": 3, "cooperation_revivals": 2,
            "final_extinction_start_round": 4, "extinct_at_end": True,
            "control_first_extinction_round": 0, "control_extinction_observed": True,
            "control_extinction_events": 1, "control_cooperation_revivals": 0,
            "control_final_extinction_start_round": 0, "control_extinct_at_end": True,
            "population_cumulative_mean_payoff": 8, "population_time_mean_payoff": 2,
            "population_early_mean_payoff": 2, "population_tail_mean_payoff": 2,
            "control_population_cumulative_mean_payoff": 0,
            "delta_population_cumulative_mean_payoff": 8,
            "time_mean_cooperation_rate": 0.5,
        }
        for name, value in expected.items():
            self.assertEqual(summary[name], value)

    def test_switching_control_off_preserves_treatment_and_leaves_control_fields_empty(self):
        config = SimulationConfig(size=4, lambda_prob=0.8, llm_access_fraction=0.5,
                                  seed=21, update_mode="asynchronous")
        paired, paired_history, paired_advisor = self.run_case("paired", config, steps=6)
        unpaired, unpaired_history, unpaired_advisor = self.run_case("unpaired", config, steps=6, paired=False)
        treatment_columns = [name for name in paired_history[0]
                             if not name.startswith(("control_", "delta_"))]
        self.assertEqual([{name: row[name] for name in treatment_columns} for row in paired_history],
                         [{name: row[name] for name in treatment_columns} for row in unpaired_history])
        self.assertEqual((self.root / "paired" / "final_grid.csv").read_bytes(),
                         (self.root / "unpaired" / "final_grid.csv").read_bytes())
        self.assertEqual(paired_advisor.calls, unpaired_advisor.calls)
        for row in unpaired_history:
            for name, value in row.items():
                if name.startswith(("control_", "delta_")):
                    self.assertEqual(value, "")
        for name, value in unpaired.items():
            if name.startswith(("control_", "delta_")):
                self.assertIsNone(value)
            elif name.startswith(GROUPS):
                self.assertEqual(value, paired[name])

    def test_cli_forwards_control_and_early_window_and_preflights_invalid_window(self):
        output = self.root / "scan"
        with patch.object(runner, "create_advisor", side_effect=lambda config, batch: AlternatingAdvisor()), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(runner.main([
                "--scan", "--b-values", "1.6", "--k-values", "0.1", "--lambda-values", "0", "1",
                "--p-values", "0.5", "--seed-values", "42", "--repeats", "1",
                "--size", "3", "--steps", "4", "--report-every", "0", "--no-scan-save-details",
                "--paired-fermi-control", "--early-average-steps", "3", "--output-dir", str(output),
            ]), 0)
        batch = next(output.glob("scan_*"))
        rows = read_csv(batch / "summary.csv")
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(int(row["early_observed_steps"]), 3)
            self.assertEqual(int(row["control_early_observed_steps"]), 3)
            self.assertNotEqual(row["control_population_time_mean_payoff"], "")
            if float(row["lambda_prob"]) == 0:
                self.assertEqual(float(row["delta_population_cumulative_mean_payoff"]), 0)
            self.assertEqual({path.name for path in (batch / row["output_dir"]).iterdir()},
                             {"history.csv", "final_grid.csv"})
        invalid = self.root / "invalid"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                runner.main(["--single", "--early-average-steps", "0", "--output-dir", str(invalid)])
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(invalid.exists())


if __name__ == "__main__":
    unittest.main()
