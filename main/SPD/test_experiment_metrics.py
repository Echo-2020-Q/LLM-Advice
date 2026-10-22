"""轮末分组福利、时间窗口和全 D/合作恢复统计的独立测试。"""

from dataclasses import replace
import random
import unittest

from main.SPD.experiment_metrics import GROUPS, HISTORY_FIELDS, SUMMARY_FIELDS, PayoffTracker
from main.SPD.simulation import SimulationConfig, SpatialPrisonersDilemma


ALL_C = ("CCC", "CCC", "CCC")
ALL_D = ("DDD", "DDD", "DDD")
CENTER_D = ("CCC", "CDC", "CCC")
CENTER_C = ("DDD", "DCD", "DDD")


class ControlledGrid:
    """人工指定各轮网格，空间收益仍由真实模型的公开接口计算。"""

    llm_access_grid = ((False, False, False), (False, True, False), (False, False, False))

    def set_grid(self, round_number, grid):
        self.round = round_number
        self._current = SpatialPrisonersDilemma(
            SimulationConfig(size=3, b=1.5, lambda_prob=0), initial_grid=grid,
        )
        return replace(self._current.measure(), round=round_number)

    def payoff(self, row, col):
        return self._current.payoff(row, col)


class PayoffTrackerTests(unittest.TestCase):
    def test_exact_group_totals_and_early_tail_means_exclude_initial_state(self):
        model = ControlledGrid()
        initial = model.set_grid(0, ALL_C)
        tracker = PayoffTracker(model, early_window_steps=2, tail_window_steps=2)
        self.assertEqual(GROUPS, ("population", "eligible", "ineligible"))
        first = tracker.observe(initial)
        self.assertEqual(tuple(first), HISTORY_FIELDS)
        for group in GROUPS:
            self.assertEqual(first[group + "_mean_payoff"], 4.0)
            self.assertEqual(first[group + "_cumulative_mean_payoff"], 0.0)
        # 各轮总收益为 34、0、6、36；中心资格者依次获得 6、0、0、4。
        for round_number, grid in enumerate((CENTER_D, ALL_D, CENTER_C, ALL_C), start=1):
            history = tracker.observe(model.set_grid(round_number, grid))
        result = tracker.summary()
        self.assertEqual(tuple(result), SUMMARY_FIELDS)
        expected = {
            "population": (9, 76, 76 / 9, 19 / 9, 17 / 9, 7 / 3),
            "eligible": (1, 10, 10, 2.5, 3, 2),
            "ineligible": (8, 66, 8.25, 2.0625, 1.75, 2.375),
        }
        suffixes = ("count", "cumulative_total_payoff", "cumulative_mean_payoff",
                    "time_mean_payoff", "early_mean_payoff", "tail_mean_payoff")
        for group in GROUPS:
            for suffix, value in zip(suffixes, expected[group]):
                with self.subTest(group=group, suffix=suffix):
                    self.assertAlmostEqual(result[group + "_" + suffix], value)
            self.assertEqual(history[group + "_mean_payoff"], 4)
            self.assertAlmostEqual(history[group + "_cumulative_mean_payoff"], expected[group][2])
        self.assertEqual(result["observed_steps"], 4)
        self.assertEqual(result["early_observed_steps"], 2)
        self.assertEqual(result["tail_observed_steps"], 2)
        self.assertEqual(result["time_mean_cooperation_rate"], 0.5)
        self.assertEqual(result["first_extinction_round"], 2)
        self.assertEqual(result["extinction_events"], 1)
        self.assertEqual(result["cooperation_revivals"], 1)
        self.assertFalse(result["extinct_at_end"])
        self.assertIsNone(result["final_extinction_start_round"])

    def test_short_windows_overlap_and_use_actual_sample_counts(self):
        model = ControlledGrid()
        initial = model.set_grid(0, ALL_C)
        tracker = PayoffTracker(model, early_window_steps=3, tail_window_steps=5)
        tracker.observe(initial)
        tracker.observe(model.set_grid(1, CENTER_D))
        tracker.observe(model.set_grid(2, CENTER_C))
        result = tracker.summary()
        self.assertEqual(result["observed_steps"], 2)
        self.assertEqual(result["early_observed_steps"], 2)
        self.assertEqual(result["tail_observed_steps"], 2)
        for group in GROUPS:
            self.assertAlmostEqual(result[group + "_early_mean_payoff"], result[group + "_time_mean_payoff"])
            self.assertAlmostEqual(result[group + "_tail_mean_payoff"], result[group + "_time_mean_payoff"])
        self.assertAlmostEqual(result["population_time_mean_payoff"], 20 / 9)
        self.assertAlmostEqual(result["time_mean_cooperation_rate"], 0.5)

    def test_empty_groups_at_zero_and_full_coverage_have_no_per_person_means(self):
        for fraction, empty, populated in ((0, "eligible", "ineligible"), (1, "ineligible", "eligible")):
            with self.subTest(fraction=fraction):
                model = SpatialPrisonersDilemma(
                    SimulationConfig(size=3, lambda_prob=0, initial_cooperation=1,
                                     llm_access_fraction=fraction),
                )
                tracker = PayoffTracker(model, 2, 2)
                first = tracker.observe(model.measure())
                self.assertIsNone(first[empty + "_mean_payoff"])
                self.assertIsNone(first[empty + "_cumulative_mean_payoff"])
                history = tracker.observe(model.step())
                summary = tracker.summary()
                self.assertEqual(summary[empty + "_count"], 0)
                self.assertEqual(summary[empty + "_cumulative_total_payoff"], 0)
                for suffix in ("cumulative_mean_payoff", "time_mean_payoff", "early_mean_payoff", "tail_mean_payoff"):
                    self.assertIsNone(summary[empty + "_" + suffix])
                self.assertIsNone(history[empty + "_mean_payoff"])
                self.assertIsNone(history[empty + "_cumulative_mean_payoff"])
                self.assertEqual(summary[populated + "_count"], 9)
                self.assertEqual(summary[populated + "_cumulative_total_payoff"], 36)
                self.assertEqual(summary[populated + "_time_mean_payoff"], 4)

    def test_zero_update_steps_have_zero_cumulative_payoff_and_no_time_means(self):
        model = ControlledGrid()
        initial = model.set_grid(0, ALL_C)
        tracker = PayoffTracker(model, 2, 2)
        tracker.observe(initial)
        summary = tracker.summary()
        for group in GROUPS:
            self.assertEqual(summary[group + "_cumulative_total_payoff"], 0)
            self.assertEqual(summary[group + "_cumulative_mean_payoff"], 0)
            for suffix in ("time_mean_payoff", "early_mean_payoff", "tail_mean_payoff"):
                self.assertIsNone(summary[group + "_" + suffix])
        for name in ("observed_steps", "early_observed_steps", "tail_observed_steps"):
            self.assertEqual(summary[name], 0)
        self.assertIsNone(summary["time_mean_cooperation_rate"])
        self.assertIsNone(summary["first_extinction_round"])
        self.assertFalse(summary["extinction_observed"])
        self.assertFalse(summary["extinct_at_end"])

    def test_initial_all_d_and_revivals_count_distinct_zero_cooperation_episodes(self):
        model = ControlledGrid()
        initial = model.set_grid(0, ALL_D)
        tracker = PayoffTracker(model, 2, 3)
        tracker.observe(initial)
        initial_summary = tracker.summary()
        self.assertEqual(initial_summary["first_extinction_round"], 0)
        self.assertTrue(initial_summary["extinction_observed"])
        self.assertEqual(initial_summary["extinction_events"], 1)
        self.assertEqual(initial_summary["cooperation_revivals"], 0)
        self.assertEqual(initial_summary["final_extinction_start_round"], 0)
        self.assertTrue(initial_summary["extinct_at_end"])
        for round_number, grid in enumerate((ALL_D, ALL_C, ALL_D, ALL_D, ALL_C), start=1):
            tracker.observe(model.set_grid(round_number, grid))
        revived = tracker.summary()
        self.assertEqual(revived["first_extinction_round"], 0)
        self.assertEqual(revived["extinction_events"], 2)
        self.assertEqual(revived["cooperation_revivals"], 2)
        self.assertIsNone(revived["final_extinction_start_round"])
        self.assertFalse(revived["extinct_at_end"])
        tracker.observe(model.set_grid(6, ALL_D))
        summary = tracker.summary()
        self.assertEqual(summary["first_extinction_round"], 0)
        self.assertEqual(summary["extinction_events"], 3)
        self.assertEqual(summary["cooperation_revivals"], 2)
        self.assertEqual(summary["final_extinction_start_round"], 6)
        self.assertTrue(summary["extinct_at_end"])
        self.assertAlmostEqual(summary["time_mean_cooperation_rate"], 1 / 3)

    def test_tracking_does_not_change_model_evolution_or_advisor_randomness(self):
        for mode in ("synchronous", "asynchronous"):
            with self.subTest(mode=mode):
                config = SimulationConfig(size=4, b=1.1, k=0.5, lambda_prob=0.7,
                                          llm_access_fraction=0.5, update_mode=mode, seed=23)
                observed_advice_rng = random.Random(31)
                baseline_advice_rng = random.Random(31)
                observed = SpatialPrisonersDilemma(
                    config, advisor=lambda context: observed_advice_rng.choice(("C", "D")),
                )
                baseline = SpatialPrisonersDilemma(
                    config, advisor=lambda context: baseline_advice_rng.choice(("C", "D")),
                )
                tracker = PayoffTracker(observed, 3, 4)
                tracker.observe(observed.measure())
                mask = observed.llm_access_grid
                for _ in range(8):
                    observed_stats = observed.step()
                    self.assertEqual(observed_stats, baseline.step())
                    history = tracker.observe(observed_stats)
                    self.assertAlmostEqual(history["population_mean_payoff"], observed_stats.average_payoff)
                    snapshot = tracker.summary()
                    self.assertEqual(snapshot, tracker.summary())
                    snapshot["population_count"] = 0
                    self.assertEqual(tracker.summary()["population_count"], 16)
                    self.assertEqual(observed.grid, baseline.grid)
                    self.assertEqual(observed.llm_access_grid, mask)

    def test_invalid_windows_or_nonsequential_observations_are_rejected(self):
        model = ControlledGrid()
        initial = model.set_grid(0, ALL_C)
        for invalid in (0, -1, 1.5, True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    PayoffTracker(model, invalid, 2)
                with self.assertRaises(ValueError):
                    PayoffTracker(model, 2, invalid)
        tracker = PayoffTracker(model, 2, 2)
        with self.assertRaises(ValueError):
            tracker.observe(replace(initial, round=1))
        tracker.observe(initial)
        with self.assertRaises(ValueError):
            tracker.observe(initial)
        with self.assertRaises(ValueError):
            tracker.observe(model.set_grid(2, ALL_D))
        self.assertEqual(tracker.summary()["observed_steps"], 0)
        tracker.observe(model.set_grid(1, ALL_D))
        self.assertEqual(tracker.summary()["observed_steps"], 1)


if __name__ == "__main__":
    unittest.main()
