"""动态信任的收益时序、决策来源、边界值和固定模式回归。"""

from dataclasses import replace
import math
import unittest

from main.SPD.simulation import SimulationConfig, SpatialPrisonersDilemma


class AlwaysZero:
    def random(self):
        return 0.0


class AdaptiveTrustTests(unittest.TestCase):
    def config(self, **changes):
        values = dict(size=3, seed=29, update_mode="synchronous",
                      advice_mode="adaptive", initial_trust=0.5,
                      trust_learning_rate=0.2, aspiration_learning_rate=0.2)
        values.update(changes)
        return SimulationConfig(**values)

    def model(self, grid, advisor, force_adoption=False, **changes):
        model = SpatialPrisonersDilemma(self.config(**changes), advisor=advisor,
                                       initial_grid=grid)
        if force_adoption:
            model._advice_rng = AlwaysZero()
        return model

    def test_initial_reference_is_own_payoff_and_snapshots_are_immutable(self):
        model = self.model(("CDC", "CCD", "DDC"), lambda _: "C",
                           llm_access_fraction=0.5)
        self.assertEqual(model.aspiration_grid,
                         tuple(tuple(model.payoff(r, c) for c in range(3))
                               for r in range(3)))
        for row in range(3):
            for col in range(3):
                self.assertEqual(model.trust_grid[row][col],
                                 0.5 if model.llm_access_grid[row][col] else 0.0)
                self.assertFalse(model.last_advice_grid[row][col])
        for snapshot in (model.trust_grid, model.aspiration_grid, model.last_advice_grid):
            with self.assertRaises(TypeError):
                snapshot[0][0] = 0.0
        self.assertIsNone(model.trust_statistics()["eligible_advice_rate"])

    def test_new_grid_payoff_uses_old_reference_even_with_beta_one(self):
        # 初态全 C；建议让中心变 D，其他人仍 C。角落收益不变，
        # 中心获益，四个正交邻居受损，同策略建议也必须获得反馈。
        model = self.model(("CCC", "CCC", "CCC"),
                           lambda context: "D" if context.position == (1, 1) else "C",
                           force_adoption=True, aspiration_learning_rate=1.0)
        old_trust = model.trust_grid
        old_reference = model.aspiration_grid
        stats = model.step()
        self.assertEqual(stats.advice_used, 9)
        self.assertEqual(stats.strategy_changes, 1)
        self.assertEqual(model.trust_grid[1][1], 0.6)
        self.assertEqual(model.aspiration_grid[1][1], 6.0)
        for row, col in ((0, 1), (1, 0), (1, 2), (2, 1)):
            self.assertEqual(model.trust_grid[row][col], 0.4)
            self.assertEqual(model.aspiration_grid[row][col], 3.0)
        for row, col in ((0, 0), (0, 2), (2, 0), (2, 2)):
            self.assertEqual(model.trust_grid[row][col], 0.5)
            self.assertEqual(model.aspiration_grid[row][col], 4.0)
        self.assertEqual(old_trust, ((0.5,) * 3,) * 3)
        self.assertEqual(old_reference, ((4.0,) * 3,) * 3)
        self.assertTrue(all(all(row) for row in model.last_advice_grid))
        feedback = model.trust_statistics()
        self.assertEqual(feedback["trust_positive_feedback"], 1)
        self.assertEqual(feedback["trust_negative_feedback"], 4)
        self.assertEqual(feedback["trust_zero_feedback"], 4)
        self.assertEqual(feedback["eligible_advice_rate"], 1.0)
        self.assertAlmostEqual(sum(feedback[f"eligible_trust_bin_{i}"]
                                   for i in range(10)), 1.0)

    def test_positive_and_negative_feedback_have_same_single_kernel(self):
        positive = self.model(("DDD",) * 3, lambda _: "C", force_adoption=True)
        negative = self.model(("CCC",) * 3, lambda _: "D", force_adoption=True)
        positive.step()
        negative.step()
        self.assertEqual(positive.trust_grid, ((0.6,) * 3,) * 3)
        self.assertEqual(negative.trust_grid, ((0.4,) * 3,) * 3)
        self.assertEqual(positive.aspiration_grid, ((0.8,) * 3,) * 3)
        self.assertEqual(negative.aspiration_grid, ((3.2,) * 3,) * 3)
        self.assertEqual(positive.trust_statistics()["trust_positive_feedback"], 9)
        self.assertEqual(negative.trust_statistics()["trust_negative_feedback"], 9)

    def test_beta_zero_keeps_reference_but_still_updates_trust(self):
        model = self.model(("CCC",) * 3, lambda _: "D", force_adoption=True,
                           aspiration_learning_rate=0.0)
        model.step()
        self.assertEqual(model.trust_grid, ((0.4,) * 3,) * 3)
        self.assertEqual(model.aspiration_grid, ((4.0,) * 3,) * 3)
        model.step()
        for row in model.trust_grid:
            for value in row:
                self.assertAlmostEqual(value, 0.32)
        self.assertEqual(model.trust_statistics()["trust_negative_feedback"], 9)

    def test_zero_trust_never_requests_but_all_references_continue_updating(self):
        def forbidden(_):
            raise AssertionError("零信任不应请求建议")

        grid = ("CDC", "CCD", "DDC")
        model = self.model(grid, forbidden, initial_trust=0.0,
                           aspiration_learning_rate=1.0)
        old_reference = model.aspiration_grid
        fixed = SpatialPrisonersDilemma(replace(model.config, advice_mode="fixed",
                                                lambda_prob=0.0), initial_grid=grid)
        for _ in range(5):
            self.assertEqual(model.step(), fixed.step())
            self.assertEqual(model.grid, fixed.grid)
            self.assertEqual(model.trust_grid, ((0.0,) * 3,) * 3)
            self.assertEqual(model.aspiration_grid,
                             tuple(tuple(model.payoff(r, c) for c in range(3))
                                   for r in range(3)))
            self.assertEqual(model.trust_statistics()["trust_positive_feedback"], 0)
        self.assertNotEqual(model.aspiration_grid, old_reference)

    def test_no_advice_falls_back_without_trust_feedback(self):
        grid = ("CDC", "CCD", "DDC")
        model = self.model(grid, lambda _: None, initial_trust=1.0,
                           aspiration_learning_rate=1.0)
        fixed = SpatialPrisonersDilemma(replace(model.config, advice_mode="fixed",
                                                lambda_prob=0.0), initial_grid=grid)
        for _ in range(5):
            actual, expected = model.step(), fixed.step()
            self.assertEqual(model.grid, fixed.grid)
            self.assertEqual(actual.advice_requests, 9)
            self.assertEqual(actual.advice_used, 0)
            self.assertEqual(actual.fermi_updates, expected.fermi_updates)
            self.assertEqual(model.trust_grid, ((1.0,) * 3,) * 3)
            self.assertFalse(any(any(row) for row in model.last_advice_grid))
            statistics = model.trust_statistics()
            for field in ("trust_positive_feedback", "trust_negative_feedback",
                          "trust_zero_feedback", "eligible_advice_rate"):
                self.assertEqual(statistics[field], 0)

    def test_alpha_zero_is_exact_fixed_probability_trajectory(self):
        for fraction in (0.0, 0.4, 1.0):
            for seed in (13, 42):
                with self.subTest(fraction=fraction, seed=seed):
                    config = self.config(initial_trust=0.3, lambda_prob=1.0,
                                         trust_learning_rate=0.0, seed=seed,
                                         llm_access_fraction=fraction)
                    def advisor(context):
                        return "D" if context.cooperating_neighbors >= 2 else "C"
                    adaptive = SpatialPrisonersDilemma(config, advisor=advisor)
                    fixed = SpatialPrisonersDilemma(
                        replace(config, advice_mode="fixed", lambda_prob=0.3), advisor=advisor)
                    for _ in range(8):
                        self.assertEqual(adaptive.step(), fixed.step())
                        self.assertEqual(adaptive.grid, fixed.grid)
                        self.assertEqual(adaptive.last_advice_grid, fixed.last_advice_grid)

    def test_adaptive_ignores_fixed_lambda_and_updates_only_eligible_trust(self):
        seen = []
        model = self.model(("CCC",) * 3,
                           lambda context: seen.append(context.position) or "D",
                           lambda_prob=0.0, initial_trust=1.0, llm_access_fraction=0.5)
        model.step()
        self.assertEqual(set(seen), {(r, c) for r in range(3) for c in range(3)
                                     if model.llm_access_grid[r][c]})
        self.assertEqual(len(seen), 4)
        for r in range(3):
            for c in range(3):
                if not model.llm_access_grid[r][c]:
                    self.assertEqual(model.trust_grid[r][c], 0.0)
        statistics = model.trust_statistics()
        eligible = [model.trust_grid[r][c] for r in range(3) for c in range(3)
                    if model.llm_access_grid[r][c]]
        self.assertAlmostEqual(statistics["eligible_trust_mean"], sum(eligible) / 4)
        self.assertEqual(statistics["eligible_advice_rate"], 1.0)

    def test_no_eligible_people_have_no_group_statistics_but_reference_updates(self):
        model = self.model(("CDC", "CCD", "DDC"), lambda _: "D",
                           llm_access_fraction=0.0, aspiration_learning_rate=1.0)
        initial = model.aspiration_grid
        model.step()
        self.assertNotEqual(initial, model.aspiration_grid)
        for key, value in model.trust_statistics().items():
            self.assertEqual(value, 0 if key.startswith("trust_") else None)

    def test_trust_histogram_edges_and_population_standard_deviation(self):
        for initial in (0.0, 0.1, 0.5, 0.9, 1.0):
            with self.subTest(initial=initial):
                model = self.model(("CCC",) * 3, lambda _: "C", initial_trust=initial)
                statistics = model.trust_statistics()
                self.assertEqual(statistics["eligible_trust_std"], 0.0)
                self.assertEqual(statistics["eligible_low_trust_fraction"], float(initial < 0.1))
                self.assertEqual(statistics["eligible_high_trust_fraction"], float(initial > 0.9))
                for i in range(10):
                    self.assertEqual(statistics[f"eligible_trust_bin_{i}"],
                                     float(i == min(int(initial * 10), 9)))
                # 测量和统计不应移动随机流或改变状态。
                before = model._rng.getstate(), model._advice_rng.getstate()
                self.assertEqual(statistics, model.trust_statistics())
                model.measure()
                self.assertEqual(before, (model._rng.getstate(), model._advice_rng.getstate()))

    def test_fixed_mode_has_no_trust_or_reference_values(self):
        model = SpatialPrisonersDilemma(SimulationConfig(size=3, lambda_prob=1.0,
                                                        llm_access_fraction=0.5),
                                       advisor=lambda context: context.strategy)
        self.assertIsNone(model.trust_grid)
        self.assertIsNone(model.aspiration_grid)
        statistics = model.trust_statistics()
        for key, value in statistics.items():
            self.assertEqual(value, 0 if key.startswith("trust_") else None)
        stats = model.step()
        self.assertEqual(model.trust_statistics()["eligible_advice_rate"], 1.0)
        self.assertGreater(stats.advice_used, 0)

    def test_learning_parameter_validation_and_adaptive_synchronous_requirement(self):
        for name in ("initial_trust", "trust_learning_rate", "aspiration_learning_rate"):
            for value in (-0.1, 1.1, math.nan, math.inf, -math.inf):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.config(**{name: value})
        with self.assertRaises(ValueError):
            self.config(advice_mode="invalid")
        for fraction in (0.0, 1.0):
            with self.assertRaises(ValueError):
                self.config(update_mode="asynchronous", llm_access_fraction=fraction)


if __name__ == "__main__":
    unittest.main()
