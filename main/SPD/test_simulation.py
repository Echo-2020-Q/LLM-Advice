"""空间囚徒困境的数学、更新时序和建议接口测试。

从项目根目录运行：
    python -m unittest discover -s main/SPD -p 'test_*.py'
"""

from dataclasses import FrozenInstanceError
import math
import unittest

from main.SPD.llm_advisor import AdviceContext
from main.SPD.simulation import (
    SimulationConfig,
    SpatialPrisonersDilemma,
    fermi_probability,
)


def as_lists(grid):
    return [list(row) for row in grid]


class PayoffAndNeighborhoodTests(unittest.TestCase):
    def setUp(self):
        self.grid = [list("CDC"), list("CCD"), list("DDC")]

    def model(self, **kwargs):
        return SpatialPrisonersDilemma(
            SimulationConfig(size=3, b=1.5, **kwargs),
            initial_grid=self.grid,
        )

    def test_periodic_neighbors_wrap_without_self_interaction(self):
        model = self.model(boundary="periodic")
        self.assertEqual(
            set(model.neighbors(0, 0)), {(2, 0), (1, 0), (0, 2), (0, 1)}
        )
        self.assertEqual(len(model.neighbors(0, 0)), 4)
        self.assertIsInstance(model.neighbors(0, 0), tuple)
        self.assertEqual(model.payoff(0, 0), 2.0)
        self.assertEqual(model.payoff(0, 1), 4.5)
        self.assertEqual(model.payoff(1, 1), 1.0)

    def test_open_boundary_uses_only_existing_neighbors(self):
        model = self.model(boundary="open")
        self.assertEqual(set(model.neighbors(0, 0)), {(1, 0), (0, 1)})
        self.assertEqual(set(model.neighbors(0, 1)), {(0, 0), (0, 2), (1, 1)})
        self.assertEqual(len(model.neighbors(1, 1)), 4)
        self.assertEqual(model.payoff(0, 0), 1.0)
        self.assertEqual(model.payoff(0, 1), 4.5)
        self.assertEqual(model.payoff(1, 1), 1.0)

    def test_payoff_matrix_in_homogeneous_and_single_defector_states(self):
        model = SpatialPrisonersDilemma(
            SimulationConfig(size=3, b=1.7), initial_grid=[["C"] * 3 for _ in range(3)]
        )
        self.assertEqual(model.measure().average_payoff, 4.0)
        one_defector = [["C"] * 3 for _ in range(3)]
        one_defector[1][1] = "D"
        model = SpatialPrisonersDilemma(
            SimulationConfig(size=3, b=1.7), initial_grid=one_defector
        )
        self.assertAlmostEqual(model.payoff(1, 1), 6.8)
        self.assertEqual(model.payoff(0, 1), 3.0)
        self.assertEqual(model.payoff(0, 0), 4.0)
        all_defectors = SpatialPrisonersDilemma(
            SimulationConfig(size=3, initial_cooperation=0.0)
        )
        self.assertEqual(all_defectors.measure().average_payoff, 0.0)

    def test_initial_grid_and_returned_snapshot_do_not_alias_internal_state(self):
        model = self.model()
        self.grid[0][0] = "D"
        self.assertEqual(model.grid[0][0], "C")
        snapshot = model.grid
        if isinstance(snapshot[0], list):
            snapshot[0][0] = "D"
            self.assertEqual(model.grid[0][0], "C")


class FermiRuleTests(unittest.TestCase):
    def test_equal_payoffs_and_direction_of_imitation(self):
        self.assertEqual(fermi_probability(2.0, 2.0, 0.1), 0.5)
        better_neighbor = fermi_probability(0.0, 1.0, 0.1)
        worse_neighbor = fermi_probability(1.0, 0.0, 0.1)
        self.assertGreater(better_neighbor, 0.99)
        self.assertLess(worse_neighbor, 0.01)
        self.assertAlmostEqual(better_neighbor + worse_neighbor, 1.0)
        self.assertAlmostEqual(fermi_probability(0.0, 1.0, 1.0), 1 / (1 + math.exp(-1)))

    def test_extreme_payoff_differences_do_not_overflow(self):
        self.assertEqual(fermi_probability(0.0, 1e308, 0.01), 1.0)
        self.assertEqual(fermi_probability(1e308, 0.0, 0.01), 0.0)


class UpdateAndAdviceTests(unittest.TestCase):
    def test_synchronous_advice_reads_a_single_old_state(self):
        observations = []

        def advisor(context):
            observations.append(context)
            return "D"

        model = SpatialPrisonersDilemma(
            SimulationConfig(size=3, lambda_prob=1.0, update_mode="synchronous"),
            advisor=advisor,
            initial_grid=[["C"] * 3 for _ in range(3)],
        )
        old_snapshot = model.grid
        first = model.step()
        self.assertEqual(len(observations), 9)
        self.assertEqual(len({item.position for item in observations}), 9)
        for context in observations:
            self.assertIsInstance(context, AdviceContext)
            self.assertEqual(context.strategy, "C")
            self.assertEqual(context.payoff, 4.0)
            self.assertEqual(context.cooperating_neighbors, 4)
            self.assertEqual(context.defecting_neighbors, 0)
            self.assertIsInstance(context.neighbors, tuple)
            self.assertEqual(context.b, 1.5)
            self.assertEqual(context.k, 0.1)
            for neighbor in context.neighbors:
                self.assertEqual(neighbor.strategy, "C")
                self.assertEqual(neighbor.payoff, 4.0)
        self.assertEqual(as_lists(old_snapshot), [["C"] * 3 for _ in range(3)])
        self.assertEqual(as_lists(model.grid), [["D"] * 3 for _ in range(3)])
        self.assertEqual(first.cooperation_rate, 0.0)
        self.assertEqual(first.average_payoff, 0.0)
        self.assertEqual(first.advice_requests, 9)
        self.assertEqual(first.advice_used, 9)
        self.assertEqual(first.fermi_updates, 0)
        self.assertEqual(first.strategy_changes, 9)
        second = model.step()
        self.assertEqual(second.advice_requests, 9)
        self.assertEqual(second.advice_used, 9)
        self.assertEqual(second.strategy_changes, 0)

    def test_asynchronous_selection_has_replacement_and_immediate_changes(self):
        expected_grid = [["C"] * 3 for _ in range(3)]
        visited = []

        def advisor(context):
            row, col = context.position
            visited.append(context.position)
            self.assertEqual(context.strategy, expected_grid[row][col])
            expected_cooperators = sum(
                expected_grid[n.position[0]][n.position[1]] == "C"
                for n in context.neighbors
            )
            self.assertEqual(context.cooperating_neighbors, expected_cooperators)
            self.assertEqual(context.defecting_neighbors, 4 - expected_cooperators)
            multiplier = 1.0 if context.strategy == "C" else 1.5
            self.assertEqual(context.payoff, expected_cooperators * multiplier)
            for neighbor in context.neighbors:
                self.assertEqual(
                    neighbor.strategy, expected_grid[neighbor.position[0]][neighbor.position[1]]
                )
            next_strategy = "D" if context.strategy == "C" else "C"
            expected_grid[row][col] = next_strategy
            return next_strategy

        model = SpatialPrisonersDilemma(
            SimulationConfig(
                size=3, b=1.5, lambda_prob=1.0, seed=42, update_mode="asynchronous"
            ),
            advisor=advisor,
            initial_grid=[["C"] * 3 for _ in range(3)],
        )
        result = model.step()
        self.assertEqual(len(visited), 9)
        self.assertLess(len(set(visited)), 9)
        self.assertEqual(as_lists(model.grid), expected_grid)
        self.assertEqual(result.strategy_changes, 9)
        self.assertEqual(result.advice_used, 9)
        self.assertEqual(result.fermi_updates, 0)

    def test_lambda_zero_never_calls_advisor(self):
        def forbidden_advisor(context):
            raise AssertionError("lambda=0 时不应请求建议")

        model = SpatialPrisonersDilemma(
            SimulationConfig(size=3, lambda_prob=0.0), advisor=forbidden_advisor
        )
        stats = model.step()
        self.assertEqual(stats.advice_requests, 0)
        self.assertEqual(stats.advice_used, 0)
        self.assertEqual(stats.fermi_updates, 9)

    def test_missing_advice_falls_back_and_preserves_fermi_trajectory(self):
        initial = [list("CDCD"), list("DCDC"), list("CDCD"), list("DCDC")]
        for update_mode in ("synchronous", "asynchronous"):
            for lambda_prob in (0.4, 1.0):
                with self.subTest(update_mode=update_mode, lambda_prob=lambda_prob):
                    common = dict(size=4, b=1.5, k=0.5, seed=23, update_mode=update_mode)
                    baseline = SpatialPrisonersDilemma(
                        SimulationConfig(lambda_prob=0.0, **common), initial_grid=initial
                    )
                    fallback = SpatialPrisonersDilemma(
                        SimulationConfig(lambda_prob=lambda_prob, **common),
                        advisor=lambda context: None,
                        initial_grid=initial,
                    )
                    total_changes = 0
                    for _ in range(10):
                        base_stats, fallback_stats = baseline.step(), fallback.step()
                        self.assertEqual(baseline.grid, fallback.grid)
                        for field in (
                            "round", "cooperation_rate", "average_payoff",
                            "fermi_updates", "strategy_changes",
                        ):
                            self.assertEqual(getattr(base_stats, field), getattr(fallback_stats, field))
                        self.assertEqual(fallback_stats.advice_used, 0)
                        self.assertEqual(fallback_stats.fermi_updates, 16)
                        if lambda_prob == 1.0:
                            self.assertEqual(fallback_stats.advice_requests, 16)
                        total_changes += fallback_stats.strategy_changes
                    self.assertGreater(total_changes, 0)

    def test_invalid_advice_and_advisor_exception_are_visible(self):
        model = SpatialPrisonersDilemma(
            SimulationConfig(size=3, lambda_prob=1.0), advisor=lambda context: "cooperate"
        )
        with self.assertRaises(ValueError):
            model.step()
        expected_error = RuntimeError("模拟建议服务故障")

        def failing_advisor(context):
            raise expected_error

        model = SpatialPrisonersDilemma(
            SimulationConfig(size=3, lambda_prob=1.0), advisor=failing_advisor
        )
        with self.assertRaises(RuntimeError) as raised:
            model.step()
        self.assertIs(raised.exception, expected_error)


class ConfigurationAndHistoryTests(unittest.TestCase):
    def test_invalid_configuration_is_rejected(self):
        invalid = {
            "size": (0, 1, 2, 3.5),
            "b": (0, 1.0, math.nan, math.inf, 1e308),
            "k": (0, -0.1, math.nan, math.inf),
            "lambda_prob": (-0.1, 1.1, math.nan, math.inf),
            "initial_cooperation": (-0.1, 1.1, math.nan, math.inf),
            "boundary": ("closed",),
            "update_mode": ("random",),
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaises((ValueError, TypeError)):
                        SimulationConfig(**{field: value})

    def test_invalid_initial_grid_is_rejected(self):
        invalid_grids = (
            [["C"] * 3 for _ in range(2)],
            [["C"] * 2 for _ in range(3)],
            [list("CCC"), list("CXC"), list("CCC")],
        )
        for grid in invalid_grids:
            with self.subTest(grid=grid):
                with self.assertRaises((ValueError, TypeError)):
                    SpatialPrisonersDilemma(SimulationConfig(size=3), initial_grid=grid)

    def test_seed_reproduces_initialization_and_evolution(self):
        for mode in ("synchronous", "asynchronous"):
            with self.subTest(mode=mode):
                config = SimulationConfig(size=5, seed=73, update_mode=mode)
                first = SpatialPrisonersDilemma(config)
                second = SpatialPrisonersDilemma(config)
                self.assertEqual(first.grid, second.grid)
                self.assertEqual(first.run(5), second.run(5))
                self.assertEqual(first.grid, second.grid)

    def test_history_includes_current_state_and_rounds_continue(self):
        model = SpatialPrisonersDilemma(
            SimulationConfig(size=3, initial_cooperation=1.0, lambda_prob=0.0)
        )
        initial = model.measure()
        self.assertEqual(initial.round, 0)
        self.assertEqual(initial.cooperation_rate, 1.0)
        self.assertEqual(initial.average_payoff, 4.0)
        for field in ("advice_requests", "advice_used", "fermi_updates", "strategy_changes"):
            self.assertEqual(getattr(initial, field), 0)
        self.assertEqual(model.run(0), [initial])
        self.assertEqual([stats.round for stats in model.run(2)], [0, 1, 2])
        self.assertEqual([stats.round for stats in model.run(1)], [2, 3])
        self.assertEqual(model.measure().cooperation_rate, 1.0)
        with self.assertRaises(FrozenInstanceError):
            initial.round = 1
        with self.assertRaises(FrozenInstanceError):
            model.config.b = 2.0


if __name__ == "__main__":
    unittest.main()
