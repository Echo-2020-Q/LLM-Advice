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
            self.assertEqual(context.grid_rows, ("CCC", "CCC", "CCC"))
            self.assertEqual(context.grid_key, "CCCCCCCCC")
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
        self.assertTrue(all(context.grid_rows == ("DDD", "DDD", "DDD") for context in observations[9:]))
        self.assertTrue(all(context.grid_rows == ("CCC", "CCC", "CCC") for context in observations[:9]))

    def test_asynchronous_selection_has_replacement_and_immediate_changes(self):
        expected_grid = [["C"] * 3 for _ in range(3)]
        visited = []

        def advisor(context):
            row, col = context.position
            visited.append(context.position)
            self.assertEqual(context.strategy, expected_grid[row][col])
            expected_window = tuple(
                "".join(expected_grid[r % 3][c % 3] for c in range(col - 1, col + 2))
                for r in range(row - 1, row + 2)
            )
            self.assertEqual(context.grid_rows, expected_window)
            self.assertEqual(context.grid_key, "".join(expected_window))
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


class LocalGridObservationTests(unittest.TestCase):
    def capture_contexts(self, grid, boundary="periodic"):
        observations = {}

        def advisor(context):
            observations[context.position] = context
            return context.strategy

        model = SpatialPrisonersDilemma(
            SimulationConfig(size=len(grid), lambda_prob=1, update_mode="synchronous", boundary=boundary),
            advisor=advisor, initial_grid=grid,
        )
        model.step()
        return model, observations

    def test_periodic_corner_windows_keep_north_south_and_west_east_orientation(self):
        model, contexts = self.capture_contexts(("CDCD", "DCCD", "CDCC", "DDCC"))
        expected = {
            (0, 0): ("CDD", "DCD", "DDC"),
            (0, 3): ("CCD", "CDC", "CDD"),
            (3, 0): ("CCD", "CDD", "DCD"),
            (3, 3): ("CCC", "CCD", "CDC"),
        }
        for position, rows in expected.items():
            with self.subTest(position=position):
                context = contexts[position]
                self.assertEqual(context.grid_rows, rows)
                self.assertEqual(context.grid_key, "".join(rows))
                self.assertEqual(context.grid_rows[1][1], context.strategy)
                self.assertEqual(len(context.neighbors), 4)
                self.assertEqual(len(model.neighbors(*position)), 4)
                self.assertEqual(
                    tuple(neighbor.strategy for neighbor in context.neighbors),
                    (rows[0][1], rows[2][1], rows[1][0], rows[1][2]),
                )

    def test_same_four_neighbors_and_payoff_can_have_distinct_diagonal_keys(self):
        first_grid = ("CDC", "DCC", "CDC")
        second_grid = ("DDC", "DCC", "CDC")
        first_model, first_contexts = self.capture_contexts(first_grid)
        second_model, second_contexts = self.capture_contexts(second_grid)
        first, second = first_contexts[(1, 1)], second_contexts[(1, 1)]
        self.assertEqual(first.grid_rows, first_grid)
        self.assertEqual(second.grid_rows, second_grid)
        self.assertEqual(first.grid_key, "CDCDCCCDC")
        self.assertEqual(second.grid_key, "DDCDCCCDC")
        self.assertNotEqual(first.grid_key, second.grid_key)
        self.assertEqual(first.strategy, "C")
        self.assertEqual(second.strategy, first.strategy)
        self.assertEqual(first.cooperating_neighbors, 1)
        self.assertEqual(second.cooperating_neighbors, 1)
        self.assertEqual(first.defecting_neighbors, 3)
        self.assertEqual(second.defecting_neighbors, 3)
        self.assertEqual(first.payoff, 1.0)
        self.assertEqual(second.payoff, 1.0)
        self.assertEqual(first_model.payoff(1, 1), second_model.payoff(1, 1))

    def test_open_boundary_omits_incomplete_windows_without_inventing_cells(self):
        _, contexts = self.capture_contexts(("CDCD", "DCCD", "CDCC", "DDCC"), boundary="open")
        for (row, col), context in contexts.items():
            with self.subTest(position=(row, col)):
                if row in (0, 3) or col in (0, 3):
                    self.assertIsNone(context.grid_rows)
                    self.assertIsNone(context.grid_key)
                else:
                    self.assertIsInstance(context.grid_rows, tuple)
                    self.assertEqual(len(context.grid_rows), 3)
                    self.assertTrue(all(len(line) == 3 for line in context.grid_rows))
                    self.assertEqual(context.grid_rows[1][1], context.strategy)
        self.assertEqual(contexts[(1, 1)].grid_rows, ("CDC", "DCC", "CDC"))
        self.assertEqual(contexts[(2, 2)].grid_rows, ("CCD", "DCC", "DCC"))


class FixedAdviceAccessTests(unittest.TestCase):
    def test_access_count_uses_decimal_floor_and_mask_is_immutable(self):
        cases = ((10, 0.29, 29), (3, 0.29, 2), (3, 0.1, 0), (3, 0.999, 8),
                 (3, 0.0, 0), (3, 1.0, 9))
        for size, fraction, expected_count in cases:
            with self.subTest(size=size, fraction=fraction):
                config = SimulationConfig(size=size, llm_access_fraction=fraction)
                self.assertEqual(config.llm_access_count, expected_count)
                model = SpatialPrisonersDilemma(config)
                mask = model.llm_access_grid
                self.assertIsInstance(mask, tuple)
                self.assertTrue(all(isinstance(row, tuple) for row in mask))
                self.assertTrue(all(type(value) is bool for row in mask for value in row))
                self.assertEqual(len(mask), size)
                self.assertTrue(all(len(row) == size for row in mask))
                self.assertEqual(sum(sum(row) for row in mask), expected_count)
                with self.assertRaises(TypeError):
                    mask[0][0] = not mask[0][0]

    def test_zero_access_never_calls_advisor_and_matches_pure_fermi(self):
        def forbidden_advisor(context):
            raise AssertionError("P=0 时不能请求建议，即使 lambda=1")

        for mode in ("synchronous", "asynchronous"):
            with self.subTest(mode=mode):
                common = dict(size=4, seed=23, b=1.1, k=0.5, update_mode=mode)
                baseline = SpatialPrisonersDilemma(SimulationConfig(lambda_prob=0, **common))
                no_access = SpatialPrisonersDilemma(
                    SimulationConfig(lambda_prob=1, llm_access_fraction=0, **common),
                    advisor=forbidden_advisor,
                )
                self.assertEqual(no_access.grid, baseline.grid)
                self.assertEqual(no_access.run(8), baseline.run(8))
                self.assertEqual(no_access.grid, baseline.grid)

    def test_full_access_preserves_pre_access_rule_trajectories(self):
        # 加入资格门控之前保存的轨迹，用于防止新增抽样消耗已有的随机数流。
        expected = {
            "synchronous": (
                [(0.625, 0, 0, 0, 0), (0.4375, 7, 7, 9, 7), (0.4375, 7, 7, 9, 10),
                 (0.3125, 6, 6, 10, 8), (0.25, 8, 8, 8, 9)],
                ["DCDD", "DDDC", "DDDC", "CDDD"],
            ),
            "asynchronous": (
                [(0.625, 0, 0, 0, 0), (0.5, 7, 7, 9, 8), (0.375, 7, 7, 9, 10),
                 (0.1875, 6, 6, 10, 7), (0.375, 8, 8, 8, 9)],
                ["DCCD", "DDDC", "DDDD", "CDCC"],
            ),
        }
        for mode in expected:
            for explicit_fraction in ({}, {"llm_access_fraction": 1.0}):
                with self.subTest(mode=mode, explicit_fraction=explicit_fraction):
                    config = SimulationConfig(
                        size=4, b=1.1, k=0.5, lambda_prob=0.4, seed=23,
                        update_mode=mode, **explicit_fraction,
                    )
                    model = SpatialPrisonersDilemma(
                        config, advisor=lambda context: "D" if context.strategy == "C" else "C",
                    )
                    actual = [
                        (row.cooperation_rate, row.advice_requests, row.advice_used,
                         row.fermi_updates, row.strategy_changes)
                        for row in model.run(4)
                    ]
                    self.assertEqual(actual, expected[mode][0])
                    self.assertEqual(["".join(row) for row in model.grid], expected[mode][1])

    def test_synchronous_partial_access_requests_same_people_each_round(self):
        observed = []

        def advisor(context):
            observed.append(context.position)
            return "D" if context.strategy == "C" else "C"

        model = SpatialPrisonersDilemma(
            SimulationConfig(size=3, lambda_prob=1, llm_access_fraction=0.5,
                             update_mode="synchronous", seed=42),
            advisor=advisor,
        )
        original_mask = model.llm_access_grid
        eligible_positions = {
            (row, col) for row in range(3) for col in range(3) if original_mask[row][col]
        }
        self.assertEqual(len(eligible_positions), 4)
        for _ in range(4):
            observed.clear()
            stats = model.step()
            self.assertEqual(set(observed), eligible_positions)
            self.assertEqual(len(observed), 4)
            self.assertEqual(stats.advice_requests, 4)
            self.assertEqual(stats.advice_used, 4)
            self.assertEqual(stats.fermi_updates, 5)
            self.assertEqual(model.llm_access_grid, original_mask)

    def test_asynchronous_advice_only_reaches_eligible_positions(self):
        observed = []

        def advisor(context):
            observed.append(context.position)
            return "D" if context.strategy == "C" else "C"

        model = SpatialPrisonersDilemma(
            SimulationConfig(size=4, lambda_prob=1, llm_access_fraction=0.4,
                             update_mode="asynchronous", seed=42),
            advisor=advisor,
        )
        original_mask = model.llm_access_grid
        total_requests = 0
        for _ in range(4):
            observed.clear()
            stats = model.step()
            self.assertTrue(all(original_mask[row][col] for row, col in observed))
            self.assertEqual(stats.advice_requests, len(observed))
            self.assertEqual(stats.advice_used, len(observed))
            self.assertEqual(stats.fermi_updates, 16 - len(observed))
            self.assertEqual(model.llm_access_grid, original_mask)
            total_requests += len(observed)
        self.assertGreater(total_requests, 0)

    def test_access_is_reproducible_without_changing_initial_state_or_none_fallback(self):
        for mode in ("synchronous", "asynchronous"):
            common = dict(size=5, seed=23, b=1.1, k=0.5, update_mode=mode)
            baseline = SpatialPrisonersDilemma(SimulationConfig(lambda_prob=0, **common))
            initial = baseline.grid
            reference = baseline.run(6)
            for fraction in (0.0, 0.25, 0.75, 1.0):
                with self.subTest(mode=mode, fraction=fraction):
                    config = SimulationConfig(lambda_prob=0.6, llm_access_fraction=fraction, **common)
                    first = SpatialPrisonersDilemma(config, advisor=lambda context: None)
                    repeat = SpatialPrisonersDilemma(config, advisor=lambda context: None)
                    self.assertEqual(first.llm_access_grid, repeat.llm_access_grid)
                    self.assertEqual(first.grid, initial)
                    self.assertEqual(repeat.grid, initial)
                    history = first.run(6)
                    self.assertEqual(history, repeat.run(6))
                    self.assertEqual(first.grid, baseline.grid)
                    for actual, expected in zip(history, reference):
                        for field in ("round", "cooperation_rate", "average_payoff",
                                      "advice_used", "fermi_updates", "strategy_changes"):
                            self.assertEqual(getattr(actual, field), getattr(expected, field))

    def test_people_without_access_can_imitate_eligible_neighbors(self):
        model = SpatialPrisonersDilemma(
            SimulationConfig(size=3, b=1.1, k=1, lambda_prob=1, initial_cooperation=0,
                             llm_access_fraction=0.12, seed=42, update_mode="synchronous"),
            advisor=lambda context: "C",
        )
        self.assertEqual(model.config.llm_access_count, 1)
        model.step()
        # 初次同步更新的旧网格全为 D，只有有资格者能直接从建议获得 C。
        for row in range(3):
            for col in range(3):
                self.assertEqual(model.grid[row][col], "C" if model.llm_access_grid[row][col] else "D")
        model.run(2)
        self.assertTrue(any(
            model.grid[row][col] == "C" and not model.llm_access_grid[row][col]
            for row in range(3) for col in range(3)
        ))


class ConfigurationAndHistoryTests(unittest.TestCase):
    def test_invalid_configuration_is_rejected(self):
        invalid = {
            "size": (0, 1, 2, 3.5),
            "b": (0, 1.0, math.nan, math.inf, 1e308),
            "k": (0, -0.1, math.nan, math.inf),
            "lambda_prob": (-0.1, 1.1, math.nan, math.inf),
            "initial_cooperation": (-0.1, 1.1, math.nan, math.inf),
            "llm_access_fraction": (-0.1, 1.1, math.nan, math.inf, -math.inf),
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
