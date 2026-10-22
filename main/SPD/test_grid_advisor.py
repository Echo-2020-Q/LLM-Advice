"""3×3 完整局面经验表测试；批次为本地合成数据，禁止联网。"""

from contextlib import redirect_stderr, redirect_stdout
import csv
from dataclasses import replace
import io
from itertools import product
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from main.SPD import run_experiment as runner
from main.SPD.llm_advisor import AdviceContext, EmpiricalAdvisor, NeighborObservation
from main.SPD.simulation import SimulationConfig, SpatialPrisonersDilemma


def grid_context(rows):
    key = "".join(rows)
    return AdviceContext(
        round=1, position=(1, 1), strategy=key[4], payoff=0, b=1.8, k=0.1,
        neighbors=tuple(
            NeighborObservation(position, key[index], 0)
            for position, index in (((0, 1), 1), ((2, 1), 7), ((1, 0), 3), ((1, 2), 5))
        ),
        grid_rows=tuple(rows),
    )


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class GridAdvisorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="spd_grid_advisor_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.batch = self.root / "batch"
        self.batch.mkdir()
        guard = patch.object(socket, "socket", side_effect=AssertionError("测试禁止联网"))
        guard.start()
        self.addCleanup(guard.stop)
        cases = []
        for symbols in product("CD", repeat=9):
            key = "".join(symbols)
            count = sum(key[index] == "C" for index in (1, 3, 5, 7))
            cases.append({
                "grid_key": key, "grid_rows": [key[:3], key[3:6], key[6:]],
                "current_action": key[4], "cooperating_neighbors": count,
                "defecting_neighbors": 4 - count,
            })
        self.config = {
            "batch_id": "synthetic_grid_batch", "model": "synthetic",
            "observation_format": "grid_3x3", "b": None, "n": 2,
            "planned_requests": 1024, "cases": cases,
        }
        self.rows = []
        for repetition in range(2):
            for case_index, case in enumerate(cases, start=1):
                key = case["grid_key"]
                # 全 C 局面保留各一条 C/D，用于验证有放回抽样。
                # 其他局面按左上角与正上方共同决定，可区分角点和方向。
                action = ("C" if repetition == 0 else "D") if key == "C" * 9 else (
                    "C" if key[0] == key[1] == "C" else "D"
                )
                self.rows.append({
                    "request_index": len(self.rows) + 1, "repetition": repetition + 1,
                    "combination_index": case_index, "observation_format": "grid_3x3",
                    "b": "unknown", **case, "grid_rows": "/".join(case["grid_rows"]),
                    "status": "success", "recommendation": action, "output_text": action,
                })
        self.write_batch()

    def write_batch(self):
        (self.batch / "config.json").write_text(json.dumps(self.config), encoding="utf-8")
        with (self.batch / "results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def invoke(self, arguments):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return runner.main(arguments)

    def test_complete_table_preserves_512_directed_grids_and_metadata(self):
        advisor = EmpiricalAdvisor(self.batch, seed=42)
        metadata = advisor.metadata()
        self.assertEqual(metadata["observation_format"], "grid_3x3")
        self.assertEqual(metadata["case_count"], 512)
        self.assertEqual(metadata["matching_fields"], ["grid_key"])
        self.assertIsNone(metadata["b"])
        self.assertEqual(metadata["sampling"], "uniform_with_replacement")
        table = advisor.table_rows()
        self.assertEqual(len(table), 512)
        self.assertEqual({row["grid_key"] for row in table}, {case["grid_key"] for case in self.config["cases"]})
        for row in table:
            key = row["grid_key"]
            self.assertEqual(row["grid_rows"], "/".join((key[:3], key[3:6], key[6:])))
            self.assertEqual(row["current_action"], key[4])
            self.assertEqual(row["cooperating_neighbors"], sum(key[index] == "C" for index in (1, 3, 5, 7)))
            self.assertEqual(row["cooperating_neighbors"] + row["defecting_neighbors"], 4)
            self.assertEqual(row["samples"], 2)
            self.assertEqual(row["C"] + row["D"], 2)
        balanced = next(row for row in table if row["grid_key"] == "C" * 9)
        self.assertEqual((balanced["C"], balanced["D"], balanced["p_C"]), (1, 1, 0.5))

    def test_same_neighbor_counts_do_not_merge_corners_directions_or_rotations(self):
        advisor = EmpiricalAdvisor(self.batch, seed=42)
        layouts = (
            ("CCD", "DCD", "DDD"),  # 左上 C、上 C。
            ("DCD", "DCD", "DDD"),  # 只改变左上角。
            ("CDD", "DCC", "DDD"),  # 同角点与邻居计数，合作邻居由上移至右。
            ("DDC", "DCC", "DDD"),  # 第一局面顺时针旋转 90 度。
        )
        contexts = [grid_context(rows) for rows in layouts]
        self.assertEqual({context.strategy for context in contexts}, {"C"})
        self.assertEqual({context.cooperating_neighbors for context in contexts}, {1})
        self.assertEqual([context.grid_key for context in contexts], ["".join(rows) for rows in layouts])
        self.assertEqual([advisor(context) for context in contexts], ["C", "D", "D", "D"])
        usage = {row["grid_key"]: row for row in advisor.usage_rows()}
        self.assertTrue(all(usage[context.grid_key]["draws"] == 1 for context in contexts))
        self.assertEqual(sum(row["draws"] for row in usage.values()), 4)

    def test_sampling_is_reproducible_with_replacement_and_independent_of_csv_row_order(self):
        first = EmpiricalAdvisor(self.batch, seed=11)
        self.rows.reverse()
        self.write_batch()
        second = EmpiricalAdvisor(self.batch, seed=11)
        other_seed = EmpiricalAdvisor(self.batch, seed=12)
        context = grid_context(("CCC", "CCC", "CCC"))
        sequence = [first(context) for _ in range(200)]
        self.assertEqual(sequence, [second(context) for _ in range(200)])
        self.assertNotEqual(sequence, [other_seed(context) for _ in range(200)])
        self.assertEqual(set(sequence), {"C", "D"})
        self.assertTrue(70 <= sequence.count("C") <= 130)
        used = next(row for row in first.usage_rows() if row["grid_key"] == "C" * 9)
        self.assertEqual(used["draws"], 200)
        self.assertEqual(used["selected_C"] + used["selected_D"], 200)
        self.assertEqual(used["samples"], 2)

    def test_invalid_rows_are_rejected_before_becoming_advice(self):
        original = dict(self.rows[0])
        invalid = (
            {"observation_format": "neighbor_counts"}, {"observation_format": "bad_format"},
            {"grid_key": "C" * 8}, {"grid_key": "X" + "C" * 8},
            {"grid_rows": "CC/CCC/CCC"}, {"grid_rows": "DDD/DDD/DDD"},
            {"current_action": "D"}, {"cooperating_neighbors": 3, "defecting_neighbors": 1},
            {"status": "error"}, {"recommendation": "cooperate"}, {"output_text": "D"},
            {"request_index": 2}, {"request_index": 1025},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                self.rows[0] = {**original, **changes}
                self.write_batch()
                with self.assertRaises(ValueError):
                    EmpiricalAdvisor(self.batch)

    def test_invalid_config_formats_cases_and_incomplete_samples_are_rejected(self):
        original_config = json.loads(json.dumps(self.config))
        invalid_configs = []
        for changes in ({"observation_format": "bad_format"}, {"observation_format": "neighbor_counts"},
                        {"planned_requests": 1022}, {"cases": self.config["cases"][:-1]}):
            invalid_configs.append({**original_config, **changes})
        duplicate = json.loads(json.dumps(original_config))
        duplicate["cases"][-1] = duplicate["cases"][0]
        invalid_configs.append(duplicate)
        bad_grid = json.loads(json.dumps(original_config))
        bad_grid["cases"][0]["grid_rows"] = ["DDD"] * 3
        invalid_configs.append(bad_grid)
        for index, config in enumerate(invalid_configs):
            with self.subTest(config=index):
                self.config = config
                self.write_batch()
                with self.assertRaises(ValueError):
                    EmpiricalAdvisor(self.batch)
        self.config = original_config
        original_rows = self.rows
        for rows in (original_rows[:-1], [row for row in original_rows if row["grid_key"] != "C" * 9]):
            with self.subTest(missing_samples=len(original_rows) - len(rows)):
                self.rows = rows
                self.write_batch()
                with self.assertRaises(ValueError):
                    EmpiricalAdvisor(self.batch)

    def test_grid_advisor_rejects_missing_or_inconsistent_context_without_fallback(self):
        advisor = EmpiricalAdvisor(self.batch)
        context = grid_context(("CCC", "CCC", "CCC"))
        inconsistent_neighbors = (replace(context.neighbors[0], strategy="D"),) + context.neighbors[1:]
        contexts = (
            replace(context, grid_rows=None),
            replace(context, grid_rows=("CC", "CCC", "CCC")),
            replace(context, grid_rows=("CCC", "CDC", "CCC")),
            replace(context, neighbors=inconsistent_neighbors),
        )
        for invalid in contexts:
            with self.subTest(context=invalid):
                with self.assertRaises(ValueError):
                    advisor(invalid)
        self.assertEqual(sum(row["draws"] for row in advisor.usage_rows()), 0)
        with self.assertRaises(ValueError):
            advisor.validate_config(1.8, "open")

    def test_simulation_exposes_full_grid_but_payoff_still_uses_four_orthogonal_neighbors(self):
        for rows, expected_payoff in (
            (("CCD", "DCD", "DDD"), 1.0),
            (("DCC", "DCD", "CDC"), 1.0),  # 四个角点改变，正交邻居完全相同。
            (("CCD", "DDD", "DDD"), 1.8),
        ):
            with self.subTest(rows=rows):
                advisor = EmpiricalAdvisor(self.batch)
                contexts = []

                def record_context(context):
                    contexts.append(context)
                    return advisor(context)

                model = SpatialPrisonersDilemma(
                    SimulationConfig(size=3, b=1.8, lambda_prob=1, update_mode="synchronous"),
                    advisor=record_context, initial_grid=[list(row) for row in rows],
                )
                self.assertEqual(model.payoff(1, 1), expected_payoff)
                result = model.step()
                center = next(context for context in contexts if context.position == (1, 1))
                self.assertEqual(center.grid_rows, rows)
                self.assertEqual(center.grid_key, "".join(rows))
                self.assertEqual(center.payoff, expected_payoff)
                self.assertEqual(len(center.neighbors), 4)
                self.assertEqual(result.advice_used, 9)

    def test_paired_scan_compact_and_full_outputs_use_grid_advice_and_match(self):
        batches = []
        for details in (False, True):
            output = self.root / ("full" if details else "compact")
            self.assertEqual(self.invoke([
                "--scan", "--size", "3", "--steps", "2", "--report-every", "0", "--repeats", "1",
                "--b-values", "1.8", "--k-values", "0.1", "--lambda-values", "0", "1",
                "--p-values", "0.5", "--seed-values", "42", "--boundary", "periodic",
                "--update-mode", "synchronous", "--paired-fermi-control", "--early-average-steps", "2",
                "--tail-average-steps", "2", "--advisor-batch", str(self.batch), "--output-dir", str(output),
                "--scan-save-details" if details else "--no-scan-save-details",
            ]), 0)
            batches.append(next(output.glob("scan_*")))
        compact, full = batches
        compact_rows = read_csv(compact / "summary.csv")
        full_rows = read_csv(full / "summary.csv")
        self.assertEqual(compact_rows, full_rows)
        self.assertEqual(len(compact_rows), 2)
        sources = read_json(compact / "scan_config.json")["advisor_sources"]
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["observation_format"], "grid_3x3")
        for row in compact_rows:
            compact_output = compact / row["output_dir"]
            full_output = full / row["output_dir"]
            self.assertEqual({path.name for path in compact_output.iterdir()}, {"history.csv", "final_grid.csv"})
            for filename in ("history.csv", "final_grid.csv"):
                self.assertEqual((compact_output / filename).read_bytes(), (full_output / filename).read_bytes())
            self.assertNotEqual(row["control_population_cumulative_mean_payoff"], "")
            metadata = read_json(full_output / "config.json")
            self.assertEqual(metadata["config"]["b"], 1.8)
            if float(row["lambda_prob"]) == 0:
                self.assertEqual(int(row["total_advice_used"]), 0)
                self.assertEqual(float(row["delta_population_cumulative_mean_payoff"]), 0)
                self.assertFalse((full_output / "advisor_table.csv").exists())
            else:
                self.assertEqual(int(row["total_advice_used"]), 8)
                self.assertEqual(metadata["advisor_source"]["case_count"], 512)
                table = read_csv(full_output / "advisor_table.csv")
                usage = read_csv(full_output / "advisor_usage.csv")
                self.assertEqual(len(table), 512)
                self.assertEqual(len(usage), 512)
                self.assertEqual(sum(int(item["samples"]) for item in table), 1024)
                self.assertEqual(sum(int(item["draws"]) for item in usage), 8)
                self.assertTrue(all(len(item["grid_key"]) == 9 for item in usage))

    def test_lambda_zero_still_runs_without_grid_batch(self):
        output = self.root / "no_batch"
        self.assertEqual(self.invoke([
            "--single", "--size", "3", "--steps", "1", "--repeats", "1", "--report-every", "0",
            "--lambda", "0", "--b", "1.2", "--boundary", "open", "--paired-fermi-control",
            "--advisor-batch", str(self.root / "missing"), "--output-dir", str(output),
        ]), 0)
        run_output = next(output.iterdir())
        metadata = read_json(run_output / "config.json")
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["summary"]["total_advice_used"], 0)
        self.assertFalse((run_output / "advisor_table.csv").exists())


if __name__ == "__main__":
    unittest.main()
