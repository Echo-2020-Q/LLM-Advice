"""历史采样不改变逐轮统计，核心列与兼容输出的独立测试。"""

import csv
from pathlib import Path
import tempfile
import unittest

from main.SPD.history_writer import COMPACT_HISTORY_FIELDS, HistoryWriter


class HistoryWriterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="spd_history_writer_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "history.csv"

    def read_rows(self):
        with self.path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return reader.fieldnames, list(reader)

    def test_full_default_matches_existing_csv_bytes(self):
        rows = [
            {"round": 0, "cooperation_rate": 0.5, "label": "初始", "empty": None},
            {"round": 1, "cooperation_rate": 0.25, "label": "x,y\nz", "empty": 2},
        ]
        expected = self.root / "old.csv"
        with expected.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with HistoryWriter(self.path, total_steps=1) as writer:
            for row in rows:
                self.assertTrue(writer.write(row))
            self.assertEqual(writer.fields, list(rows[0]))
            self.assertEqual(writer.rows_written, 2)
        self.assertEqual(self.path.read_bytes(), expected.read_bytes())

    def test_sampling_keeps_initial_and_nonmultiple_final_round(self):
        with HistoryWriter(self.path, every=3, total_steps=8) as writer:
            written = [writer.write({"round": index, "value": index * 2}) for index in range(9)]
            self.assertEqual(writer.rows_written, 4)
            self.assertEqual(written, [True, False, False, True, False, False, True, False, True])
        fields, rows = self.read_rows()
        self.assertEqual(fields, ["round", "value"])
        self.assertEqual([int(row["round"]) for row in rows], [0, 3, 6, 8])

    def test_compact_selects_ordered_core_columns_without_mutating_row(self):
        row = {field: None for field in reversed(COMPACT_HISTORY_FIELDS)}
        row.update(round=0, cooperation_rate=0.5, average_payoff=1.2, extra="omit")
        before = dict(row)
        with HistoryWriter(self.path, mode="compact") as writer:
            self.assertTrue(writer.write(row))
            self.assertEqual(writer.fields, list(COMPACT_HISTORY_FIELDS))
        fields, rows = self.read_rows()
        self.assertEqual(fields, list(COMPACT_HISTORY_FIELDS))
        self.assertEqual(len(rows[0]), 18)
        self.assertEqual(row, before)
        self.assertEqual(rows[0]["cooperation_rate"], "0.5")
        self.assertEqual(rows[0]["control_cooperation_rate"], "")

    def test_compact_missing_required_column_is_not_silently_empty(self):
        row = dict.fromkeys(COMPACT_HISTORY_FIELDS, 0)
        with HistoryWriter(self.path, mode="compact", every=2, total_steps=3) as writer:
            writer.write(row)
            del row["eligible_trust_mean"]
            row["round"] = 2
            with self.assertRaisesRegex(ValueError, "eligible_trust_mean"):
                writer.write(row)
            self.assertEqual(writer.rows_written, 1)

    def test_none_creates_no_file_and_flush_is_safe(self):
        path = self.root / "uncreated" / "history.csv"
        with HistoryWriter(path, mode="none", every=3, total_steps=10) as writer:
            for index in range(11):
                self.assertFalse(writer.write({"round": index}))
            writer.flush()
            self.assertEqual(writer.fields, [])
            self.assertEqual(writer.rows_written, 0)
        self.assertFalse(path.parent.exists())

    def test_zero_steps_writes_only_initial_row(self):
        with HistoryWriter(self.path, every=10, total_steps=0) as writer:
            writer.write({"round": 0, "value": 1})
        self.assertEqual(len(self.read_rows()[1]), 1)

    def test_exception_closes_and_flushes_without_suppressing_original(self):
        error = RuntimeError("stop")
        with self.assertRaises(RuntimeError) as raised:
            with HistoryWriter(self.path) as writer:
                writer.write({"round": 0, "value": "saved"})
                raise error
        self.assertIs(raised.exception, error)
        self.assertTrue(writer._file.closed)
        self.assertEqual(self.read_rows()[1][0]["value"], "saved")

    def test_invalid_settings_are_rejected_before_creating_file(self):
        for options in (
            {"mode": "other"}, {"every": 0}, {"every": -1}, {"every": 1.5},
            {"every": True}, {"total_steps": -1}, {"total_steps": 1.5},
            {"total_steps": False},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                HistoryWriter(self.path, **options)
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
