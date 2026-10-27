"""汇总文件替换失败时的离线故障注入测试，不访问模型接口。"""

from contextlib import redirect_stderr, redirect_stdout
import csv
import errno
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from main.SPD import summary_writer


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class SummaryWriterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="spd_summary_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.writer = summary_writer.SummaryWriter(self.root, ["run_id", "result"])
        self.primary = self.root / "summary.csv"
        self.first = {"run_id": 1, "result": "C"}
        self.second = {"run_id": 2, "result": "D"}
        self.third = {"run_id": 3, "result": "C"}
        self.output = io.StringIO()
        self.addCleanup(patch.stopall)
        self.sleep = patch.object(summary_writer.time, "sleep").start()

    def save(self, rows):
        with redirect_stdout(self.output), redirect_stderr(self.output):
            return self.writer.save(rows)

    def assert_saved_rows(self, path, rows):
        self.assertEqual(read_rows(path), [
            {name: str(value) for name, value in row.items()} for row in rows
        ])

    def test_normal_save_consumes_generator_and_replaces_complete_csv(self):
        self.assertEqual(self.save(iter([self.first])), self.primary)
        self.assertEqual(self.save(iter([self.first, self.second])), self.primary)
        self.assert_saved_rows(self.primary, [self.first, self.second])
        self.assertEqual(self.writer.latest_path, self.primary)
        self.assertFalse(self.writer.pending)
        self.assertFalse((self.root / "summary.csv.tmp").exists())
        self.assertEqual(self.output.getvalue(), "")
        self.sleep.assert_not_called()

    def test_transient_permission_error_is_retried_without_fallback(self):
        original_replace = Path.replace
        attempts = []

        def deny_once(path, target):
            if Path(target) == self.primary:
                attempts.append(path)
                if len(attempts) == 1:
                    raise PermissionError("synthetic transient Windows file lock")
            return original_replace(path, target)

        with patch.object(Path, "replace", deny_once):
            self.assertEqual(self.save(iter([self.first, self.second])), self.primary)
        self.assertEqual(len(attempts), 2)
        self.sleep.assert_called_once_with(0.1)
        self.assert_saved_rows(self.primary, [self.first, self.second])
        self.assertFalse(self.writer.pending)
        self.assertEqual(list(self.root.glob("summary.pending*.csv")), [])
        self.assertEqual(self.output.getvalue(), "")

    def test_persistent_lock_preserves_primary_and_updates_complete_backup_then_recovers(self):
        self.save([self.first])
        original_bytes = self.primary.read_bytes()
        original_replace = Path.replace
        attempts = []

        def deny_primary(path, target):
            if Path(target) == self.primary:
                attempts.append(path)
                raise PermissionError("synthetic persistent Windows file lock")
            return original_replace(path, target)

        backup = self.root / "summary.pending.csv"
        with patch.object(Path, "replace", deny_primary):
            self.assertEqual(self.save(iter([self.first, self.second])), backup)
            first_warning = self.output.getvalue()
            self.assertIn(str(backup), first_warning)
            self.assertEqual(self.save(iter([self.first, self.second, self.third])), backup)
            self.assertEqual(self.output.getvalue(), first_warning)
        self.assertEqual(len(attempts), 3)
        self.sleep.assert_called_once_with(0.1)
        self.assertEqual(self.primary.read_bytes(), original_bytes)
        self.assert_saved_rows(backup, [self.first, self.second, self.third])
        self.assertEqual(self.writer.latest_path, backup)
        self.assertTrue(self.writer.pending)
        backup_bytes = backup.read_bytes()

        self.assertEqual(self.save([self.first, self.second, self.third]), self.primary)
        self.assert_saved_rows(self.primary, [self.first, self.second, self.third])
        self.assertFalse(self.writer.pending)
        self.assertEqual(self.writer.latest_path, self.primary)
        self.assertEqual(backup.read_bytes(), backup_bytes)
        self.assertGreater(len(self.output.getvalue()), len(first_warning))
        self.assertIn(str(self.primary), self.output.getvalue()[len(first_warning):])

    def test_locked_backup_uses_unique_file_and_reuses_it_until_locked(self):
        self.save([self.first])
        primary_bytes = self.primary.read_bytes()
        fixed_backup = self.root / "summary.pending.csv"
        fixed_backup.write_text("older backup must remain intact\n", encoding="utf-8")
        fixed_backup_bytes = fixed_backup.read_bytes()
        locked = {self.primary, fixed_backup}
        original_replace = Path.replace

        def deny_locked_targets(path, target):
            if Path(target) in locked:
                raise PermissionError("synthetic locked CSV")
            return original_replace(path, target)

        with patch.object(Path, "replace", deny_locked_targets):
            unique = self.save([self.first, self.second])
            self.assertRegex(unique.name, r"^summary\.pending\.[^.]+\.csv$")
            first_warning = self.output.getvalue()
            self.assertIn(str(unique), first_warning)
            self.assertEqual(self.save([self.first, self.second, self.third]), unique)
            self.assertEqual(self.output.getvalue(), first_warning)
            unique_bytes = unique.read_bytes()

            locked.add(unique)
            replacement = self.save([self.first, self.second, self.third])
            self.assertNotEqual(replacement, unique)
            self.assertRegex(replacement.name, r"^summary\.pending\.[^.]+\.csv$")
            self.assertIn(str(replacement), self.output.getvalue()[len(first_warning):])
        self.assertEqual(self.primary.read_bytes(), primary_bytes)
        self.assertEqual(fixed_backup.read_bytes(), fixed_backup_bytes)
        self.assertEqual(unique.read_bytes(), unique_bytes)
        self.assert_saved_rows(replacement, [self.first, self.second, self.third])
        self.assertEqual(self.writer.latest_path, replacement)
        self.assertTrue(self.writer.pending)
        self.assertEqual(len(list(self.root.glob("summary.pending.*.csv"))), 2)

    def test_non_permission_replace_error_propagates_without_fallback(self):
        original_replace = Path.replace

        def disk_full(path, target):
            if Path(target) == self.primary:
                raise OSError(errno.ENOSPC, "synthetic disk full")
            return original_replace(path, target)

        with patch.object(Path, "replace", disk_full):
            with self.assertRaises(OSError) as raised:
                self.save([self.first])
        self.assertEqual(raised.exception.errno, errno.ENOSPC)
        self.assertFalse(self.writer.pending)
        self.assertEqual(list(self.root.glob("summary.pending*.csv")), [])
        self.sleep.assert_not_called()

    def test_temporary_file_write_permission_error_propagates(self):
        original_open = Path.open

        def deny_temporary_open(path, *args, **kwargs):
            if path == self.root / "summary.csv.tmp":
                raise PermissionError("synthetic directory write denied")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", deny_temporary_open), \
                patch.object(Path, "replace") as replace:
            with self.assertRaises(PermissionError):
                self.save([self.first])
        replace.assert_not_called()
        self.sleep.assert_not_called()
        self.assertFalse(self.writer.pending)
        self.assertEqual(list(self.root.glob("summary.pending*.csv")), [])

    def test_all_replacement_targets_denied_propagates_and_preserves_temporary_csv(self):
        with patch.object(Path, "replace", side_effect=PermissionError("synthetic access denied")):
            with self.assertRaises(PermissionError):
                self.save(iter([self.first, self.second]))
        self.assertFalse(self.writer.pending)
        self.assertEqual(self.writer.latest_path, self.primary)
        self.assert_saved_rows(self.root / "summary.csv.tmp", [self.first, self.second])
        self.assertFalse(self.primary.exists())
        self.assertEqual(list(self.root.glob("summary.pending*.csv")), [])

    def test_keyboard_interrupt_is_not_swallowed(self):
        with patch.object(Path, "replace", side_effect=KeyboardInterrupt("synthetic interruption")):
            with self.assertRaises(KeyboardInterrupt):
                self.save([self.first])
        self.sleep.assert_not_called()
        self.assertFalse(self.writer.pending)
        self.assert_saved_rows(self.root / "summary.csv.tmp", [self.first])
        self.assertEqual(list(self.root.glob("summary.pending*.csv")), [])


if __name__ == "__main__":
    unittest.main()
