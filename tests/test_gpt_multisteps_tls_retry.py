"""Offline regressions for connection-close retries in new and resumed batches.

Run with: python -m unittest discover -s tests -p test_gpt_multisteps_tls_retry.py -v
The shared fixtures forbid network/credential access; sleeps are always mocked.
"""

import json
import ssl
import unittest
from pathlib import Path
from unittest.mock import call, patch

import test_gpt_multisteps_resume as fixtures
import test_gpt_multisteps_grid as grid_fixtures


def closed_connection(cause=None, error_type=fixtures.FakeConnection):
    error = error_type('Connection error.')
    wrapper = RuntimeError('Transport failed')
    wrapper.__cause__ = cause if cause is not None else ssl.SSLEOFError(
        8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol')
    error.__cause__ = wrapper
    return error


class TLSRetryTests(unittest.TestCase):
    # Reuse fixture helpers without inheriting and rerunning the resume test suite.
    setUpClass = classmethod(fixtures.TestResume.setUpClass.__func__)
    setUp = fixtures.TestResume.setUp
    tearDown = fixtures.TestResume.tearDown
    write = fixtures.TestResume.write
    fixture = fixtures.TestResume.fixture
    invoke = fixtures.TestResume.invoke
    invoke_new = fixtures.TestResume.invoke_new
    assert_final = fixtures.TestResume.assert_final

    @staticmethod
    def read(path):
        return json.loads(path.read_text(encoding='utf-8'))

    def last_pending(self):
        return self.fixture(n=1, successes=range(1, 10), failures={10})

    def test_resume_retries_same_payload_and_preserves_successes_and_failures(self):
        directory, _, records = self.last_pending()
        config_bytes = (directory / 'config.json').read_bytes()
        successes = {index: (directory / f'records/{index:06d}.json').read_bytes()
                     for index in range(1, 10)}
        old_failure = (directory / 'records/000010.json').read_bytes()
        failures_during_wait = []

        def during_wait(seconds):
            self.assertEqual(seconds, 5)
            latest = directory / 'records/000010.json'
            self.assertEqual(self.read(latest)['status'], 'error')
            self.assertIn('diagnostics', self.read(latest)['error'])
            failures_during_wait.append(latest.read_bytes())

        with patch.object(self.module.time, 'sleep', side_effect=during_wait) as sleep:
            result = self.invoke(directory, effects=[closed_connection(), closed_connection(),
                                                     fixtures.FakeResponse('C')])
        self.assertEqual(result['code'], 0, result['stderr'])
        self.assertEqual(sleep.call_args_list, [call(5), call(5)])
        self.assertEqual(len(result['sdk'].calls), 3)
        for client, request in result['sdk'].calls:
            self.assertEqual(request, records[10]['request'])
            self.assertEqual(client.options['max_retries'], 0)
            self.assertTrue(client.exited)
        self.assertEqual((directory / 'config.json').read_bytes(), config_bytes)
        for index, original in successes.items():
            self.assertEqual((directory / f'records/{index:06d}.json').read_bytes(), original)
        archived = [path.read_bytes() for path in (directory / 'attempts/000010').glob('*.json')]
        self.assertEqual(len(archived), 3)
        self.assertIn(old_failure, archived)
        for failure in failures_during_wait:
            self.assertIn(failure, archived)
        summary = self.assert_final(directory, 10)
        self.assertEqual(summary['archived_attempts'], 3)
        self.assertEqual(summary['api_calls_started_total'], 13)
        self.assertIn('重试', result['stdout'])
        self.assertFalse((directory / '.run.lock').exists())

    def test_default_retry_budget_is_five_extra_attempts_then_resume_is_possible(self):
        directory, _, _ = self.last_pending()
        with patch.object(self.module.time, 'sleep') as sleep:
            result = self.invoke(directory, effects=[closed_connection() for _ in range(7)])
        self.assertEqual(result['code'], 1)
        self.assertEqual(len(result['sdk'].calls), 6)
        self.assertEqual(sleep.call_args_list, [call(5)] * 5)
        failure_bytes = (directory / 'records/000010.json').read_bytes()
        summary = self.read(directory / 'summary.json')
        self.assertEqual(summary['status'], 'stopped_on_error')
        self.assertEqual(summary['successful_requests'], 9)
        self.assertEqual(summary['failed_requests'], 1)
        self.assertEqual(summary['pending_requests'], 1)
        self.assertEqual(summary['api_calls_started_total'], 16)
        self.assertEqual(summary['archived_attempts'], 6)
        self.assertFalse((directory / '.run.lock').exists())

        with patch.object(self.module.time, 'sleep') as sleep:
            resumed = self.invoke(directory)
        self.assertEqual(resumed['code'], 0, resumed['stderr'])
        self.assertEqual(len(resumed['sdk'].calls), 1)
        sleep.assert_not_called()
        self.assertIn(failure_bytes, [path.read_bytes() for path in
                                     (directory / 'attempts/000010').glob('*.json')])
        self.assertEqual(self.assert_final(directory, 10)['api_calls_started_total'], 17)

    def test_new_batch_retries_each_item_with_its_own_budget(self):
        effects = [closed_connection(), fixtures.FakeResponse('C'),
                   closed_connection(), fixtures.FakeResponse('D')]
        with patch.object(self.module.time, 'sleep') as sleep:
            result = self.invoke_new(extra=['--tls-retries', '1', '--retry-delay', '0.25'],
                                     effects=effects)
        self.assertEqual(result['code'], 0, result['stderr'])
        self.assertEqual(len(result['sdk'].calls), 12)
        self.assertEqual(sleep.call_args_list, [call(0.25), call(0.25)])
        requests = [request for _, request in result['sdk'].calls]
        self.assertEqual(requests[0], requests[1])
        self.assertEqual(requests[2], requests[3])
        self.assertNotEqual(requests[0], requests[2])
        self.assertTrue(all(client.options['max_retries'] == 0 for client in result['sdk'].clients))
        directory = next((self.root / 'new_logs').iterdir())
        summary = self.assert_final(directory, 10)
        self.assertEqual(summary['archived_attempts'], 2)
        self.assertEqual(summary['api_calls_started_total'], 12)
        self.assertEqual(len(list((directory / 'attempts/000001').glob('*.json'))), 1)
        self.assertEqual(len(list((directory / 'attempts/000002').glob('*.json'))), 1)

    def test_zero_retries_keeps_stop_on_first_failure(self):
        directory, _, _ = self.last_pending()
        with patch.object(self.module.time, 'sleep') as sleep:
            result = self.invoke(directory, extra=['--tls-retries', '0'],
                                 effects=[closed_connection()])
        self.assertEqual(result['code'], 1)
        self.assertEqual(len(result['sdk'].calls), 1)
        sleep.assert_not_called()
        self.assertEqual(self.read(directory / 'summary.json')['archived_attempts'], 1)

    def test_timeout_certificate_http_and_unrelated_connection_errors_do_not_retry(self):
        errors = [
            closed_connection(error_type=fixtures.FakeTimeout),
            closed_connection(ssl.SSLCertVerificationError(1, 'certificate verify failed')),
            closed_connection(ssl.SSLError(1, '[SSL: WRONG_VERSION_NUMBER] wrong version number')),
            fixtures.FakeConnection('Connection refused'),
            fixtures.FakeStatus(400),
            fixtures.FakeStatus(503),
        ]
        for index, error in enumerate(errors):
            with self.subTest(error=type(error).__name__, index=index):
                directory, _, _ = self.last_pending()
                renamed = self.root / f'failure_{index}'
                directory.rename(renamed)
                with patch.object(self.module.time, 'sleep') as sleep:
                    result = self.invoke(renamed, effects=[error])
                self.assertEqual(result['code'], 1)
                self.assertEqual(len(result['sdk'].calls), 1)
                sleep.assert_not_called()
                self.assertEqual(self.read(renamed / 'summary.json')['pending_requests'], 1)

    def test_context_chain_and_connection_reset_are_retried(self):
        errors = [closed_connection(ConnectionResetError(104, 'Connection reset by peer')),
                  closed_connection(RuntimeError('Server disconnected without sending a response.'))]
        context_error = fixtures.FakeConnection('Connection error.')
        context_error.__context__ = ssl.SSLEOFError(8, 'EOF occurred in violation of protocol')
        errors.append(context_error)
        for index, error in enumerate(errors):
            with self.subTest(index=index):
                directory, _, _ = self.last_pending()
                renamed = self.root / f'closed_{index}'
                directory.rename(renamed)
                with patch.object(self.module.time, 'sleep'):
                    result = self.invoke(renamed, extra=['--tls-retries', '1', '--retry-delay', '0'],
                                         effects=[error, fixtures.FakeResponse('D')])
                self.assertEqual(result['code'], 0, result['stderr'])
                self.assertEqual(len(result['sdk'].calls), 2)
                self.assert_final(renamed, 10)

    def test_keyboard_interrupt_during_delay_keeps_failure_and_allows_resume(self):
        directory, _, _ = self.last_pending()
        with patch.object(self.module.time, 'sleep', side_effect=KeyboardInterrupt()):
            result = self.invoke(directory, effects=[closed_connection()])
        self.assertEqual(result['code'], 130)
        self.assertEqual(len(result['sdk'].calls), 1)
        failure_bytes = (directory / 'records/000010.json').read_bytes()
        self.assertEqual(self.read(directory / 'records/000010.json')['status'], 'error')
        summary = self.read(directory / 'summary.json')
        self.assertEqual(summary['status'], 'cancelled')
        self.assertEqual(summary['pending_requests'], 1)
        self.assertEqual(summary['api_calls_started_total'], 11)
        self.assertEqual(summary['archived_attempts'], 1)
        self.assertFalse((directory / '.run.lock').exists())

        with patch.object(self.module.time, 'sleep') as sleep:
            resumed = self.invoke(directory)
        self.assertEqual(resumed['code'], 0, resumed['stderr'])
        sleep.assert_not_called()
        self.assertEqual(len(resumed['sdk'].calls), 1)
        self.assertEqual(self.assert_final(directory, 10)['api_calls_started_total'], 12)
        self.assertIn(failure_bytes, [path.read_bytes() for path in
                                     (directory / 'attempts/000010').glob('*.json')])

    def test_new_batch_keyboard_interrupt_during_delay_can_resume(self):
        with patch.object(self.module.time, 'sleep', side_effect=KeyboardInterrupt()):
            result = self.invoke_new(effects=[closed_connection()])
        self.assertEqual(result['code'], 130)
        self.assertEqual(len(result['sdk'].calls), 1)
        directory = next((self.root / 'new_logs').iterdir())
        summary = self.read(directory / 'summary.json')
        self.assertEqual(summary['status'], 'cancelled')
        self.assertEqual(summary['api_calls_started_total'], 1)
        self.assertEqual(summary['successful_requests'], 0)
        self.assertEqual(self.read(directory / 'records/000001.json')['status'], 'error')
        self.assertFalse((directory / '.run.lock').exists())
        with patch.object(self.module.time, 'sleep') as sleep:
            resumed = self.invoke(directory)
        self.assertEqual(resumed['code'], 0, resumed['stderr'])
        sleep.assert_not_called()
        self.assertEqual(len(resumed['sdk'].calls), 10)
        self.assertEqual(self.assert_final(directory, 10)['api_calls_started_total'], 11)

    def test_continue_on_error_advances_only_after_retry_budget_is_exhausted(self):
        with patch.object(self.module.time, 'sleep') as sleep:
            result = self.invoke_new(extra=['--continue-on-error', '--tls-retries', '2'],
                                     effects=[closed_connection() for _ in range(3)])
        self.assertEqual(result['code'], 1)
        self.assertEqual(len(result['sdk'].calls), 12)
        self.assertEqual(sleep.call_args_list, [call(5), call(5)])
        requests = [request for _, request in result['sdk'].calls]
        self.assertEqual(requests[:3], [requests[0]] * 3)
        self.assertNotEqual(requests[2], requests[3])
        directory = next((self.root / 'new_logs').iterdir())
        summary = self.read(directory / 'summary.json')
        self.assertEqual(summary['status'], 'completed_with_errors')
        self.assertEqual(summary['successful_requests'], 9)
        self.assertEqual(summary['failed_requests'], 1)
        self.assertEqual(summary['api_calls_started_total'], 12)
        self.assertEqual(summary['archived_attempts'], 2)

    def test_archive_failure_or_interrupt_after_move_never_duplicates_an_attempt(self):
        for after_move in (False, True):
            with self.subTest(interrupt_after_move=after_move):
                directory, _, _ = self.last_pending()
                renamed = self.root / f'archive_failure_{after_move}'
                directory.rename(renamed)
                latest = renamed / 'records/000010.json'
                saved_failures = []
                original_replace = Path.replace
                interrupted = False

                def replace(source, target):
                    nonlocal interrupted
                    is_retry_move = (source == latest and saved_failures and not interrupted
                                     and Path(target).parent == renamed / 'attempts/000010')
                    if is_retry_move:
                        interrupted = True
                        if after_move:
                            original_replace(source, target)
                            raise KeyboardInterrupt()
                        raise OSError('Offline archive move failure')
                    return original_replace(source, target)

                with patch.object(self.module.time, 'sleep',
                                  side_effect=lambda _: saved_failures.append(latest.read_bytes())), \
                        patch.object(Path, 'replace', new=replace):
                    result = self.invoke(renamed, effects=[closed_connection()])
                self.assertTrue(interrupted)
                self.assertEqual(result['code'], 130 if after_move else 1)
                self.assertEqual(len(result['sdk'].calls), 1)
                self.assertEqual(latest.read_bytes(), saved_failures[0])
                _, records, _, archived = self.module.read_batch(renamed)
                self.assertEqual(len(archived), 1)
                self.assertEqual(sum(record['api_call_started'] for record in records.values())
                                 + sum(record['api_call_started'] for record in archived), 11)
                summary = self.read(renamed / 'summary.json')
                self.assertEqual(summary['status'], 'cancelled' if after_move else 'logging_error')
                self.assertEqual(summary['api_calls_started_total'], 11)
                self.assertEqual(summary['archived_attempts'], 1)
                self.assertFalse((renamed / '.run.lock').exists())

    def test_retry_record_write_failure_stops_before_second_api_call(self):
        directory, _, _ = self.last_pending()
        latest = directory / 'records/000010.json'
        saved_failures = []
        original_save = self.module.save_record
        failed_write = False

        def save(path, record, api_key):
            nonlocal failed_write
            if (path == latest and record['status'] == 'running'
                    and saved_failures and not failed_write):
                failed_write = True
                raise OSError('Offline retry record write failure')
            return original_save(path, record, api_key)

        with patch.object(self.module.time, 'sleep',
                          side_effect=lambda _: saved_failures.append(latest.read_bytes())), \
                patch.object(self.module, 'save_record', side_effect=save):
            result = self.invoke(directory, effects=[closed_connection()])
        self.assertTrue(failed_write)
        self.assertEqual(result['code'], 1)
        self.assertEqual(len(result['sdk'].calls), 1)
        _, records, _, archived = self.module.read_batch(directory)
        self.assertEqual(sum(record['api_call_started'] for record in records.values())
                         + sum(record['api_call_started'] for record in archived), 11)
        attempt_bytes = [path.read_bytes() for path in
                         (directory / 'attempts/000010').glob('*.json')]
        attempt_bytes.append(latest.read_bytes())
        self.assertEqual(attempt_bytes.count(saved_failures[0]), 1)
        summary = self.read(directory / 'summary.json')
        self.assertEqual(summary['status'], 'logging_error')
        self.assertEqual(summary['api_calls_started_total'], 11)
        self.assertFalse((directory / '.run.lock').exists())

    def test_interrupt_after_archive_move_and_failed_rollback_keeps_exact_call_count(self):
        directory, _, _ = self.last_pending()
        latest = directory / 'records/000010.json'
        saved_failures = []
        original_replace = Path.replace
        moved_archive = None
        rollback_failed = False

        def replace(source, target):
            nonlocal moved_archive, rollback_failed
            if source == latest and saved_failures and moved_archive is None:
                moved_archive = Path(target)
                original_replace(source, target)
                raise KeyboardInterrupt()
            if source == moved_archive and Path(target) == latest:
                rollback_failed = True
                raise OSError('Offline archive rollback failure')
            return original_replace(source, target)

        with patch.object(self.module.time, 'sleep',
                          side_effect=lambda _: saved_failures.append(latest.read_bytes())), \
                patch.object(Path, 'replace', new=replace):
            result = self.invoke(directory, effects=[closed_connection()])
        self.assertTrue(rollback_failed)
        self.assertEqual(result['code'], 1)
        self.assertEqual(len(result['sdk'].calls), 1)
        self.assertEqual(moved_archive.read_bytes(), saved_failures[0])
        _, records, _, archived = self.module.read_batch(directory)
        self.assertFalse(records[10]['api_call_started'])
        self.assertEqual(len(archived), 2)
        self.assertEqual(sum(record['api_call_started'] for record in records.values())
                         + sum(record['api_call_started'] for record in archived), 11)
        summary = self.read(directory / 'summary.json')
        self.assertEqual(summary['status'], 'logging_error')
        self.assertEqual(summary['api_calls_started_total'], 11)
        self.assertFalse((directory / '.run.lock').exists())

    def test_invalid_retry_settings_fail_before_credentials_or_network(self):
        invalid_options = [
            ['--tls-retries', '-1'], ['--tls-retries', '1.5'],
            ['--retry-delay', '-0.1'], ['--retry-delay', 'nan'],
            ['--retry-delay', 'inf'], ['--retry-delay', '-inf'],
        ]
        for options in invalid_options:
            with self.subTest(options=options), self.assertRaises(SystemExit) as raised:
                self.invoke_new(extra=[*options, '--dry-run'], forbid_config=True, forbid_sdk=True)
            self.assertEqual(raised.exception.code, 2)
        self.assertFalse((self.root / 'new_logs').exists())


class GridTLSRetryTests(unittest.TestCase):
    setUpClass = classmethod(grid_fixtures.GridBatchTests.setUpClass.__func__)
    setUp = grid_fixtures.GridBatchTests.setUp
    invoke = grid_fixtures.GridBatchTests.invoke
    write_json = staticmethod(grid_fixtures.GridBatchTests.write_json)
    fixture = grid_fixtures.GridBatchTests.fixture

    def test_grid_resume_retries_only_pending_case_with_original_request(self):
        directory, config = self.fixture()
        successes = {index: (directory / f'records/{index:06d}.json').read_bytes()
                     for index in range(1, 511)}
        error = grid_fixtures.FakeConnectionError('Connection error.')
        error.__cause__ = ssl.SSLEOFError(8, 'EOF occurred in violation of protocol')
        with patch.object(self.module.time, 'sleep') as sleep:
            result = self.invoke(['--resume', str(directory)], effects=[error])
        self.assertEqual(result['code'], 0, result['stderr'])
        self.assertEqual(sleep.call_args_list, [call(5)])
        self.assertEqual(result['sdk'].calls, [config['requests'][510], config['requests'][510],
                                              config['requests'][511]])
        for index, original in successes.items():
            self.assertEqual((directory / f'records/{index:06d}.json').read_bytes(), original)
        summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(summary['successful_requests'], 512)
        self.assertEqual(summary['pending_requests'], 0)
        self.assertEqual(summary['api_calls_started_total'], 514)
        self.assertEqual(summary['archived_attempts'], 2)


if __name__ == '__main__':
    unittest.main()
