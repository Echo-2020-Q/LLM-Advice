"""Offline regression tests for legacy neighbor-count requests and batch resumption.

Run with: python -m unittest discover -s tests -p test_gpt_multisteps_resume.py -v
All records are synthetic; SDK calls, network access, and real credentials are blocked.
"""
import builtins
import contextlib
import copy
import csv
import importlib.util
import io
import json
import os
import socket
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]


def guard_open(original):
    def guarded(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            candidate = Path(os.fsdecode(file)).resolve()
            if candidate.is_relative_to(PROJECT / 'API'):
                raise AssertionError('Test attempted to access real credentials')
        return original(file, *args, **kwargs)
    return guarded


class FakeError(Exception):
    pass


class FakeConnection(FakeError):
    pass


class FakeTimeout(FakeConnection):
    pass


class FakeStatus(FakeError):
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__('Do not log secret sk-fake-test-token')


class FakeResponse:
    def __init__(self, text='D', status='completed', index=1):
        self.output_text = text
        self.status = status
        self.id = f'resp_fake_{index}'
        self.model = 'gpt-5.5'

    def model_dump(self, mode='json'):
        return {
            'id': self.id, 'model': self.model, 'status': self.status,
            'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': self.output_text}]}],
            'usage': {'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12},
        }


class FakeSDK(types.ModuleType):
    def __init__(self, behavior=None):
        super().__init__('openai')
        self.behavior = behavior or []
        self.clients = []
        self.calls = []
        self.http_clients = []
        self.APITimeoutError = FakeTimeout
        self.APIConnectionError = FakeConnection
        self.APIStatusError = FakeStatus
        self.APIError = FakeError
        sdk = self

        class HttpClient:
            def __init__(self, **kwargs):
                self.options = dict(kwargs)
                self.closed = False
                sdk.http_clients.append(self)

            def close(self):
                self.closed = True

        class Client:
            def __init__(self, **kwargs):
                self.options = copy.deepcopy(kwargs)
                self.http_client = kwargs.get('http_client')
                self.entered = False
                self.exited = False
                self.responses = self
                sdk.clients.append(self)

            def __enter__(self):
                self.entered = True
                return self

            def __exit__(self, *exc):
                self.exited = True
                if self.http_client is not None:
                    self.http_client.close()
                return False

            def create(self, **kwargs):
                sdk.calls.append((self, copy.deepcopy(kwargs)))
                index = len(sdk.calls)
                effect = sdk.behavior[index - 1] if index <= len(sdk.behavior) else FakeResponse('C' if index % 2 else 'D', index=index)
                if isinstance(effect, BaseException):
                    raise effect
                if callable(effect):
                    return effect()
                return effect

        self.OpenAI = Client
        self.DefaultHttpxClient = HttpClient


KEY = 'sk-fake-resume-token'
URL = 'https://example.invalid/v1'


class TestResume(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Class fixtures also run under unittest discovery, unlike an __main__ guard.
        guards = contextlib.ExitStack()
        cls.addClassCleanup(guards.close)
        guards.enter_context(patch('builtins.open', side_effect=guard_open(builtins.open)))
        guards.enter_context(patch('io.open', side_effect=guard_open(io.open)))
        for target in ('socket.socket.connect', 'socket.socket.connect_ex', 'socket.create_connection'):
            guards.enter_context(patch(target, side_effect=AssertionError('Network forbidden during offline tests')))
        original_path = sys.path[:]
        cls.addClassCleanup(lambda: sys.path.__setitem__(slice(None), original_path))
        guards.enter_context(patch.object(sys, 'dont_write_bytecode', True))
        sys.path.insert(0, str(PROJECT / 'main'))
        spec = importlib.util.spec_from_file_location('tested_resume', PROJECT / 'main/GPT_Multisteps.py')
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def setUp(self):
        # 旧格式 GPT 批次测试独立于用户当前选择的默认模型服务。
        self.patch_provider = patch.object(self.module, 'PROVIDER', 'gpt')
        self.patch_provider.start()
        self.temp = tempfile.TemporaryDirectory(prefix='offline_resume_')
        self.root = Path(self.temp.name)
        self.patch_logs = patch.object(self.module, 'LOG_DIR', self.root / 'new_logs')
        self.patch_logs.start()
        self.patch_format = patch.object(self.module, 'OBSERVATION_FORMAT', 'neighbor_counts')
        self.patch_format.start()

    def tearDown(self):
        self.patch_provider.stop()
        self.patch_format.stop()
        self.patch_logs.stop()
        self.temp.cleanup()

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

    def fixture(self, n=10, successes=None, failures=None, running=None, overrides=None):
        successes = set(range(1, 46)) if successes is None else set(successes)
        failures = {46} if failures is None else set(failures)
        running = set() if running is None else set(running)
        directory = self.root / 'old_batch'
        cases = self.module.build_cases(1.5)
        config = {
            'batch_id': 'fake-existing-batch', 'model': 'gpt-5.5', 'base_url': URL,
            'b': 1.5, 'n': n, 'planned_requests': 10*n, 'system_prompt': self.module.SYSTEM_PROMPT,
            'cases': cases, 'max_output_tokens': 1024, 'timeout_seconds': 120,
            'max_retries': 0, 'store': False, 'continue_on_error': False,
            'order': '每轮依次 C/0..4、D/0..4；共 n 轮；每次只提供当前局面，无历史。',
        }
        if overrides:
            config.update(overrides)
        config['cases'] = self.module.build_cases(config['b'])
        self.write(directory/'config.json', config)
        records = {}
        for index in sorted(successes | failures | running):
            case = config['cases'][(index-1)%10]
            success = index in successes
            status = 'success' if success else 'running' if index in running else 'error'
            response = FakeResponse('D', index=index).model_dump() if success else None
            record = {
                'batch_id': config['batch_id'], 'request_index': index,
                'repetition': (index-1)//10+1, 'combination_index': (index-1)%10+1,
                'observation': case, 'base_url': config['base_url'],
                'client_options': {'timeout_seconds': config['timeout_seconds'], 'max_retries': 0},
                'request': {'model': config['model'], 'instructions': config['system_prompt'],
                    'input': self.module.build_user_prompt(case), 'max_output_tokens': config['max_output_tokens'], 'store': False},
                'api_call_started': True, 'started_at': '2026-10-15T10:00:00.000+08:00',
                'finished_at': None if status == 'running' else '2026-10-15T10:00:01.000+08:00',
                'duration_seconds': None if status == 'running' else 1.0,
                'status': status, 'action': 'D' if success else None,
                'output_text': 'D' if success else None, 'response': response,
                'error': None if success or status == 'running' else {'type': 'APIConnectionError', 'http_status': None, 'message': '连接失败，请检查网络与代理。'},
            }
            self.write(directory/f'records/{index:06d}.json', record)
            records[index] = record
        # Deliberately stale summary and CSV verify records are the source of truth.
        self.write(directory/'summary.json', {
            'batch_id': config['batch_id'], 'started_at': '2026-10-15T10:00:00.000+08:00',
            'finished_at': '2026-10-15T11:00:00.000+08:00', 'status': 'stopped_on_error',
            'b': config['b'], 'n': n, 'planned_requests': 10*n, 'api_calls_started': len(records),
            'processed_cases': len(records), 'successful_requests': len(successes),
            'failed_requests': len(failures), 'cancelled_requests': 0,
            'not_run_requests': 10*n-len(records),
            'cases': [{**c, 'C': 0, 'D': 0, 'errors': 0, 'cancelled': 0} for c in config['cases']],
        })
        with (directory/'results.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.module.CSV_FIELDS)
            writer.writeheader()
            for index, record in records.items():
                writer.writerow(self.module.csv_row(record, f'records/{index:06d}.json'))
        return directory, config, records

    def invoke(self, directory, extra=(), effects=None, forbid_config=False, forbid_sdk=False, configured_url=URL):
        sdk = FakeSDK(effects)
        stdout, stderr = io.StringIO(), io.StringIO()
        original_import = builtins.__import__
        def imports(name, *args, **kwargs):
            if forbid_sdk and (name == 'openai' or name.startswith('openai.')):
                raise AssertionError('This mode must not import the SDK')
            return original_import(name, *args, **kwargs)
        config_kwargs = {'side_effect': AssertionError('This mode must not read credentials')} if forbid_config else {'return_value': (KEY, configured_url)}
        with patch.dict(sys.modules, {'openai': sdk}), patch.object(self.module, 'load_config', **config_kwargs) as config, patch('builtins.__import__', side_effect=imports), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = self.module.main(['--resume', str(directory), *extra])
        return {'code': code, 'sdk': sdk, 'stdout': stdout.getvalue(), 'stderr': stderr.getvalue(), 'config_calls': config.call_count}

    def invoke_new(self, extra=(), effects=None, forbid_config=False, forbid_sdk=False):
        sdk = FakeSDK(effects)
        stdout, stderr = io.StringIO(), io.StringIO()
        original_import = builtins.__import__

        def imports(name, *args, **kwargs):
            if forbid_sdk and (name == 'openai' or name.startswith('openai.')):
                raise AssertionError('This mode must not import the SDK')
            return original_import(name, *args, **kwargs)

        config_kwargs = {'side_effect': AssertionError('This mode must not read credentials')} if forbid_config else {'return_value': (KEY, URL)}
        with patch.dict(sys.modules, {'openai': sdk}), patch.object(self.module, 'load_config', **config_kwargs) as config, patch('builtins.__import__', side_effect=imports), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = self.module.main(['--n', '1', *extra])
        return {'code': code, 'sdk': sdk, 'stdout': stdout.getvalue(), 'stderr': stderr.getvalue(), 'config_calls': config.call_count}

    def snapshot(self, directory):
        return {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob('*') if p.is_file()}

    def assert_final(self, directory, total):
        records = {int(p.stem): json.loads(p.read_text(encoding='utf-8')) for p in (directory/'records').glob('*.json') if p.stem.isdigit()}
        self.assertEqual(len(records), total)
        self.assertEqual({r['status'] for r in records.values()}, {'success'})
        rows = list(csv.DictReader(io.StringIO((directory/'results.csv').read_text(encoding='utf-8-sig'))))
        self.assertEqual(len(rows), total)
        self.assertEqual({int(r['request_index']) for r in rows}, set(range(1,total+1)))
        self.assertEqual({r['status'] for r in rows}, {'success'})
        summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(summary['successful_requests'], total)
        self.assertEqual(summary['failed_requests'], 0)
        self.assertEqual(summary['not_run_requests'], 0)
        self.assertEqual(sum(c['C']+c['D'] for c in summary['cases']), total)
        return summary

    def assert_archived(self, directory, record):
        candidates = []
        for p in directory.rglob('*.json'):
            if p.parent == directory/'records' or p.name in ('config.json', 'summary.json'):
                continue
            try:
                data = json.loads(p.read_text(encoding='utf-8'))
            except (ValueError, OSError):
                continue
            if isinstance(data, dict) and data.get('request_index') == record['request_index'] and data.get('status') == record['status']:
                candidates.append(data)
        self.assertTrue(candidates, f'No archive found for {record["request_index"]}/{record["status"]}')
        self.assertTrue(any(all(c.get(k) == v for k, v in record.items()) for c in candidates))

    def test_45_successes_resume_only_55_and_preserve_old_data(self):
        directory, config, records = self.fixture()
        before = {i:(directory/f'records/{i:06d}.json').read_bytes() for i in range(1,46)}
        result = self.invoke(directory)
        self.assertEqual(result['code'], 0, result['stderr'])
        self.assertEqual(len(result['sdk'].calls), 55)
        self.assertEqual(len(result['sdk'].clients), 55)
        for i, old in before.items():
            self.assertEqual((directory/f'records/{i:06d}.json').read_bytes(), old)
        self.assertEqual(json.loads((directory/'config.json').read_text(encoding='utf-8')), config)
        self.assert_archived(directory, records[46])
        summary = self.assert_final(directory, 100)
        self.assertEqual(summary['api_calls_started'], 100)
        self.assertEqual(summary['api_calls_started_total'], 101)
        self.assertEqual(summary['archived_attempts'], 1)
        self.assertEqual(summary['pending_requests'], 0)
        for i, (client, request) in enumerate(result['sdk'].calls, 46):
            self.assertEqual(request, {'model': config['model'], 'instructions': config['system_prompt'],
                'input': self.module.build_user_prompt(config['cases'][(i-1)%10]),
                'max_output_tokens': 1024, 'store': False})
            self.assertEqual(client.options['max_retries'], 0)
            self.assertTrue(client.entered and client.exited)
        self.assertFalse((self.root/'new_logs').exists())

    def test_failure_again_then_resume_preserves_both_attempts(self):
        directory, _, records = self.fixture()
        first = self.invoke(directory, effects=[FakeConnection('fake offline failure')])
        self.assertEqual(first['code'], 1)
        self.assertEqual(len(first['sdk'].calls), 1)
        failed_again = json.loads((directory/'records/000046.json').read_text(encoding='utf-8'))
        second = self.invoke(directory)
        self.assertEqual(second['code'], 0, second['stderr'])
        self.assertEqual(len(second['sdk'].calls), 55)
        self.assert_archived(directory, records[46])
        self.assert_archived(directory, failed_again)
        self.assert_final(directory, 100)

    def test_sparse_successes_errors_running_and_missing(self):
        directory, _, records = self.fixture(n=1, successes={2,4,6,8,10}, failures={1,3}, running={5})
        before = {i:(directory/f'records/{i:06d}.json').read_bytes() for i in [2,4,6,8,10]}
        result = self.invoke(directory)
        self.assertEqual(result['code'], 0, result['stderr'])
        self.assertEqual(len(result['sdk'].calls), 5)
        for i, old in before.items():
            self.assertEqual((directory/f'records/{i:06d}.json').read_bytes(), old)
        for i in [1,3,5]:
            self.assert_archived(directory, records[i])
        self.assert_final(directory, 10)

    def test_complete_batch_no_sdk_or_key(self):
        directory, _, _ = self.fixture(n=1, successes=range(1,11), failures=[])
        before = self.snapshot(directory)
        result = self.invoke(directory, forbid_config=True, forbid_sdk=True)
        self.assertEqual(result['code'], 0, result['stderr'])
        self.assertEqual(result['config_calls'], 0)
        self.assertEqual(len(result['sdk'].calls), 0)
        self.assertEqual(self.snapshot(directory), before)

    def test_dryrun_is_readonly_and_no_sdk_or_key(self):
        directory, _, _ = self.fixture()
        before = self.snapshot(directory)
        result = self.invoke(directory, ['--dry-run'], forbid_config=True, forbid_sdk=True)
        self.assertEqual(result['code'], 0, result['stderr'])
        self.assertEqual(self.snapshot(directory), before)
        self.assertEqual(result['config_calls'], 0)
        self.assertEqual(len(result['sdk'].calls), 0)
        self.assertIn('55', result['stdout'])

    def test_resume_saved_prompt_model_b_n_limits(self):
        directory, config, _ = self.fixture(n=1, successes={1}, failures={2}, overrides={
            'model': 'gpt-5.5-saved', 'system_prompt': 'Saved original advisor prompt: C or D only.',
            'b': 1.875, 'max_output_tokens': 2048, 'timeout_seconds': 45,
        })
        with patch.object(self.module, 'B', 9.9), patch.object(self.module, 'N', 99), patch.object(self.module, 'MODEL', 'different-current-model'), patch.object(self.module, 'SYSTEM_PROMPT', 'current modified prompt'), patch.object(self.module, 'MAX_OUTPUT_TOKENS', 4096):
            result = self.invoke(directory)
        self.assertEqual(result['code'], 0, result['stderr'])
        self.assertEqual(len(result['sdk'].calls), 9)
        for client, request in result['sdk'].calls:
            self.assertEqual(request['model'], config['model'])
            self.assertEqual(request['instructions'], config['system_prompt'])
            self.assertIn('b = 1.875', request['input'])
            self.assertEqual(request['max_output_tokens'], 2048)
            self.assertEqual(client.options['timeout'], 45)
        self.assert_final(directory, 10)

    def test_saved_base_url_is_used(self):
        directory, config, _ = self.fixture(n=1, successes={1}, failures={2}, overrides={'base_url':'https://original.invalid/v1'})
        result = self.invoke(directory, configured_url=config['base_url'])
        self.assertEqual(result['code'], 0, result['stderr'])
        for client, _ in result['sdk'].calls:
            self.assertEqual(client.options['base_url'], config['base_url'])

    def test_changed_endpoint_rejected_without_network(self):
        directory, _, _ = self.fixture(n=1, successes={1}, failures={2}, overrides={'base_url':'https://original.invalid/v1'})
        before = self.snapshot(directory)
        result = self.invoke(directory)
        self.assertNotEqual(result['code'], 0)
        self.assertEqual(len(result['sdk'].calls), 0)
        self.assertEqual(self.snapshot(directory), before)

    def test_experiment_overrides_rejected(self):
        directory, _, _ = self.fixture()
        before = self.snapshot(directory)
        for option, value in [('--b','1.6'), ('--n','2'), ('--max-tokens','2048')]:
            with self.subTest(option=option), self.assertRaises(SystemExit) as raised:
                self.invoke(directory, [option,value], forbid_config=True)
            self.assertEqual(raised.exception.code, 2)
        self.assertEqual(self.snapshot(directory), before)

    def test_timeout_override_does_not_modify_original_config(self):
        directory, config, _ = self.fixture(n=1, successes={1}, failures={2})
        result = self.invoke(directory, ['--timeout','300'])
        self.assertEqual(result['code'], 0, result['stderr'])
        for client, _ in result['sdk'].calls:
            self.assertEqual(client.options['timeout'], 300)
        self.assertEqual(json.loads((directory/'config.json').read_text(encoding='utf-8')), config)

    def test_bad_record_json_fails_before_key_or_network(self):
        directory, _, _ = self.fixture()
        (directory/'records/000046.json').write_text('{broken')
        before = self.snapshot(directory)
        result = self.invoke(directory, forbid_config=True)
        self.assertNotEqual(result['code'], 0)
        self.assertEqual(len(result['sdk'].calls), 0)
        self.assertEqual(self.snapshot(directory), before)

    def test_bad_config_json_fails_before_key_or_network(self):
        directory, _, _ = self.fixture()
        (directory/'config.json').write_text('{broken')
        result = self.invoke(directory, forbid_config=True)
        self.assertNotEqual(result['code'], 0)
        self.assertEqual(len(result['sdk'].calls), 0)

    def test_wrong_record_index_fails_closed(self):
        directory, _, records = self.fixture()
        bad = records[46]
        bad['request_index'] = 47
        self.write(directory/'records/000046.json', bad)
        result = self.invoke(directory, forbid_config=True)
        self.assertNotEqual(result['code'], 0)
        self.assertEqual(len(result['sdk'].calls), 0)

    def test_prompt_mismatch_fails_closed(self):
        directory, _, records = self.fixture()
        bad = records[46]
        bad['request']['instructions'] = 'A prompt from a different batch'
        self.write(directory/'records/000046.json', bad)
        result = self.invoke(directory, forbid_config=True)
        self.assertNotEqual(result['code'], 0)
        self.assertEqual(len(result['sdk'].calls), 0)

    def test_corrupt_summary_does_not_lose_saved_success(self):
        directory, _, _ = self.fixture()
        (directory/'summary.json').write_text('{bad summary')
        result = self.invoke(directory)
        self.assertEqual(result['code'], 0, result['stderr'])
        self.assertEqual(len(result['sdk'].calls), 55)
        self.assert_final(directory, 100)

    def test_write_failure_before_request_prevents_charge(self):
        directory, _, _ = self.fixture()
        with patch.object(self.module, 'save_record', side_effect=OSError('fake disk full')):
            result = self.invoke(directory)
        self.assertNotEqual(result['code'], 0)
        self.assertEqual(len(result['sdk'].calls), 0)

    def test_continue_on_error_then_resume_only_failure(self):
        directory, _, _ = self.fixture(n=1, successes={1,2}, failures={3})
        result = self.invoke(directory, ['--continue-on-error'], effects=[FakeConnection('offline failure')])
        self.assertEqual(result['code'], 1)
        self.assertEqual(len(result['sdk'].calls), 8)
        result2 = self.invoke(directory)
        self.assertEqual(result2['code'], 0)
        self.assertEqual(len(result2['sdk'].calls), 1)
        self.assert_final(directory, 10)

    def test_fresh_batch_still_produces_ten_independent_calls(self):
        sdk = FakeSDK()
        with patch.dict(sys.modules, {'openai': sdk}), patch.object(self.module, 'load_config', return_value=(KEY, URL)), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = self.module.main(['--n', '1', '--b', '1.25'])
        self.assertEqual(code, 0)
        self.assertEqual(len(sdk.calls), 10)
        self.assertEqual(len(sdk.clients), 10)
        self.assertTrue(all(c.options['max_retries'] == 0 for c in sdk.clients))
        for _, request in sdk.calls:
            self.assertNotIn('previous_response_id', request)
            self.assertNotIn('conversation', request)
            self.assertIn('b = 1.25', request['input'])
        directory = next((self.root/'new_logs').iterdir())
        self.assert_final(directory, 10)

    def test_default_unknown_b_requests_cover_all_cases_and_are_logged(self):
        self.assertIsNone(self.module.B)
        custom_system = self.module.SYSTEM_PROMPT + '\n\nb is an unknown parameter satisfying 1 < b < 2.\nDo not assume a specific value or a probability distribution for b.'
        with patch.object(self.module, 'SYSTEM_PROMPT', custom_system):
            result = self.invoke_new()
        self.assertEqual(result['code'], 0, result['stderr'])
        self.assertEqual(len(result['sdk'].calls), 10)
        self.assertEqual(len(result['sdk'].clients), 10)
        directory = next((self.root/'new_logs').iterdir())
        config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
        self.assertIsNone(config['b'])
        self.assertEqual(config['b_mode'], 'unknown_interval')
        self.assertEqual(config['b_constraint'], '1 < b < 2')
        self.assertEqual(
            {(c['current_action'], c['cooperating_neighbors'], c['defecting_neighbors']) for c in config['cases']},
            {(action, cooperating, 4-cooperating) for action in ('C', 'D') for cooperating in range(5)},
        )
        for index, (client, request) in enumerate(result['sdk'].calls, 1):
            self.assertNotIn('previous_response_id', request)
            self.assertNotIn('conversation', request)
            self.assertTrue(request['input'].startswith('Current observation of the focal player:'))
            self.assertNotIn('1 < b < 2', request['input'])
            self.assertNotIn('unknown', request['input'].lower())
            self.assertNotIn('Do not assume a specific value or a probability distribution for b.', request['input'])
            self.assertNotIn('b = None', request['input'])
            self.assertNotIn('b = 1.5', request['input'])
            self.assertEqual(request['instructions'], custom_system)
            self.assertEqual(request, config['requests'][index-1])
            self.assertEqual(client.options['max_retries'], 0)
            record = json.loads((directory/f'records/{index:06d}.json').read_text(encoding='utf-8'))
            self.assertIsNone(record['observation']['b'])
            self.assertEqual(record['b_mode'], 'unknown_interval')
            self.assertEqual(record['b_constraint'], '1 < b < 2')
            self.assertEqual(record['request'], request)
        summary = self.assert_final(directory, 10)
        self.assertIsNone(summary['b'])
        self.assertEqual(summary['b_mode'], 'unknown_interval')
        self.assertEqual(summary['b_constraint'], '1 < b < 2')
        rows = list(csv.DictReader(io.StringIO((directory/'results.csv').read_text(encoding='utf-8-sig'))))
        self.assertEqual({row['b'] for row in rows}, {'unknown'})
        self.assertEqual({row['b_mode'] for row in rows}, {'unknown_interval'})
        self.assertEqual({row['b_constraint'] for row in rows}, {'1 < b < 2'})
        self.assertIn('1 < b < 2', result['stdout'])
        self.assertNotIn('b=None', result['stdout'])

    def test_unknown_b_failure_resumes_saved_requests_and_preserves_success(self):
        first = self.invoke_new(effects=[FakeResponse('C'), FakeConnection('offline failure')])
        self.assertEqual(first['code'], 1, first['stderr'])
        self.assertEqual(len(first['sdk'].calls), 2)
        directory = next((self.root/'new_logs').iterdir())
        original_config = (directory/'config.json').read_bytes()
        original_success = (directory/'records/000001.json').read_bytes()
        config = json.loads(original_config)
        failed = json.loads((directory/'records/000002.json').read_text(encoding='utf-8'))
        with patch.object(self.module, 'B', 1.75), patch.object(self.module, 'build_user_prompt', return_value='Changed current template'):
            resumed = self.invoke(directory)
        self.assertEqual(resumed['code'], 0, resumed['stderr'])
        self.assertEqual(len(resumed['sdk'].calls), 9)
        self.assertEqual((directory/'config.json').read_bytes(), original_config)
        self.assertEqual((directory/'records/000001.json').read_bytes(), original_success)
        for index, (_, request) in enumerate(resumed['sdk'].calls, 1):
            self.assertEqual(request, config['requests'][index])
            self.assertTrue(request['input'].startswith('Current observation of the focal player:'))
            self.assertNotIn('1 < b < 2', request['input'])
            self.assertNotIn('b = 1.75', request['input'])
        self.assert_archived(directory, failed)
        summary = self.assert_final(directory, 10)
        self.assertIsNone(summary['b'])
        self.assertEqual(summary['b_mode'], 'unknown_interval')
        self.assertEqual(summary['b_constraint'], '1 < b < 2')
        self.assertEqual(summary['api_calls_started_total'], 11)
        self.assertIn('1 < b < 2', resumed['stdout'])
        self.assertNotIn('b=None', resumed['stdout'])

    def test_default_unknown_b_dryrun_needs_no_sdk_key_or_files(self):
        result = self.invoke_new(['--dry-run'], forbid_config=True, forbid_sdk=True)
        self.assertEqual(result['code'], 0, result['stderr'])
        self.assertEqual(result['config_calls'], 0)
        self.assertEqual(len(result['sdk'].calls), 0)
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertIn('1 < b < 2', result['stdout'])
        self.assertNotIn('b=None', result['stdout'])

    def test_null_b_without_unknown_condition_is_rejected_before_api(self):
        directory, _, _ = self.fixture(n=1, successes={1}, failures={2}, overrides={'b': None})
        before = self.snapshot(directory)
        result = self.invoke(directory, forbid_config=True, forbid_sdk=True)
        self.assertNotEqual(result['code'], 0)
        self.assertEqual(len(result['sdk'].calls), 0)
        self.assertEqual(self.snapshot(directory), before)

    def test_resume_by_batch_name(self):
        directory, _, _ = self.fixture(n=1, successes={1}, failures={2})
        with patch.object(self.module, 'LOG_DIR', directory.parent):
            result = self.invoke(Path(directory.name))
        self.assertEqual(result['code'], 0, result['stderr'])
        self.assertEqual(len(result['sdk'].calls), 9)
        self.assert_final(directory, 10)

    def test_existing_lock_blocks_resume_without_key_or_writes(self):
        directory, _, _ = self.fixture()
        (directory/'.run.lock').write_text('another active process')
        before = self.snapshot(directory)
        result = self.invoke(directory, forbid_config=True, forbid_sdk=True)
        self.assertNotEqual(result['code'], 0)
        self.assertEqual(len(result['sdk'].calls), 0)
        self.assertEqual(self.snapshot(directory), before)

    def test_archive_move_then_record_write_failure_is_recoverable(self):
        directory, _, records = self.fixture()
        original_save = self.module.save_record
        failed_bytes = (directory/'records/000046.json').read_bytes()
        def fail_candidate(path, record, key=''):
            if path.name == '000046.json' and record.get('status') == 'running':
                raise OSError('fake disk full after archive move')
            return original_save(path, record, key)
        with patch.object(self.module, 'save_record', side_effect=fail_candidate):
            result = self.invoke(directory)
        self.assertNotEqual(result['code'], 0)
        self.assertEqual(len(result['sdk'].calls), 0)
        archived = list((directory/'attempts/000046').glob('*.json'))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].read_bytes(), failed_bytes)
        self.assertFalse((directory/'.run.lock').exists())
        resumed = self.invoke(directory)
        self.assertEqual(resumed['code'], 0, resumed['stderr'])
        self.assertEqual(len(resumed['sdk'].calls), 55)
        summary = self.assert_final(directory, 100)
        self.assertEqual(summary['archived_attempts'], 1)
        self.assertEqual(summary['api_calls_started_total'], 101)

    def test_complete_saved_templates_ignore_changed_current_user_builder(self):
        directory, config, _ = self.fixture(n=2, successes=range(1,11), failures={11})
        saved_inputs = {i:self.module.build_user_prompt(c) for i,c in enumerate(config['cases'])}
        with patch.object(self.module, 'build_user_prompt', return_value='Changed current user template'):
            result = self.invoke(directory)
        self.assertEqual(result['code'], 0, result['stderr'])
        self.assertEqual(len(result['sdk'].calls), 10)
        for i, (_, request) in enumerate(result['sdk'].calls):
            self.assertEqual(request['input'], saved_inputs[i])
        self.assertFalse((directory/'.run.lock').exists())

    def test_successful_response_save_failure_is_recovered_without_repeating_api(self):
        directory, _, _ = self.fixture(n=1, successes={1}, failures={2})
        original_save = self.module.save_record
        failed = False

        def fail_once(path, record, key=''):
            nonlocal failed
            if path.name == '000002.json' and record.get('status') == 'success' and not failed:
                failed = True
                raise OSError('temporary write failure after successful API response')
            return original_save(path, record, key)

        with patch.object(self.module, 'save_record', side_effect=fail_once):
            first = self.invoke(directory)
        self.assertTrue(failed)
        self.assertEqual(first['code'], 1)
        self.assertEqual(len(first['sdk'].calls), 1)
        saved = json.loads((directory/'records/000002.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['status'], 'success')
        self.assertEqual(saved['response']['status'], 'completed')
        before = (directory/'records/000002.json').read_bytes()
        second = self.invoke(directory)
        self.assertEqual(second['code'], 0, second['stderr'])
        self.assertEqual(len(second['sdk'].calls), 8)
        self.assertEqual((directory/'records/000002.json').read_bytes(), before)
        self.assert_final(directory, 10)

    def test_final_csv_failure_is_saved_as_logging_error_in_summary(self):
        directory, _, _ = self.fixture(n=1, successes=range(1,10), failures={10})
        original_write = self.module.write_results
        writes = 0

        def fail_final(*args, **kwargs):
            nonlocal writes
            writes += 1
            if writes == 3:
                raise OSError('simulated final CSV replacement failure')
            return original_write(*args, **kwargs)

        with patch.object(self.module, 'write_results', side_effect=fail_final):
            result = self.invoke(directory)
        self.assertEqual(writes, 3)
        self.assertEqual(result['code'], 1)
        self.assertEqual(len(result['sdk'].calls), 1)
        summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['status'], 'logging_error')
        self.assertEqual(summary['successful_requests'], 10)
        self.assertEqual(summary['pending_requests'], 0)
        self.assertIn('批次状态=logging_error', result['stdout'])
        self.assertFalse((directory/'.run.lock').exists())

    def test_archive_only_template_can_recover_legacy_batch(self):
        directory, _, _ = self.fixture(n=1, successes=[], failures={1})
        archive = directory/'attempts/000001/old-attempt.json'
        archive.parent.mkdir(parents=True)
        (directory/'records/000001.json').replace(archive)
        before = archive.read_bytes()
        result = self.invoke(directory)
        self.assertEqual(result['code'], 0, result['stderr'])
        self.assertEqual(len(result['sdk'].calls), 10)
        self.assertEqual(archive.read_bytes(), before)
        summary = self.assert_final(directory, 10)
        self.assertEqual(summary['api_calls_started_total'], 11)
        self.assertEqual(summary['archived_attempts'], 1)

    def test_safety_guards_are_active_under_discovery(self):
        with self.assertRaisesRegex(AssertionError, 'real credentials'):
            (PROJECT/'API/gpt.env').read_text(encoding='utf-8')
        with self.assertRaisesRegex(AssertionError, 'real credentials'):
            builtins.open(PROJECT/'API/gpt.env')
        with self.assertRaisesRegex(AssertionError, 'Network forbidden'):
            socket.create_connection(('example.invalid', 443))

    def test_interrupt_then_resume(self):
        directory, _, _ = self.fixture(n=1, successes={1}, failures={2})
        result = self.invoke(directory, effects=[FakeResponse('C'), KeyboardInterrupt()])
        self.assertEqual(result['code'], 130)
        self.assertEqual(len(result['sdk'].calls), 2)
        record = json.loads((directory/'records/000003.json').read_text(encoding='utf-8'))
        self.assertEqual(record['status'], 'cancelled')
        second = self.invoke(directory)
        self.assertEqual(second['code'], 0)
        self.assertEqual(len(second['sdk'].calls), 8)
        self.assert_archived(directory, record)
        self.assert_final(directory, 10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
