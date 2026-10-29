"""DeepSeek batch checks with fake clients, temporary logs and blocked network access."""

import builtins
import contextlib
import copy
import csv
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
KEY = "sk-offline-deepseek-batch-test"
URL = "https://deepseek.example.invalid/v1"
MODEL = "deepseek-v4.1-flash"


def guard_credentials(original):
    def guarded(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            if Path(os.fsdecode(file)).resolve().is_relative_to(PROJECT / "API"):
                raise AssertionError("Offline test attempted to read real credentials")
        return original(file, *args, **kwargs)
    return guarded


class FakeAPIError(Exception):
    pass


class FakeConnectionError(FakeAPIError):
    pass


class FakeTimeoutError(FakeConnectionError):
    pass


class FakeStatusError(FakeAPIError):
    pass


class FakeResponse:
    def __init__(self, index=1, action="C", finish_reason="stop"):
        self.id = f"chatcmpl_offline_deepseek_{index}"
        self.model = MODEL
        self.choices = [types.SimpleNamespace(
            index=0, finish_reason=finish_reason,
            message=types.SimpleNamespace(role="assistant", content=action,
                                          reasoning_content="Offline DeepSeek reasoning."),
        )]

    def model_dump(self, mode="json"):
        choice = self.choices[0]
        return {
            "id": self.id, "model": self.model, "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": choice.finish_reason,
                         "message": vars(choice.message).copy()}],
            "usage": {"prompt_tokens": 173, "completion_tokens": 11, "total_tokens": 184},
        }


class FakeSDK(types.ModuleType):
    def __init__(self, effects=(), before_call=None):
        super().__init__("openai")
        self.calls, self.clients, self.http_clients = [], [], []
        self.APIError = FakeAPIError
        self.APIConnectionError = FakeConnectionError
        self.APITimeoutError = FakeTimeoutError
        self.APIStatusError = FakeStatusError
        sdk = self

        class HttpClient:
            def __init__(self, **kwargs):
                self.options, self.closed = kwargs, False
                sdk.http_clients.append(self)

            def close(self):
                self.closed = True

        class Client:
            def __init__(self, **kwargs):
                self.options, self.closed, self.entered = kwargs, False, False
                self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))
                self.responses = types.SimpleNamespace(create=self.forbidden_responses)
                sdk.clients.append(self)

            @staticmethod
            def forbidden_responses(**kwargs):
                raise AssertionError("DeepSeek must use chat.completions")

            def __enter__(self):
                self.entered = True
                return self

            def __exit__(self, *args):
                self.close()
                return False

            def close(self):
                self.closed = True
                self.options["http_client"].close()

            def create(self, **kwargs):
                if any(not client.closed for client in sdk.clients[:-1]):
                    raise AssertionError("Previous request client remained open")
                if any(not client.closed for client in sdk.http_clients[:-1]):
                    raise AssertionError("Previous request transport remained open")
                sdk.calls.append(copy.deepcopy(kwargs))
                index = len(sdk.calls)
                if before_call is not None:
                    before_call(index, kwargs)
                effect = effects[index - 1] if index <= len(effects) else FakeResponse(index)
                if isinstance(effect, BaseException):
                    raise effect
                return effect

        self.DefaultHttpxClient, self.OpenAI = HttpClient, Client


class DeepSeekBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        guards = contextlib.ExitStack()
        cls.addClassCleanup(guards.close)
        guards.enter_context(patch("builtins.open", side_effect=guard_credentials(builtins.open)))
        guards.enter_context(patch("io.open", side_effect=guard_credentials(io.open)))
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            guards.enter_context(patch(target, side_effect=AssertionError("Network forbidden in DeepSeek tests")))
        guards.enter_context(patch.object(sys, "dont_write_bytecode", True))
        previous_path = sys.path[:]
        cls.addClassCleanup(lambda: sys.path.__setitem__(slice(None), previous_path))
        sys.path.insert(0, str(PROJECT / "main"))
        spec = importlib.util.spec_from_file_location("tested_deepseek_batch", PROJECT / "main/GPT_Multisteps.py")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="offline_deepseek_batch_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.logs = self.root / "logs"
        for name, value in (("LOG_DIR", self.logs), ("TRUST_ENV", True)):
            setting = patch.object(self.module, name, value)
            setting.start()
            self.addCleanup(setting.stop)

    def invoke(self, args, *, effects=(), before_call=None, forbidden=False, base_url=URL):
        sdk = FakeSDK(effects, before_call)
        stdout, stderr = io.StringIO(), io.StringIO()
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if forbidden and (name == "openai" or name.startswith("openai.")):
                raise AssertionError("This path must not import the SDK")
            return original_import(name, *args, **kwargs)

        credential_options = ({"side_effect": AssertionError("This path must not load credentials")}
                              if forbidden else {"return_value": (KEY, base_url)})
        with patch.dict(sys.modules, {"openai": sdk}), \
                patch.object(self.module, "load_config", side_effect=AssertionError("GPT credentials forbidden")), \
                patch.object(self.module, "load_qwen_config", side_effect=AssertionError("Qwen credentials forbidden")), \
                patch.object(self.module, "load_deepseek_config", **credential_options) as loader, \
                patch("builtins.__import__", side_effect=guarded_import), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = self.module.main(args)
        return {"code": code, "sdk": sdk, "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(), "credential_calls": loader.call_count}

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def snapshot(directory):
        return {str(path.relative_to(directory)): path.read_bytes()
                for path in directory.rglob("*") if path.is_file()}

    def fixture(self, *, records=True):
        cases = self.module.build_cases(None, "grid_3x3")
        config = {
            "batch_id": "offline-deepseek-resume", "model": MODEL,
            "provider": "deepseek", "api_interface": "chat.completions",
            "env_file": "API/Deepseek4.1flash", "base_url": URL,
            "b": None, "n": 1, **self.module.b_metadata(None),
            "observation_format": "grid_3x3", "case_count": 512,
            "planned_requests": 512, "cases": cases,
            "system_prompt": self.module.SYSTEM_PROMPT,
            "requests": [{"model": MODEL, "max_tokens": 1600, "stream": False,
                          "messages": [{"role": "system", "content": self.module.SYSTEM_PROMPT},
                                       {"role": "user", "content": self.module.build_user_prompt(case)}]}
                         for case in cases],
            "max_output_tokens": 1600, "timeout_seconds": 120,
            "trust_env": True, "max_retries": 0, "continue_on_error": False,
        }
        directory = self.root / "resume"
        (directory / "records").mkdir(parents=True)
        self.write_json(directory / "config.json", config)
        if records:
            for index in range(1, 512):
                record = {
                    "batch_id": config["batch_id"], "request_index": index,
                    "provider": "deepseek", "api_interface": "chat.completions", "model": MODEL,
                    "repetition": 1, "combination_index": index,
                    "observation": cases[index - 1], "base_url": URL,
                    "observation_format": "grid_3x3", **self.module.b_metadata(None),
                    "client_options": {"timeout_seconds": 120, "max_retries": 0, "trust_env": True},
                    "request": config["requests"][index - 1], "api_call_started": True,
                    "started_at": "2026-10-29T10:00:00.000+08:00",
                    "finished_at": "2026-10-29T10:00:01.000+08:00", "duration_seconds": 1.0,
                    "status": "success", "action": "C", "output_text": "C",
                    "response": FakeResponse(index).model_dump(), "error": None,
                }
                if index == 511:
                    record.update(status="error", action=None, output_text=None, response=None,
                                  error={"category": "connection", "message": "offline connection failure"})
                self.write_json(directory / "records" / f"{index:06d}.json", record)
        return directory, config

    def assert_independent(self, request):
        self.assertEqual(set(request), {"model", "messages", "max_tokens", "stream"})
        self.assertEqual(request["model"], MODEL)
        self.assertIs(request["stream"], False)
        self.assertEqual([message["role"] for message in request["messages"]], ["system", "user"])
        self.assertTrue(all(set(message) == {"role", "content"} and isinstance(message["content"], str)
                            for message in request["messages"]))

    def assert_clients_closed(self, sdk, *, trust_env=True):
        self.assertEqual(len(sdk.clients), len(sdk.calls))
        self.assertEqual(len(sdk.http_clients), len(sdk.calls))
        self.assertTrue(all(client.entered and client.closed for client in sdk.clients))
        self.assertTrue(all(client.closed for client in sdk.http_clients))
        self.assertEqual(len({id(client) for client in sdk.clients}), len(sdk.calls))
        self.assertEqual(len({id(client) for client in sdk.http_clients}), len(sdk.calls))
        for client in sdk.clients:
            self.assertEqual(client.options["api_key"], KEY)
            self.assertEqual(client.options["base_url"], URL)
            self.assertEqual(client.options["max_retries"], 0)
            self.assertIs(client.options["http_client"].options["trust_env"], trust_env)
            self.assertIs(client.options["http_client"].options["verify"], True)

    def test_preview_cli_and_file_default_need_no_credentials_sdk_or_logs(self):
        for args, default in ((["--provider", "deepseek", "--n", "2", "--dry-run"], "qwen"),
                              (["--n", "2", "--dry-run"], "deepseek")):
            with self.subTest(default=default), patch.object(self.module, "PROVIDER", default):
                result = self.invoke(args, forbidden=True)
                self.assertEqual(result["code"], 0, result["stderr"])
                self.assertIn("服务=deepseek", result["stdout"])
                self.assertIn(MODEL, result["stdout"])
                self.assertIn("1024", result["stdout"])
                self.assertIn("网格=DDD/DDD/DDD", result["stdout"])
                self.assertNotIn("Qwen 思考强度", result["stdout"])
                self.assertEqual(result["credential_calls"], 0)
                self.assertFalse(result["sdk"].calls)
                self.assertFalse(self.logs.exists())

    def test_full_batch_512_fresh_clients_and_raw_chat_usage_are_saved(self):
        def before_call(index, request):
            directory = next(self.logs.iterdir())
            record = json.loads((directory / "records" / f"{index:06d}.json").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "running")
            self.assertEqual(record["provider"], "deepseek")
            self.assertEqual(record["request"], request)

        with patch.object(self.module, "PROVIDER", "qwen"):
            result = self.invoke(["--provider", "deepseek", "--n", "1"], before_call=before_call)
        self.assertEqual(result["code"], 0, result["stderr"])
        self.assertEqual(result["credential_calls"], 1)
        sdk = result["sdk"]
        self.assertEqual(len(sdk.calls), 512)
        self.assert_clients_closed(sdk)
        for request in sdk.calls:
            self.assert_independent(request)
        self.assertEqual(len({request["messages"][1]["content"] for request in sdk.calls}), 512)
        directory = next(self.logs.iterdir())
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        for metadata in (config, summary):
            self.assertEqual(metadata["provider"], "deepseek")
            self.assertEqual(metadata["model"], MODEL)
            self.assertEqual(metadata["api_interface"], "chat.completions")
            self.assertEqual(metadata["env_file"], "API/Deepseek4.1flash")
        self.assertEqual(config["requests"], sdk.calls)
        self.assertIs(config["trust_env"], True)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["successful_requests"], 512)
        rows = list(csv.DictReader(io.StringIO((directory / "results.csv").read_text(encoding="utf-8-sig"))))
        self.assertEqual(len(rows), 512)
        for row, case in zip(rows, config["cases"]):
            self.assertEqual(row["provider"], "deepseek")
            self.assertEqual(row["api_interface"], "chat.completions")
            self.assertEqual(row["model"], MODEL)
            self.assertEqual(row["qwen_reasoning_effort"], "")
            self.assertEqual(row["grid_key"], case["grid_key"])
            self.assertEqual(row["recommendation"], "C")
            self.assertEqual((row["input_tokens"], row["output_tokens"], row["total_tokens"]), ("173", "11", "184"))
        saved = json.loads((directory / "records/000001.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["response"]["choices"][0]["message"]["reasoning_content"],
                         "Offline DeepSeek reasoning.")

    def test_errors_and_incomplete_responses_are_saved_without_success(self):
        for effect in (FakeTimeoutError("offline timeout"), FakeConnectionError("connection refused"),
                       FakeResponse(finish_reason="length"), FakeResponse(action="Choose C")):
            with self.subTest(effect=type(effect).__name__):
                result = self.invoke(["--provider", "deepseek", "--n", "1", "--tls-retries", "0"],
                                     effects=[effect])
                self.assertEqual(result["code"], 1)
                self.assertEqual(len(result["sdk"].calls), 1)
                self.assert_clients_closed(result["sdk"])
                self.assert_independent(result["sdk"].calls[0])
                directory = max(self.logs.iterdir(), key=lambda path: path.name)
                saved = json.loads((directory / "records/000001.json").read_text(encoding="utf-8"))
                summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["status"], "error")
                self.assertIsNone(saved["action"])
                self.assertEqual(summary["successful_requests"], 0)
                self.assertEqual(summary["failed_requests"], 1)
                if isinstance(effect, FakeResponse):
                    self.assertEqual(saved["response"], effect.model_dump())

    def test_tls_retry_keeps_request_independent_and_closes_every_client(self):
        result = self.invoke(
            ["--provider", "deepseek", "--n", "1", "--tls-retries", "1", "--retry-delay", "0", "--no-trust-env"],
            effects=[FakeConnectionError("TLS/SSL connection has been closed (EOF)"),
                     FakeResponse(2), FakeConnectionError("connection refused")],
        )
        self.assertEqual(result["code"], 1)
        sdk = result["sdk"]
        self.assertEqual(len(sdk.calls), 3)
        self.assertEqual(sdk.calls[0], sdk.calls[1])
        self.assertNotEqual(sdk.calls[1], sdk.calls[2])
        self.assert_clients_closed(sdk, trust_env=False)
        for request in sdk.calls:
            self.assert_independent(request)
        directory = next(self.logs.iterdir())
        self.assertEqual(len(list((directory / "attempts/000001").glob("*.json"))), 1)

    def test_resume_preserves_original_requests_and_successes_archives_failed_attempt(self):
        directory, config = self.fixture()
        originals = self.snapshot(directory)
        with patch.object(self.module, "PROVIDER", "qwen"), \
                patch.object(self.module, "DEEPSEEK_MODEL", "different-model"), \
                patch.object(self.module, "SYSTEM_PROMPT", "Changed after batch was created"), \
                patch.object(self.module, "MAX_OUTPUT_TOKENS", 16), \
                patch.object(self.module, "build_user_prompt", return_value="Changed formatter"):
            result = self.invoke(["--resume", str(directory)])
        self.assertEqual(result["code"], 0, result["stderr"])
        self.assertEqual(result["credential_calls"], 1)
        self.assertEqual(result["sdk"].calls, config["requests"][510:])
        self.assert_clients_closed(result["sdk"])
        for relative, contents in originals.items():
            if relative != "records/000511.json":
                self.assertEqual((directory / relative).read_bytes(), contents)
        archived = list((directory / "attempts/000511").glob("*.json"))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].read_bytes(), originals["records/000511.json"])
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["successful_requests"], 512)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["model"], MODEL)

    def test_resume_preview_has_no_credentials_calls_or_file_mutations(self):
        directory, _ = self.fixture()
        originals = self.snapshot(directory)
        result = self.invoke(["--resume", str(directory), "--dry-run"], forbidden=True)
        self.assertEqual(result["code"], 0, result["stderr"])
        self.assertIn("511, 512", result["stdout"])
        self.assertEqual(result["credential_calls"], 0)
        self.assertFalse(result["sdk"].calls)
        self.assertEqual(self.snapshot(directory), originals)

    def test_provider_switch_and_endpoint_change_stop_before_any_model_call(self):
        directory, _ = self.fixture(records=False)
        originals = self.snapshot(directory)
        for provider in ("gpt", "qwen"):
            with self.subTest(provider=provider):
                result = self.invoke(["--resume", str(directory), "--provider", provider], forbidden=True)
                self.assertEqual(result["code"], 1)
                self.assertFalse(result["sdk"].calls)
                self.assertEqual(self.snapshot(directory), originals)
        result = self.invoke(["--resume", str(directory)], base_url="https://different.example.invalid/v1")
        self.assertEqual(result["code"], 1)
        self.assertFalse(result["sdk"].calls)
        self.assertEqual(self.snapshot(directory), originals)

    def test_resume_rejects_history_qwen_options_and_inconsistent_metadata(self):
        directory, original = self.fixture(records=False)
        mutations = [
            lambda config: config["requests"][0]["messages"].append({"role": "assistant", "content": "C"}),
            lambda config: config["requests"][0].update(previous_response_id="resp_previous"),
            lambda config: config["requests"][0].update(extra_body={"reasoning_effort": "medium"}),
            lambda config: config["requests"][0].update(model="gpt-5.5"),
            lambda config: config["requests"][0].update(stream=True),
            lambda config: config.update(api_interface="responses"),
            lambda config: config.update(provider="qwen"),
            lambda config: config.update(qwen_reasoning_effort="medium"),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                config = copy.deepcopy(original)
                mutate(config)
                self.write_json(directory / "config.json", config)
                before = self.snapshot(directory)
                result = self.invoke(["--resume", str(directory)], forbidden=True)
                self.assertEqual(result["code"], 1)
                self.assertFalse(result["sdk"].calls)
                self.assertEqual(self.snapshot(directory), before)

    def test_resume_rejects_inconsistent_success_in_original_chat_response(self):
        directory, _ = self.fixture()
        path = directory / "records/000001.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        for change in ("content", "finish_reason"):
            with self.subTest(change=change):
                record = copy.deepcopy(original)
                choice = record["response"]["choices"][0]
                if change == "content":
                    choice["message"]["content"] = "D"
                else:
                    choice["finish_reason"] = "length"
                self.write_json(path, record)
                before = self.snapshot(directory)
                result = self.invoke(["--resume", str(directory)], forbidden=True)
                self.assertEqual(result["code"], 1)
                self.assertFalse(result["sdk"].calls)
                self.assertEqual(self.snapshot(directory), before)

    def test_qwen_only_cli_options_are_rejected_before_credentials_or_logs(self):
        with self.assertRaises(SystemExit) as stopped:
            self.invoke(["--provider", "deepseek", "--qwen-reasoning-effort", "medium"], forbidden=True)
        self.assertEqual(stopped.exception.code, 2)
        self.assertFalse(self.logs.exists())


if __name__ == "__main__":
    unittest.main()
