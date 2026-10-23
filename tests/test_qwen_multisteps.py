"""Qwen batch integration checks using fake SDK clients and temporary logs only."""

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
KEY = "sk-offline-qwen-batch-tests"
URL = "http://qwen.example.invalid:8000/v1"
QWEN_MODEL = "/home/tianyuxuan/qwen/Qwen3.8-27B-f16.gguf"


def guard_credentials(original):
    def guarded(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            candidate = Path(os.fsdecode(file)).resolve()
            if candidate.is_relative_to(PROJECT / "API"):
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


class FakeChatResponse:
    def __init__(self, index=1, action="D", finish_reason="stop"):
        self.id = f"chatcmpl_offline_qwen_{index}"
        self.model = QWEN_MODEL
        self.choices = [types.SimpleNamespace(
            index=0,
            finish_reason=finish_reason,
            message=types.SimpleNamespace(
                role="assistant", content=action,
                reasoning_content="Offline reasoning retained in the raw record.",
            ),
        )]

    def model_dump(self, mode="json"):
        choice = self.choices[0]
        return {
            "id": self.id, "model": self.model, "object": "chat.completion",
            "choices": [{
                "index": 0, "finish_reason": choice.finish_reason,
                "message": vars(choice.message).copy(),
            }],
            "usage": {"prompt_tokens": 123, "completion_tokens": 7, "total_tokens": 130},
        }


class FakeSDK(types.ModuleType):
    def __init__(self, effects=(), before_call=None, construction_error=None):
        super().__init__("openai")
        self.effects = list(effects)
        self.before_call = before_call
        self.construction_error = construction_error
        self.calls = []
        self.clients = []
        self.http_clients = []
        self.APIError = FakeAPIError
        self.APIConnectionError = FakeConnectionError
        self.APITimeoutError = FakeTimeoutError
        self.APIStatusError = FakeStatusError
        sdk = self

        class HttpClient:
            def __init__(self, **kwargs):
                self.options = kwargs
                self.closed = False
                sdk.http_clients.append(self)

            def close(self):
                self.closed = True

        class Client:
            def __init__(self, **kwargs):
                self.options = kwargs
                self.entered = False
                self.closed = False
                if sdk.construction_error is not None:
                    raise sdk.construction_error
                self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))
                self.responses = types.SimpleNamespace(create=self.forbidden_responses)
                sdk.clients.append(self)

            @staticmethod
            def forbidden_responses(**kwargs):
                raise AssertionError("Qwen requests must use chat.completions")

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
                sdk.calls.append(copy.deepcopy(kwargs))
                index = len(sdk.calls)
                if sdk.before_call is not None:
                    sdk.before_call(index, kwargs)
                response = sdk.effects[index - 1] if index <= len(sdk.effects) else FakeChatResponse(index)
                if isinstance(response, BaseException):
                    raise response
                return response

        self.DefaultHttpxClient = HttpClient
        self.OpenAI = Client


class QwenBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        guards = contextlib.ExitStack()
        cls.addClassCleanup(guards.close)
        guards.enter_context(patch("builtins.open", side_effect=guard_credentials(builtins.open)))
        guards.enter_context(patch("io.open", side_effect=guard_credentials(io.open)))
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            guards.enter_context(patch(target, side_effect=AssertionError("Network forbidden in Qwen tests")))
        guards.enter_context(patch.object(sys, "dont_write_bytecode", True))
        previous_path = sys.path[:]
        cls.addClassCleanup(lambda: sys.path.__setitem__(slice(None), previous_path))
        sys.path.insert(0, str(PROJECT / "main"))
        spec = importlib.util.spec_from_file_location("tested_qwen_batch", PROJECT / "main/GPT_Multisteps.py")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="offline_qwen_batch_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.logs = self.root / "logs"
        log_patch = patch.object(self.module, "LOG_DIR", self.logs)
        log_patch.start()
        self.addCleanup(log_patch.stop)

    def invoke(self, args, *, effects=(), before_call=None, forbidden=False, construction_error=None):
        sdk = FakeSDK(effects, before_call, construction_error)
        stdout, stderr = io.StringIO(), io.StringIO()
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if forbidden and (name == "openai" or name.startswith("openai.")):
                raise AssertionError("This path must not import the model SDK")
            return original_import(name, *args, **kwargs)

        qwen_options = {"side_effect": AssertionError("This path must not load credentials")} if forbidden else {
            "return_value": (KEY, URL)
        }
        with patch.dict(sys.modules, {"openai": sdk}), \
                patch.object(self.module, "load_config", side_effect=AssertionError("GPT config must not be read")) as gpt_config, \
                patch.object(self.module, "load_qwen_config", **qwen_options) as qwen_config, \
                patch("builtins.__import__", side_effect=guarded_import), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = self.module.main(args)
        return {
            "code": code, "sdk": sdk, "stdout": stdout.getvalue(), "stderr": stderr.getvalue(),
            "gpt_config_calls": gpt_config.call_count, "qwen_config_calls": qwen_config.call_count,
        }

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
            "batch_id": "offline-qwen-resume", "model": QWEN_MODEL,
            "provider": "qwen", "api_interface": "chat.completions",
            "base_url": URL, "b": None, "n": 1, **self.module.b_metadata(None),
            "observation_format": "grid_3x3", "case_count": 512,
            "planned_requests": 512, "cases": cases,
            "system_prompt": self.module.SYSTEM_PROMPT,
            "requests": [self.module.make_request(case, 1500, provider="qwen") for case in cases],
            "qwen_reasoning_effort": self.module.QWEN_REASONING_EFFORT,
            "max_output_tokens": 1500, "timeout_seconds": 120,
            "trust_env": False, "max_retries": 0, "cache_prompt": False,
            "continue_on_error": False,
        }
        directory = self.root / "resume"
        (directory / "records").mkdir(parents=True)
        self.write_json(directory / "config.json", config)
        if records:
            for index in range(1, 512):
                response = FakeChatResponse(index).model_dump()
                record = {
                    "batch_id": config["batch_id"], "request_index": index,
                    "provider": "qwen", "api_interface": "chat.completions", "model": QWEN_MODEL,
                    "repetition": 1, "combination_index": index,
                    "observation": cases[index - 1], "base_url": URL,
                    "observation_format": "grid_3x3", **self.module.b_metadata(None),
                    "client_options": {"timeout_seconds": 120, "max_retries": 0, "trust_env": False},
                    "request": config["requests"][index - 1], "api_call_started": True,
                    "started_at": "2026-10-23T10:00:00.000+08:00",
                    "finished_at": "2026-10-23T10:00:01.000+08:00", "duration_seconds": 1.0,
                    "status": "success", "action": "D", "output_text": "D",
                    "response": response, "error": None,
                }
                self.write_json(directory / "records" / f"{index:06d}.json", record)
        return directory, config

    def assert_independent(self, request, effort="medium"):
        self.assertEqual(set(request), {"model", "messages", "max_tokens", "stream", "extra_body"})
        self.assertEqual(request["model"], QWEN_MODEL)
        self.assertIs(request["stream"], False)
        expected = {"cache_prompt": False, "reasoning_format": "deepseek"}
        if effort is not None:
            expected["reasoning_effort"] = effort
        self.assertEqual(request["extra_body"], expected)
        self.assertEqual([message["role"] for message in request["messages"]], ["system", "user"])
        self.assertTrue(all(set(message) == {"role", "content"} and isinstance(message["content"], str)
                            for message in request["messages"]))

    def assert_clients_closed(self, sdk):
        self.assertEqual(len(sdk.clients), len(sdk.calls))
        self.assertTrue(all(client.entered and client.closed for client in sdk.clients))
        self.assertTrue(all(http.closed for http in sdk.http_clients))
        self.assertEqual(len({id(client) for client in sdk.clients}), len(sdk.calls))
        self.assertEqual(len({id(http) for http in sdk.http_clients}), len(sdk.calls))
        for client in sdk.clients:
            self.assertEqual(client.options["max_retries"], 0)
            self.assertEqual(client.options["api_key"], KEY)
            self.assertEqual(client.options["base_url"], URL)
            self.assertIs(client.options["http_client"].options["trust_env"], False)
            self.assertIs(client.options["http_client"].options["verify"], True)

    def test_qwen_preview_and_file_default_need_no_credentials_sdk_or_logs(self):
        for args, default in ((["--provider", "qwen", "--n", "2", "--dry-run"], "gpt"),
                              (["--n", "2", "--dry-run"], "qwen")):
            with self.subTest(args=args), patch.object(self.module, "PROVIDER", default):
                result = self.invoke(args, forbidden=True)
                self.assertEqual(result["code"], 0, result["stderr"])
                self.assertIn(QWEN_MODEL, result["stdout"])
                self.assertIn("1024", result["stdout"])
                self.assertIn("网格=DDD/DDD/DDD", result["stdout"])
                self.assertIn("Qwen 思考强度=medium", result["stdout"])
                self.assertEqual(result["gpt_config_calls"], 0)
                self.assertEqual(result["qwen_config_calls"], 0)
                self.assertFalse(result["sdk"].calls)
                self.assertFalse(self.logs.exists())

    def test_full_qwen_batch_selects_env_keeps_history_out_and_logs_raw_responses(self):
        def before_call(index, request):
            directory = next(self.logs.iterdir())
            record = json.loads((directory / "records" / f"{index:06d}.json").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "running")
            self.assertEqual(record["provider"], "qwen")
            self.assertEqual(record["api_interface"], "chat.completions")
            self.assertEqual(record["request"], request)

        with patch.object(self.module, "PROVIDER", "gpt"), patch.object(self.module, "TRUST_ENV", True):
            result = self.invoke(["--provider", "qwen", "--n", "1"], before_call=before_call)
        self.assertEqual(result["code"], 0, result["stderr"])
        self.assertEqual(result["qwen_config_calls"], 1)
        self.assertEqual(result["gpt_config_calls"], 0)
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
            self.assertEqual(metadata["provider"], "qwen")
            self.assertEqual(metadata["api_interface"], "chat.completions")
            self.assertEqual(metadata["model"], QWEN_MODEL)
            self.assertEqual(metadata["qwen_reasoning_effort"], "medium")
        self.assertEqual(config["requests"], sdk.calls)
        self.assertIs(config["trust_env"], False)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["successful_requests"], 512)
        rows = list(csv.DictReader(io.StringIO((directory / "results.csv").read_text(encoding="utf-8-sig"))))
        self.assertEqual(len(rows), 512)
        for row, case in zip(rows, config["cases"]):
            self.assertEqual(row["provider"], "qwen")
            self.assertEqual(row["api_interface"], "chat.completions")
            self.assertEqual(row["model"], QWEN_MODEL)
            self.assertEqual(row["qwen_reasoning_effort"], "medium")
            self.assertEqual(row["grid_key"], case["grid_key"])
            self.assertEqual(row["recommendation"], "D")
            self.assertEqual(row["input_tokens"], "123")
            self.assertEqual(row["output_tokens"], "7")
            self.assertEqual(row["total_tokens"], "130")
        saved = json.loads((directory / "records/000001.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["output_text"], "D")
        self.assertEqual(saved["response"]["choices"][0]["message"]["reasoning_content"],
                         "Offline reasoning retained in the raw record.")

    def test_cli_reasoning_effort_reaches_request_and_saved_metadata(self):
        for argument, expected in ((" LOW ", "low"), ("xhigh", "xhigh")):
            with self.subTest(effort=argument):
                result = self.invoke(
                    ["--provider", "qwen", "--n", "1", "--qwen-reasoning-effort", argument,
                     "--tls-retries", "0"],
                    effects=[FakeConnectionError("connection refused")],
                )
                self.assertEqual(result["code"], 1)
                self.assertEqual(len(result["sdk"].calls), 1)
                self.assert_independent(result["sdk"].calls[0], expected)
                self.assert_clients_closed(result["sdk"])
                self.assertIn(f"Qwen 思考强度={expected}", result["stdout"])
                directory = max(self.logs.iterdir(), key=lambda path: path.name)
                config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
                summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
                for metadata in (config, summary):
                    self.assertEqual(metadata["qwen_reasoning_effort"], expected)
                self.assertTrue(all(request["extra_body"]["reasoning_effort"] == expected
                                    for request in config["requests"]))
                rows = list(csv.DictReader(io.StringIO(
                    (directory / "results.csv").read_text(encoding="utf-8-sig"))))
                self.assertEqual(rows[0]["qwen_reasoning_effort"], expected)

    def test_invalid_reasoning_options_stop_before_credentials_or_logs(self):
        for args in (["--provider", "qwen", "--qwen-reasoning-effort", "high"],
                     ["--provider", "gpt", "--qwen-reasoning-effort", "medium"]):
            with self.subTest(args=args), self.assertRaises(SystemExit) as stopped:
                self.invoke(args, forbidden=True)
            self.assertEqual(stopped.exception.code, 2)
            self.assertFalse(self.logs.exists())
        with patch.object(self.module, "QWEN_REASONING_EFFORT", "unsupported"), \
                self.assertRaises(SystemExit) as stopped:
            self.invoke(["--provider", "qwen", "--dry-run"], forbidden=True)
        self.assertEqual(stopped.exception.code, 2)
        self.assertFalse(self.logs.exists())

    def test_tls_retry_uses_new_clients_and_same_independent_request(self):
        effects = [FakeConnectionError("TLS/SSL connection has been closed (EOF)"),
                   FakeChatResponse(2), FakeConnectionError("connection refused")]
        result = self.invoke(["--provider", "qwen", "--n", "1", "--tls-retries", "1", "--retry-delay", "0"],
                             effects=effects)
        self.assertEqual(result["code"], 1)
        sdk = result["sdk"]
        self.assertEqual(len(sdk.calls), 3)
        self.assertEqual(sdk.calls[0], sdk.calls[1])
        self.assertNotEqual(sdk.calls[1], sdk.calls[2])
        self.assert_clients_closed(sdk)
        for request in sdk.calls:
            self.assert_independent(request)
        directory = next(self.logs.iterdir())
        archived = list((directory / "attempts/000001").glob("*.json"))
        self.assertEqual(len(archived), 1)
        first = json.loads((directory / "records/000001.json").read_text(encoding="utf-8"))
        self.assertEqual(first["status"], "success")
        self.assertEqual(json.loads(archived[0].read_text(encoding="utf-8"))["status"], "error")

    def test_incomplete_or_explanatory_output_is_not_counted_as_an_action(self):
        for action, reason in (("D", "length"), ("D", "content_filter"), ("Choose D", "stop")):
            with self.subTest(action=action, reason=reason):
                result = self.invoke(["--provider", "qwen", "--n", "1", "--tls-retries", "0"],
                                     effects=[FakeChatResponse(action=action, finish_reason=reason)])
                self.assertEqual(result["code"], 1)
                self.assertEqual(len(result["sdk"].calls), 1)
                self.assert_clients_closed(result["sdk"])
                newest = max(self.logs.iterdir(), key=lambda path: path.name)
                saved = json.loads((newest / "records/000001.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["status"], "error")
                self.assertIsNone(saved["action"])
                self.assertEqual(saved["response"]["choices"][0]["finish_reason"], reason)
                self.assertEqual(saved["output_text"], action)

    def test_client_construction_failure_closes_http_client_and_records_failure(self):
        result = self.invoke(["--provider", "qwen", "--n", "1", "--tls-retries", "0"],
                             construction_error=ValueError("invalid test transport configuration"))
        self.assertEqual(result["code"], 1)
        self.assertFalse(result["sdk"].calls)
        self.assertEqual(len(result["sdk"].http_clients), 1)
        self.assertTrue(result["sdk"].http_clients[0].closed)
        directory = next(self.logs.iterdir())
        saved = json.loads((directory / "records/000001.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "error")
        self.assertFalse(saved["api_call_started"])
        self.assertIn("diagnostics", saved["error"])

    def test_resume_uses_saved_qwen_provider_request_and_preserves_successes(self):
        directory, config = self.fixture()
        originals = self.snapshot(directory)
        with patch.object(self.module, "PROVIDER", "gpt"), \
                patch.object(self.module, "QWEN_REASONING_EFFORT", "xhigh"), \
                patch.object(self.module, "SYSTEM_PROMPT", "Edited after the original experiment"), \
                patch.object(self.module, "MAX_OUTPUT_TOKENS", 16), \
                patch.object(self.module, "build_user_prompt", return_value="Changed formatter"):
            result = self.invoke(["--resume", str(directory)])
        self.assertEqual(result["code"], 0, result["stderr"])
        self.assertEqual(result["qwen_config_calls"], 1)
        self.assertEqual(result["gpt_config_calls"], 0)
        self.assertEqual(result["sdk"].calls, config["requests"][511:])
        self.assert_clients_closed(result["sdk"])
        for relative, contents in originals.items():
            self.assertEqual((directory / relative).read_bytes(), contents)
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["successful_requests"], 512)
        self.assertEqual(summary["provider"], "qwen")
        self.assertEqual(summary["model"], QWEN_MODEL)
        self.assertEqual(summary["qwen_reasoning_effort"], "medium")
        self.assert_independent(result["sdk"].calls[0], "medium")

    def test_legacy_resume_preserves_server_default_including_failed_attempts(self):
        directory, config = self.fixture()
        config.pop("qwen_reasoning_effort")
        for request in config["requests"]:
            request["extra_body"].pop("reasoning_effort")
        self.write_json(directory / "config.json", config)
        for path in (directory / "records").glob("*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            record["request"]["extra_body"].pop("reasoning_effort")
            if record["request_index"] == 511:
                record.update(status="error", action=None, output_text=None, response=None,
                              error={"category": "connection", "message": "old connection failure"})
            self.write_json(path, record)
        originals = self.snapshot(directory)
        with patch.object(self.module, "QWEN_REASONING_EFFORT", "medium"):
            result = self.invoke(["--resume", str(directory)])
        self.assertEqual(result["code"], 0, result["stderr"])
        self.assertEqual(result["sdk"].calls, config["requests"][510:])
        self.assert_clients_closed(result["sdk"])
        for request in result["sdk"].calls:
            self.assert_independent(request, effort=None)
        for relative, contents in originals.items():
            if relative != "records/000511.json":
                self.assertEqual((directory / relative).read_bytes(), contents)
        archived = list((directory / "attempts/000511").glob("*.json"))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].read_bytes(), originals["records/000511.json"])
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        self.assertIsNone(summary["qwen_reasoning_effort"])
        self.assertEqual(summary["successful_requests"], 512)
        rows = list(csv.DictReader(io.StringIO((directory / "results.csv").read_text(encoding="utf-8-sig"))))
        self.assertEqual(len(rows), 512)
        self.assertTrue(all(row["qwen_reasoning_effort"] == "server_default" for row in rows))

    def test_resume_effort_override_is_rejected_without_changing_saved_batch(self):
        directory, _ = self.fixture(records=False)
        before = self.snapshot(directory)
        with self.assertRaises(SystemExit) as stopped:
            self.invoke(["--resume", str(directory), "--qwen-reasoning-effort", "medium"], forbidden=True)
        self.assertEqual(stopped.exception.code, 2)
        self.assertEqual(self.snapshot(directory), before)

    def test_resume_explicit_provider_mismatch_stops_before_io_or_changes(self):
        directory, _ = self.fixture(records=False)
        before = self.snapshot(directory)
        result = self.invoke(["--resume", str(directory), "--provider", "gpt"], forbidden=True)
        self.assertEqual(result["code"], 1)
        self.assertEqual(result["qwen_config_calls"], 0)
        self.assertEqual(result["gpt_config_calls"], 0)
        self.assertFalse(result["sdk"].calls)
        self.assertEqual(self.snapshot(directory), before)

    def test_resume_rejects_history_cache_changes_and_inconsistent_provider_metadata(self):
        directory, original = self.fixture(records=False)

        def extra_history(config):
            config["requests"][0]["messages"].append({"role": "assistant", "content": "D"})

        def previous_id(config):
            config["requests"][0]["previous_response_id"] = "resp_previous"

        def cached_prompt(config):
            config["requests"][0]["extra_body"]["cache_prompt"] = True

        def wrong_model(config):
            config["requests"][0]["model"] = "gpt-5.5"

        def wrong_interface(config):
            config["api_interface"] = "responses"

        def no_provider(config):
            config.pop("provider")

        def noncanonical_provider(config):
            config["provider"] = " QWEN "

        def stream_output(config):
            config["requests"][0]["stream"] = True

        def unknown_effort(config):
            config["requests"][0]["extra_body"]["reasoning_effort"] = "high"

        def mismatched_effort(config):
            config["requests"][0]["extra_body"]["reasoning_effort"] = "xhigh"

        def missing_effort(config):
            config["requests"][0]["extra_body"].pop("reasoning_effort")

        def unknown_config_effort(config):
            config["qwen_reasoning_effort"] = "unsupported"

        def request_effort_without_config(config):
            config.pop("qwen_reasoning_effort")

        for mutate in (extra_history, previous_id, cached_prompt, wrong_model,
                       wrong_interface, no_provider, noncanonical_provider, stream_output,
                       unknown_effort, mismatched_effort, missing_effort, unknown_config_effort,
                       request_effort_without_config):
            with self.subTest(mutate=mutate.__name__):
                config = copy.deepcopy(original)
                mutate(config)
                self.write_json(directory / "config.json", config)
                before = self.snapshot(directory)
                result = self.invoke(["--resume", str(directory)], forbidden=True)
                self.assertEqual(result["code"], 1, result["stdout"])
                self.assertEqual(result["qwen_config_calls"], 0)
                self.assertEqual(result["gpt_config_calls"], 0)
                self.assertFalse(result["sdk"].calls)
                self.assertEqual(self.snapshot(directory), before)

    def test_resume_rejects_success_claim_when_raw_chat_output_is_incomplete_or_different(self):
        directory, _ = self.fixture()
        path = directory / "records/000001.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        for change in ("content", "finish_reason", "provider"):
            with self.subTest(change=change):
                record = copy.deepcopy(original)
                if change == "content":
                    record["response"]["choices"][0]["message"]["content"] = "C"
                elif change == "finish_reason":
                    record["response"]["choices"][0]["finish_reason"] = "length"
                else:
                    record["provider"] = "gpt"
                self.write_json(path, record)
                before = self.snapshot(directory)
                result = self.invoke(["--resume", str(directory)], forbidden=True)
                self.assertEqual(result["code"], 1, result["stdout"])
                self.assertFalse(result["sdk"].calls)
                self.assertEqual(self.snapshot(directory), before)

    def test_resume_qwen_preview_reads_saved_provider_without_credentials_or_mutations(self):
        directory, _ = self.fixture()
        before = self.snapshot(directory)
        result = self.invoke(["--resume", str(directory), "--dry-run"], forbidden=True)
        self.assertEqual(result["code"], 0, result["stderr"])
        self.assertIn("512", result["stdout"])
        self.assertEqual(result["qwen_config_calls"], 0)
        self.assertEqual(result["gpt_config_calls"], 0)
        self.assertFalse(result["sdk"].calls)
        self.assertEqual(self.snapshot(directory), before)


if __name__ == "__main__":
    unittest.main()
