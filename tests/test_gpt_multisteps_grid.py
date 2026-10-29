"""Offline checks for exhaustive 3x3 observations and independent batch requests.

No real credentials or network access are permitted. All output goes to temporary
directories, and every model response comes from the fake SDK below.
"""

import builtins
import contextlib
import copy
import csv
import importlib.util
import io
import itertools
import json
import math
import os
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
KEY = "sk-offline-grid-tests"
URL = "https://example.invalid/v1"


def guarded_open(original):
    def open_without_credentials(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            candidate = Path(os.fsdecode(file)).resolve()
            if candidate.is_relative_to(PROJECT / "API"):
                raise AssertionError("Offline test tried to read real credentials")
        return original(file, *args, **kwargs)

    return open_without_credentials


class FakeAPIError(Exception):
    pass


class FakeConnectionError(FakeAPIError):
    pass


class FakeTimeoutError(FakeConnectionError):
    pass


class FakeStatusError(FakeAPIError):
    pass


class FakeResponse:
    def __init__(self, index=1, action="D"):
        self.id = f"resp_offline_grid_{index}"
        self.model = "gpt-5.5"
        self.status = "completed"
        self.output_text = action

    def model_dump(self, mode="json"):
        return {
            "id": self.id,
            "model": self.model,
            "status": self.status,
            "output": [{"type": "message", "content": [
                {"type": "output_text", "text": self.output_text}
            ]}],
            "usage": {"input_tokens": 20, "output_tokens": 1, "total_tokens": 21},
        }


class FakeSDK(types.ModuleType):
    def __init__(self, effects=(), before_call=None):
        super().__init__("openai")
        self.effects = list(effects)
        self.before_call = before_call
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
                sdk.http_clients.append(self)

            def close(self):
                pass

        class Client:
            def __init__(self, **kwargs):
                self.options = kwargs
                self.responses = self
                self.entered = False
                self.exited = False
                sdk.clients.append(self)

            def __enter__(self):
                self.entered = True
                return self

            def __exit__(self, *args):
                self.exited = True
                return False

            def create(self, **kwargs):
                sdk.calls.append(copy.deepcopy(kwargs))
                index = len(sdk.calls)
                if sdk.before_call is not None:
                    sdk.before_call(index, kwargs)
                result = sdk.effects[index - 1] if index <= len(sdk.effects) else FakeResponse(index)
                if isinstance(result, BaseException):
                    raise result
                return result

        self.DefaultHttpxClient = HttpClient
        self.OpenAI = Client


class GridBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        guards = contextlib.ExitStack()
        cls.addClassCleanup(guards.close)
        guards.enter_context(patch("builtins.open", side_effect=guarded_open(builtins.open)))
        guards.enter_context(patch("io.open", side_effect=guarded_open(io.open)))
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            guards.enter_context(patch(target, side_effect=AssertionError("Network forbidden in offline grid tests")))
        guards.enter_context(patch.object(sys, "dont_write_bytecode", True))
        original_path = sys.path[:]
        cls.addClassCleanup(lambda: sys.path.__setitem__(slice(None), original_path))
        sys.path.insert(0, str(PROJECT / "main"))
        spec = importlib.util.spec_from_file_location("tested_grid_batch", PROJECT / "main/GPT_Multisteps.py")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def setUp(self):
        # 此组测试针对 GPT，不能依赖用户在脚本中选择的默认服务。
        provider = patch.object(self.module, "PROVIDER", "gpt")
        provider.start()
        self.addCleanup(provider.stop)
        temporary = tempfile.TemporaryDirectory(prefix="offline_grid_batch_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.logs = self.root / "logs"
        self.log_patch = patch.object(self.module, "LOG_DIR", self.logs)
        self.log_patch.start()
        self.addCleanup(self.log_patch.stop)

    def invoke(self, args, *, effects=(), before_call=None, forbidden=False):
        sdk = FakeSDK(effects, before_call)
        stdout, stderr = io.StringIO(), io.StringIO()
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if forbidden and (name == "openai" or name.startswith("openai.")):
                raise AssertionError("This path must not import the SDK")
            return original_import(name, *args, **kwargs)

        config_options = {"side_effect": AssertionError("This path must not read credentials")} if forbidden else {
            "return_value": (KEY, URL)
        }
        with patch.dict(sys.modules, {"openai": sdk}), \
                patch.object(self.module, "load_config", **config_options) as config, \
                patch("builtins.__import__", side_effect=guarded_import), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = self.module.main(args)
        return {"code": code, "sdk": sdk, "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(), "config_calls": config.call_count}

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def snapshot(directory):
        return {str(path.relative_to(directory)): path.read_bytes()
                for path in directory.rglob("*") if path.is_file()}

    def fixture(self, *, records=True):
        cases = self.module.build_cases(1.5, "grid_3x3")
        config = {
            "batch_id": "offline-grid-resume", "model": self.module.MODEL,
            "base_url": URL, "b": 1.5, "n": 1, **self.module.b_metadata(1.5),
            "observation_format": "grid_3x3", "case_count": 512,
            "planned_requests": 512, "cases": cases,
            "system_prompt": self.module.SYSTEM_PROMPT,
            "requests": [self.module.make_request(case, 1500) for case in cases],
            "max_output_tokens": 1500, "timeout_seconds": 120,
            "trust_env": False, "max_retries": 0, "store": False,
            "continue_on_error": False,
        }
        directory = self.root / "resume"
        (directory / "records").mkdir(parents=True)
        self.write_json(directory / "config.json", config)
        if records:
            for index in range(1, 512):
                success = index != 511
                response = FakeResponse(index).model_dump() if success else None
                record = {
                    "batch_id": config["batch_id"], "request_index": index,
                    "repetition": 1, "combination_index": index,
                    "observation": cases[index - 1], "base_url": URL,
                    "observation_format": "grid_3x3", **self.module.b_metadata(1.5),
                    "client_options": {"timeout_seconds": 120, "max_retries": 0, "trust_env": False},
                    "request": config["requests"][index - 1], "api_call_started": True,
                    "started_at": "2026-10-16T16:00:00.000+08:00",
                    "finished_at": "2026-10-16T16:00:01.000+08:00", "duration_seconds": 1.0,
                    "status": "success" if success else "error",
                    "action": "D" if success else None,
                    "output_text": "D" if success else None, "response": response,
                    "error": None if success else {"type": "APIConnectionError", "message": "连接失败。"},
                }
                self.write_json(directory / "records" / f"{index:06d}.json", record)
        return directory, config

    def assert_independent(self, request):
        self.assertEqual(set(request), {"model", "input", "max_output_tokens", "store"})
        self.assertIs(request["store"], False)
        self.assertEqual([message["role"] for message in request["input"]], ["system", "user"])
        self.assertTrue(all(isinstance(message["content"], str) for message in request["input"]))

    def test_exhaustive_grids_have_correct_center_and_cardinal_neighbors(self):
        self.assertEqual(self.module.OBSERVATION_FORMAT, "grid_3x3")
        cases = self.module.build_cases(None)
        expected = ["".join(cells) for cells in itertools.product("CD", repeat=9)]
        self.assertEqual([case["grid_key"] for case in cases], expected)
        self.assertEqual(len({case["grid_key"] for case in cases}), 512)
        counts = Counter()
        cardinal_patterns = Counter()
        for case, key in zip(cases, expected):
            self.assertEqual(case["grid_rows"], [key[:3], key[3:6], key[6:]])
            self.assertEqual(case["current_action"], key[4])
            cardinal = "".join(key[index] for index in (1, 3, 5, 7))
            cooperating = cardinal.count("C")
            self.assertEqual(case["cooperating_neighbors"], cooperating)
            self.assertEqual(case["defecting_neighbors"], 4 - cooperating)
            counts[key[4], cooperating] += 1
            cardinal_patterns[key[4], cardinal] += 1
        for center in "CD":
            for cooperating in range(5):
                self.assertEqual(counts[center, cooperating], 16 * math.comb(4, cooperating))
        self.assertEqual(set(cardinal_patterns.values()), {16})
        self.assertEqual(len(cardinal_patterns), 32)

    def test_prompts_show_geometry_and_calculated_payoffs(self):
        unknown_cases = {case["grid_key"]: case for case in self.module.build_cases(None)}
        # Same orthogonal neighbors, with only the focal player's action changed.
        cooperative = unknown_cases["CCCDCCDDD"]
        defective = unknown_cases["CCCDDCDDD"]
        for case in (cooperative, defective):
            prompt = self.module.build_user_prompt(case)
            self.assertIn("Current local 3x3 grid", prompt)
            for row in case["grid_rows"]:
                self.assertIn(" ".join(row), prompt)
            self.assertIn("row 2, column 2", prompt)
            self.assertIn("Return only one action: C or D.", prompt)
            self.assertNotIn("b = None", prompt)
        self.assertIn("My current payoff is 2.", self.module.build_user_prompt(cooperative))
        self.assertRegex(self.module.build_user_prompt(defective), r"My current payoff is (?:2\s*\*?\s*b|b\s*\*\s*2)")
        self.assertIn("My current payoff is 0.", self.module.build_user_prompt(unknown_cases["DDDDDDDDD"]))
        fixed = dict(defective, b=1.5)
        self.assertIn("b = 1.5", self.module.build_user_prompt(fixed))
        self.assertRegex(self.module.build_user_prompt(fixed), r"My current payoff is 3(?:\.0)?\.")

    def test_dry_run_previews_all_patterns_without_sdk_credentials_or_writes(self):
        result = self.invoke(["--n", "10", "--dry-run"], forbidden=True)
        self.assertEqual(result["code"], 0, result["stderr"])
        self.assertEqual(result["config_calls"], 0)
        self.assertFalse(result["sdk"].calls)
        self.assertFalse(self.logs.exists())
        self.assertIn("5120", result["stdout"])
        self.assertIn("网格=CCC/CCC/CCC", result["stdout"])
        self.assertIn("网格=DDD/DDD/DDD", result["stdout"])
        self.assertIn("Current local 3x3 grid", result["stdout"])
        self.assertIn("Return only one action: C or D.", result["stdout"])

    def test_full_512_batch_is_independent_checkpoints_and_exports_exact_grids(self):
        def before_call(index, request):
            directories = list(self.logs.iterdir())
            self.assertEqual(len(directories), 1)
            directory = directories[0]
            record = json.loads((directory / "records" / f"{index:06d}.json").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "running")
            self.assertFalse(record["api_call_started"])
            self.assertEqual(record["request"], request)
            self.assertTrue((directory / "config.json").is_file())
            if index > 1:
                previous = json.loads((directory / "records" / f"{index - 1:06d}.json").read_text(encoding="utf-8"))
                self.assertEqual(previous["status"], "success")
                self.assertEqual(previous["action"], "D")

        with patch.object(self.module, "TRUST_ENV", False):
            result = self.invoke(["--n", "1"], before_call=before_call)
        self.assertEqual(result["code"], 0, result["stderr"])
        sdk = result["sdk"]
        self.assertEqual(len(sdk.calls), 512)
        for request in sdk.calls:
            self.assert_independent(request)
            self.assertEqual(request["max_output_tokens"], self.module.MAX_OUTPUT_TOKENS)
        for client in sdk.clients:
            self.assertEqual(client.options["max_retries"], 0)
            self.assertTrue(client.entered and client.exited)
            self.assertIs(client.options["http_client"].options["trust_env"], False)
            self.assertIs(client.options["http_client"].options["verify"], True)
        directory = next(self.logs.iterdir())
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["observation_format"], "grid_3x3")
        self.assertEqual(config["case_count"], 512)
        self.assertEqual(config["planned_requests"], 512)
        self.assertIs(config["trust_env"], False)
        self.assertEqual(config["requests"], sdk.calls)
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["successful_requests"], 512)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(len(summary["cases"]), 512)
        rows = list(csv.DictReader(io.StringIO((directory / "results.csv").read_text(encoding="utf-8-sig"))))
        self.assertEqual(len(rows), 512)
        for row, case in zip(rows, config["cases"]):
            self.assertEqual(row["grid_key"], case["grid_key"])
            self.assertEqual(row["grid_rows"], "/".join(case["grid_rows"]))
            self.assertEqual(row["observation_format"], "grid_3x3")
            self.assertEqual(row["recommendation"], "D")
            saved = json.loads((directory / row["record_file"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["observation"], case)
            self.assertEqual(saved["request"]["input"][1]["content"], self.module.build_user_prompt(case))
        self.assertIn("网格=CCC/CCC/CCC", result["stdout"])
        self.assertIn("网格=DDD/DDD/DDD", result["stdout"])

    def test_connection_failure_stops_and_keeps_first_grid_and_diagnostics(self):
        result = self.invoke(["--n", "1", "--tls-retries", "0"],
                             effects=[FakeConnectionError("TLS EOF sk-private-test")])
        self.assertEqual(result["code"], 1)
        self.assertEqual(len(result["sdk"].calls), 1)
        directory = next(self.logs.iterdir())
        records = list((directory / "records").glob("*.json"))
        self.assertEqual(len(records), 1)
        record = json.loads(records[0].read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["observation"]["grid_key"], "CCCCCCCCC")
        self.assertTrue(record["api_call_started"])
        self.assertIsNone(record["action"])
        self.assertIn("diagnostics", record["error"])
        self.assertNotIn("sk-private-test", records[0].read_text(encoding="utf-8"))
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "stopped_on_error")
        self.assertEqual(summary["successful_requests"], 0)
        self.assertEqual(summary["failed_requests"], 1)
        self.assertEqual(summary["not_run_requests"], 511)

    def test_resume_only_pending_uses_saved_templates_and_preserves_success_bytes(self):
        directory, config = self.fixture()
        original_config = (directory / "config.json").read_bytes()
        original_successes = {path: path.read_bytes() for path in (directory / "records").glob("*.json")
                              if int(path.stem) < 511}
        failed_bytes = (directory / "records" / "000511.json").read_bytes()
        with patch.object(self.module, "SYSTEM_PROMPT", "Changed after collection"), \
                patch.object(self.module, "MODEL", "different-default-model"), \
                patch.object(self.module, "MAX_OUTPUT_TOKENS", 16), \
                patch.object(self.module, "TRUST_ENV", True), \
                patch.object(self.module, "build_user_prompt", return_value="Changed formatter"):
            result = self.invoke(["--resume", str(directory)])
        self.assertEqual(result["code"], 0, result["stderr"])
        self.assertEqual(result["sdk"].calls, config["requests"][510:])
        self.assertEqual((directory / "config.json").read_bytes(), original_config)
        for path, original in original_successes.items():
            self.assertEqual(path.read_bytes(), original)
        archived = list((directory / "attempts" / "000511").glob("*.json"))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].read_bytes(), failed_bytes)
        for client in result["sdk"].clients:
            self.assertIs(client.options["http_client"].options["trust_env"], False)
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["successful_requests"], 512)
        self.assertEqual(summary["pending_requests"], 0)
        self.assertEqual(summary["archived_attempts"], 1)
        self.assertEqual(len(summary["cases"]), 512)
        rows = list(csv.DictReader(io.StringIO((directory / "results.csv").read_text(encoding="utf-8-sig"))))
        self.assertEqual(len(rows), 512)
        self.assertEqual(rows[-1]["grid_rows"], "DDD/DDD/DDD")

    def test_invalid_grid_plan_stops_before_credentials_sdk_and_mutations(self):
        directory, original = self.fixture(records=False)

        def wrong_rows(config):
            config["cases"][0]["grid_rows"][0] = "DDD"

        def wrong_center(config):
            config["cases"][0]["current_action"] = "D"

        def duplicate_pattern(config):
            config["cases"][1] = copy.deepcopy(config["cases"][0])

        def no_full_templates(config):
            config["requests"].pop()

        def stored_request(config):
            config["requests"][0]["store"] = True

        def includes_history(config):
            config["requests"][0]["previous_response_id"] = "resp_other"

        def wrong_roles(config):
            config["requests"][0]["input"][0]["role"] = "assistant"

        for mutation in (wrong_rows, wrong_center, duplicate_pattern, no_full_templates,
                         stored_request, includes_history, wrong_roles):
            with self.subTest(mutation=mutation.__name__):
                config = copy.deepcopy(original)
                mutation(config)
                self.write_json(directory / "config.json", config)
                before = self.snapshot(directory)
                result = self.invoke(["--resume", str(directory)], forbidden=True)
                self.assertEqual(result["code"], 1, result["stdout"])
                self.assertEqual(result["config_calls"], 0)
                self.assertFalse(result["sdk"].calls)
                self.assertEqual(self.snapshot(directory), before)

    def test_resume_preview_does_not_touch_records_or_import_sdk(self):
        directory, _ = self.fixture()
        before = self.snapshot(directory)
        result = self.invoke(["--resume", str(directory), "--dry-run"], forbidden=True)
        self.assertEqual(result["code"], 0, result["stderr"])
        self.assertIn("511, 512", result["stdout"])
        self.assertEqual(result["config_calls"], 0)
        self.assertEqual(self.snapshot(directory), before)


if __name__ == "__main__":
    unittest.main()
