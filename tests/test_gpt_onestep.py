"""Offline regression coverage for single/two-turn advice and error diagnostics."""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from main import GPT_onestep_test as advisor


class OneStepAdvisorTests(unittest.TestCase):
    API_KEY = "configured-test-secret"
    OTHER_KEY = "sk-unconfigured-test-secret"
    BEARER_TOKEN = "test-bearer-token.abc123"
    PROXY_USERNAME = "test-proxy-user"
    PROXY_PASSWORD = "test-proxy-password"

    def setUp(self):
        for target in ("socket.socket.connect", "socket.create_connection"):
            guard = patch(target, side_effect=AssertionError("Network access is forbidden in offline tests"))
            guard.start()
            self.addCleanup(guard.stop)

    def response(self, action="D", response_id="response-first", status="completed"):
        return SimpleNamespace(id=response_id, model=advisor.MODEL, status=status, output_text=action)

    def make_sdk(self, *, response=None, responses=None, constructor_error=None, request_error=None):
        """Provide the SDK interface without importing it or making requests."""
        sdk = ModuleType("openai")
        sdk.APIError = type("APIError", (Exception,), {})
        for name in ("APIConnectionError", "APIStatusError", "APITimeoutError"):
            setattr(sdk, name, type(name, (sdk.APIError,), {}))
        client = MagicMock()
        client.__enter__.return_value = client
        if request_error is not None:
            client.responses.create.side_effect = request_error
        elif response is not None:
            client.responses.create.return_value = response
        else:
            client.responses.create.side_effect = responses or [
                self.response(), self.response("C", "response-second"),
            ]
        sdk.OpenAI = MagicMock(return_value=client, side_effect=constructor_error)
        sdk.DefaultHttpxClient = MagicMock()
        return sdk, client

    def run_main(self, sdk, *, save_hook=None, expected_logs=1, single_question=False):
        stdout, stderr = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as temporary:
            log_dir = Path(temporary) / "logs"
            with (
                patch.object(advisor, "LOG_DIR", log_dir),
                patch.object(advisor, "SINGLE_QUESTION", single_question),
                patch.object(advisor, "load_config", return_value=(self.API_KEY, "https://example.invalid/v1")),
                patch.dict(sys.modules, {"openai": sdk}),
                patch.object(advisor, "save_record", side_effect=save_hook or advisor.save_record),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = advisor.main()
            paths = list(log_dir.glob("*.json"))
            self.assertEqual(len(paths), expected_logs)
            self.assertFalse(list(log_dir.glob("*.tmp")))
            raw_log = paths[0].read_text(encoding="utf-8") if paths else ""
            record = json.loads(raw_log) if raw_log else None
        all_output = raw_log + stdout.getvalue() + stderr.getvalue()
        for secret in (
            self.API_KEY, self.OTHER_KEY, self.BEARER_TOKEN,
            self.PROXY_USERNAME, self.PROXY_PASSWORD,
        ):
            self.assertNotIn(secret, all_output)
        return exit_code, record, stdout.getvalue(), stderr.getvalue()

    def assert_diagnostics(self, record, exception_type, detail_fragment):
        diagnostics = record["error"]["diagnostics"]
        self.assertEqual(diagnostics["exception_type"], exception_type)
        self.assertIn(detail_fragment, diagnostics["detail"])
        self.assertTrue(diagnostics["frames"])
        for frame in diagnostics["frames"]:
            self.assertTrue(frame["filename"])
            self.assertGreater(frame["lineno"], 0)
            self.assertTrue(frame["function"])
        runtime = record["runtime"]
        self.assertEqual(runtime["python_executable"], sys.executable)
        self.assertTrue(runtime["python_version"])
        self.assertIsInstance(runtime["packages"], dict)
        self.assertTrue(runtime["openssl_version"])

    def assert_dependency_failure(self, sdk, exception_type, detail_fragment):
        exit_code, record, stdout, stderr = self.run_main(sdk)
        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertEqual(record["status"], "error")
        self.assertIsNone(record["action"])
        self.assertEqual(record["error"]["type"], "dependency_error")
        self.assertIn(detail_fragment, record["error"]["message"])
        self.assertIn("pip install", record["error"]["message"])
        self.assertIn(detail_fragment, stderr)
        self.assertIsNotNone(record["finished_at"])
        self.assert_diagnostics(record, exception_type, detail_fragment)

    def test_import_error_during_client_construction_is_actionable(self):
        sdk, client = self.make_sdk(constructor_error=ImportError(
            "Using SOCKS proxy, but the 'socksio' package is not installed. "
            "Make sure to install httpx using `pip install httpx[socks]`."
        ))
        self.assert_dependency_failure(sdk, "ImportError", "socksio")
        client.responses.create.assert_not_called()

    def test_import_error_during_request_is_actionable_and_redacted(self):
        sdk, client = self.make_sdk(request_error=ImportError(
            f"typing_extensions failed: {self.API_KEY}; {self.OTHER_KEY}; "
            f"Authorization: Bearer {self.BEARER_TOKEN}; "
            f"socks5://{self.PROXY_USERNAME}:{self.PROXY_PASSWORD}@proxy.example.invalid:1080"
        ))
        self.assert_dependency_failure(sdk, "ImportError", "typing_extensions")
        client.responses.create.assert_called_once()

    def test_missing_openai_is_actionable(self):
        # A None entry forces import failure even if openai is installed locally.
        self.assert_dependency_failure(None, "ModuleNotFoundError", "openai")

    def test_configurable_environment_mode_preserves_tls_verification(self):
        for trust_env in (True, False):
            with self.subTest(trust_env=trust_env):
                sdk, client = self.make_sdk()
                with patch.object(advisor, "TRUST_ENV", trust_env):
                    exit_code, record, _, _ = self.run_main(sdk, single_question=True)
                self.assertEqual(exit_code, 0)
                sdk.DefaultHttpxClient.assert_called_once_with(trust_env=trust_env, verify=True)
                self.assertIs(
                    sdk.OpenAI.call_args.kwargs["http_client"],
                    sdk.DefaultHttpxClient.return_value,
                )
                self.assertEqual(sdk.OpenAI.call_args.kwargs["max_retries"], 0)
                self.assertEqual(record["client_options"]["trust_env"], trust_env)
                client.responses.create.assert_called_once()

    def test_constructor_failure_closes_transport_without_api_request(self):
        sdk, client = self.make_sdk(constructor_error=RuntimeError("offline constructor failure"))
        exit_code, record, stdout, _ = self.run_main(sdk, single_question=True)
        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertEqual(record["completed_turns"], 0)
        sdk.DefaultHttpxClient.return_value.close.assert_called_once_with()
        client.responses.create.assert_not_called()

    def test_network_environment_logs_names_without_proxy_or_ca_values(self):
        configuration = {
            "HTTP_PROXY": "http://proxy-user:proxy-password@proxy.invalid:8000",
            "https_proxy": "https://second-secret.invalid:8443",
            "ALL_PROXY": "socks5://third-secret.invalid:1080",
            "no_proxy": "private-internal-host.invalid",
            "SSL_CERT_FILE": "/private/secret-ca-file.pem",
            "SSL_CERT_DIR": "/private/secret-ca-directory",
        }
        sdk, _ = self.make_sdk()
        with patch.dict(os.environ, configuration, clear=True):
            exit_code, record, stdout, stderr = self.run_main(sdk, single_question=True)
        self.assertEqual(exit_code, 0)
        network = record["runtime"]["network_environment"]
        self.assertEqual(set(network["proxy_variables_present"]), {
            "HTTP_PROXY", "https_proxy", "ALL_PROXY", "no_proxy",
        })
        self.assertEqual(set(network["custom_ca_variables_present"]), {
            "SSL_CERT_FILE", "SSL_CERT_DIR",
        })
        combined_output = json.dumps(record, ensure_ascii=False) + stdout + stderr
        for value in configuration.values():
            self.assertNotIn(value, combined_output)

    def test_tls_eof_hint_keeps_underlying_cause_without_retry(self):
        sdk, client = self.make_sdk()
        error = sdk.APIConnectionError("Connection error.")
        error.__cause__ = OSError("TLS/SSL connection has been closed (EOF) (_ssl.c:1133)")
        client.responses.create.side_effect = error
        exit_code, record, stdout, stderr = self.run_main(sdk, single_question=True)
        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertEqual(record["error"]["type"], "connection_error")
        for detail in ("连接失败", "TLS", "TRUST_ENV"):
            self.assertIn(detail, stderr)
        self.assertIn("EOF", record["error"]["diagnostics"]["causes"][0]["detail"])
        self.assertEqual(record["completed_turns"], 0)
        client.responses.create.assert_called_once()

    def test_unrelated_connection_error_keeps_generic_network_message(self):
        sdk, client = self.make_sdk()
        client.responses.create.side_effect = sdk.APIConnectionError("Name resolution failed")
        exit_code, record, _, stderr = self.run_main(sdk, single_question=True)
        self.assertEqual(exit_code, 1)
        self.assertIn("连接失败", stderr)
        self.assertNotIn("TRUST_ENV", stderr)
        self.assertEqual(record["error"]["type"], "connection_error")
        client.responses.create.assert_called_once()

    def test_single_question_sends_and_saves_only_first_prompt(self):
        sdk, client = self.make_sdk()
        first_prompt, second_prompt = "First observed game state", "Unused second game state"
        with (
            patch.object(advisor, "USER_PROMPT", first_prompt),
            patch.object(advisor, "USER_PROMPT_2", second_prompt),
        ):
            exit_code, record, stdout, stderr = self.run_main(sdk, single_question=True)
        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, "第 1 轮建议：D\n")
        self.assertIn("运行记录", stderr)
        client.responses.create.assert_called_once()
        request = client.responses.create.call_args.kwargs
        self.assertEqual(request["input"], [
            {"role": "system", "content": advisor.SYSTEM_PROMPT},
            {"role": "user", "content": first_prompt},
        ])
        self.assertNotIn("previous_response_id", request)
        self.assertEqual(record["status"], "success")
        self.assertTrue(record["single_question"])
        self.assertEqual(record["conversation_mode"], "single_turn")
        self.assertEqual(record["planned_turns"], 1)
        self.assertEqual(record["completed_turns"], 1)
        self.assertEqual(record["user_prompts"], [first_prompt])
        self.assertEqual(len(record["turns"]), 1)
        self.assertEqual(record["turns"][0]["request"], request)
        self.assertEqual(record["turns"][0]["action"], "D")
        self.assertEqual(record["action"], "D")
        self.assertIsNone(record["error"])
        client.__exit__.assert_called_once_with(None, None, None)

    def test_single_question_does_not_require_response_id(self):
        sdk, client = self.make_sdk(response=SimpleNamespace(status="completed", output_text="C"))
        exit_code, record, stdout, _ = self.run_main(sdk, single_question=True)
        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, "第 1 轮建议：C\n")
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["completed_turns"], 1)
        self.assertEqual(record["action"], "C")
        self.assertIsNone(record["error"])
        client.responses.create.assert_called_once()

    def test_single_question_failure_stops_after_one_request(self):
        sdk, client = self.make_sdk()
        client.responses.create.side_effect = sdk.APIConnectionError("simulated connection failure")
        exit_code, record, stdout, _ = self.run_main(sdk, single_question=True)
        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["planned_turns"], 1)
        self.assertEqual(record["completed_turns"], 0)
        self.assertEqual(len(record["turns"]), 1)
        self.assertIsNone(record["action"])
        self.assertIsNotNone(record["error"])
        client.responses.create.assert_called_once()
        client.__exit__.assert_called_once()

    def test_two_turns_send_system_once_and_chain_response_id(self):
        sdk, client = self.make_sdk()
        first_prompt, second_prompt = "First observed game state", "A different second game state"
        with (
            patch.object(advisor, "USER_PROMPT", first_prompt),
            patch.object(advisor, "USER_PROMPT_2", second_prompt),
        ):
            exit_code, record, stdout, stderr = self.run_main(sdk)
        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, "第 1 轮建议：D\n第 2 轮建议：C\n")
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["action"], "C")
        self.assertEqual(record["output_text"], "C")
        self.assertIsNone(record["error"])
        self.assertIn("运行记录", stderr)
        self.assertEqual(client.responses.create.call_count, 2)
        first, second = [call.kwargs for call in client.responses.create.call_args_list]
        self.assertEqual(first, advisor.build_request(first_prompt))
        self.assertEqual(first["input"], [
            {"role": "system", "content": advisor.SYSTEM_PROMPT},
            {"role": "user", "content": first_prompt},
        ])
        self.assertNotIn("previous_response_id", first)
        self.assertEqual(second["input"], [{"role": "user", "content": second_prompt}])
        self.assertEqual(second["previous_response_id"], "response-first")
        for request in (first, second):
            self.assertNotIn("instructions", request)
            self.assertNotIn("conversation", request)
            self.assertTrue(request["store"])
        sdk.OpenAI.assert_called_once_with(
            api_key=self.API_KEY,
            base_url="https://example.invalid/v1",
            timeout=advisor.TIMEOUT_SECONDS,
            max_retries=0,
            http_client=sdk.DefaultHttpxClient.return_value,
        )
        sdk.DefaultHttpxClient.assert_called_once_with(trust_env=advisor.TRUST_ENV, verify=True)
        client.__enter__.assert_called_once()
        client.__exit__.assert_called_once_with(None, None, None)
        self.assertEqual(record["completed_turns"], 2)
        self.assertFalse(record["single_question"])
        self.assertEqual(record["conversation_mode"], "previous_response_id")
        self.assertEqual(record["planned_turns"], 2)
        self.assertEqual(record["user_prompts"], [first_prompt, second_prompt])
        self.assertEqual([turn["action"] for turn in record["turns"]], ["D", "C"])
        self.assertEqual([turn["request"] for turn in record["turns"]], [first, second])
        self.assertEqual(record["response"]["id"], "response-second")
        for turn in record["turns"]:
            self.assertEqual(turn["status"], "success")
            self.assertTrue(turn["api_call_started"])
            self.assertIsNotNone(turn["finished_at"])
            self.assertGreaterEqual(turn["duration_seconds"], 0)

    def test_shared_client_remains_open_between_turns(self):
        sdk, client = self.make_sdk()
        replies = iter([self.response(), self.response("C", "response-second")])

        def reply(**request):
            sdk.OpenAI.assert_called_once()
            client.__enter__.assert_called_once()
            client.__exit__.assert_not_called()
            return next(replies)

        client.responses.create.side_effect = reply
        exit_code, _, _, _ = self.run_main(sdk)
        self.assertEqual(exit_code, 0)
        self.assertEqual(client.responses.create.call_count, 2)
        client.__exit__.assert_called_once_with(None, None, None)

    def test_standalone_run_advisor_owns_and_closes_its_client(self):
        sdk, client = self.make_sdk(response=self.response())
        with patch.dict(sys.modules, {"openai": sdk}):
            action = advisor.run_advisor(self.API_KEY, "https://example.invalid/v1", {})
        self.assertEqual(action, "D")
        sdk.OpenAI.assert_called_once()
        client.responses.create.assert_called_once()
        client.__enter__.assert_called_once()
        client.__exit__.assert_called_once_with(None, None, None)

    def test_supplied_client_remains_owned_by_the_caller(self):
        sdk, client = self.make_sdk(response=self.response())
        with patch.dict(sys.modules, {"openai": sdk}):
            action = advisor.run_advisor(self.API_KEY, "https://example.invalid/v1", {}, client=client)
        self.assertEqual(action, "D")
        sdk.OpenAI.assert_not_called()
        client.responses.create.assert_called_once()
        client.__enter__.assert_not_called()
        client.__exit__.assert_not_called()

    def test_first_result_and_each_input_are_saved_before_next_request(self):
        sdk, client = self.make_sdk()
        saved_snapshots = []

        def reply(**request):
            path, = advisor.LOG_DIR.glob("*.json")
            saved = json.loads(path.read_text(encoding="utf-8"))
            saved_snapshots.append(saved)
            self.assertEqual(saved["turns"][-1]["request"], request)
            self.assertEqual(saved["turns"][-1]["status"], "running")
            self.assertIsNone(saved["turns"][-1]["response"])
            if len(saved_snapshots) == 2:
                self.assertEqual(saved["turns"][0]["status"], "success")
                self.assertEqual(saved["turns"][0]["response"]["id"], "response-first")
                self.assertEqual(saved["turns"][0]["action"], "D")
                return self.response("C", "response-second")
            return self.response()

        client.responses.create.side_effect = reply
        exit_code, _, _, _ = self.run_main(sdk)
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(saved_snapshots), 2)

    def test_missing_first_response_id_preserves_advice_and_prevents_second_request(self):
        for response_id in (None, "", "   ", 42):
            with self.subTest(response_id=response_id):
                sdk, client = self.make_sdk(response=self.response(response_id=response_id))
                exit_code, record, stdout, _ = self.run_main(sdk)
                self.assertEqual(exit_code, 1)
                self.assertEqual(stdout, "第 1 轮建议：D\n")
                self.assertEqual(record["error"]["type"], "conversation_error")
                self.assertEqual(record["completed_turns"], 1)
                self.assertEqual(len(record["turns"]), 1)
                self.assertEqual(record["turns"][0]["action"], "D")
                self.assertEqual(record["turns"][0]["status"], "success")
                client.responses.create.assert_called_once()

    def test_second_turn_connection_failure_preserves_first_turn(self):
        sdk, client = self.make_sdk()
        client.responses.create.side_effect = [self.response(), sdk.APIConnectionError("offline failure")]
        exit_code, record, stdout, stderr = self.run_main(sdk)
        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "第 1 轮建议：D\n")
        self.assertIn("连接失败", stderr)
        self.assertEqual(record["error"]["type"], "connection_error")
        self.assertEqual(record["completed_turns"], 1)
        self.assertEqual(record["turns"][0]["status"], "success")
        self.assertEqual(record["turns"][0]["action"], "D")
        self.assertEqual(record["turns"][1]["status"], "error")
        self.assertEqual(record["turns"][1]["error"]["type"], "connection_error")
        self.assertIsNone(record["action"])
        self.assertIsNone(record["response"])
        self.assertEqual(client.responses.create.call_count, 2)

        sdk.OpenAI.assert_called_once()
        client.__exit__.assert_called_once()

    def test_second_turn_http_400_preserves_first_turn_and_redacts_diagnostics(self):
        sdk, client = self.make_sdk()
        error = sdk.APIStatusError(
            "Unsupported previous_response_id; "
            f"configured key: {self.API_KEY}; other key: {self.OTHER_KEY}; "
            f"Authorization: Bearer {self.BEARER_TOKEN}; "
            f"socks5://{self.PROXY_USERNAME}:{self.PROXY_PASSWORD}@proxy.example.invalid:1080"
        )
        error.status_code = 400
        client.responses.create.side_effect = [self.response(), error]

        exit_code, record, stdout, stderr = self.run_main(sdk)

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "第 1 轮建议：D\n")
        self.assertIn("HTTP 400", stderr)
        self.assertIn("error.diagnostics", stderr)
        self.assertNotIn("Unsupported previous_response_id", stderr)
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["conversation_mode"], "previous_response_id")
        self.assertEqual(record["completed_turns"], 1)
        self.assertEqual(record["error"]["type"], "http_error")
        self.assertEqual(record["error"]["http_status"], 400)
        self.assert_diagnostics(record, "APIStatusError", "Unsupported previous_response_id")
        detail = record["error"]["diagnostics"]["detail"]
        self.assertIn("configured key: [REDACTED]", detail)
        self.assertIn("other key: [REDACTED]", detail)
        self.assertIn("Authorization: Bearer [REDACTED]", detail)
        self.assertIn("socks5://[REDACTED]@proxy.example.invalid:1080", detail)
        first_turn, second_turn = record["turns"]
        self.assertEqual(first_turn["status"], "success")
        self.assertEqual(first_turn["action"], "D")
        self.assertEqual(first_turn["response"]["id"], "response-first")
        self.assertIsNone(first_turn["error"])
        self.assertEqual(second_turn["status"], "error")
        self.assertEqual(second_turn["error"], record["error"])
        self.assertIsNone(second_turn["action"])
        self.assertIsNone(second_turn["response"])
        self.assertEqual(client.responses.create.call_count, 2)
        sdk.OpenAI.assert_called_once()
        client.__exit__.assert_called_once()
        second_request = client.responses.create.call_args_list[1].kwargs
        self.assertEqual(second_request["previous_response_id"], "response-first")
        self.assertEqual(second_request["input"], [
            {"role": "user", "content": advisor.USER_PROMPT_2},
        ])
        for call in sdk.OpenAI.call_args_list:
            self.assertEqual(call.kwargs["max_retries"], 0)

    def test_missing_previous_response_is_specific_without_retry_or_history_fallback(self):
        for code_source in ("attribute", "nested_body"):
            with self.subTest(code_source=code_source):
                sdk, client = self.make_sdk()
                error = sdk.APIStatusError(
                    "Previous response with id 'response-first' not found. "
                    f"configured key: {self.API_KEY}; other key: {self.OTHER_KEY}"
                )
                error.status_code = 400
                if code_source == "attribute":
                    error.code = "previous_response_not_found"
                else:
                    error.code = None
                    error.body = {"error": {"code": "previous_response_not_found"}}
                client.responses.create.side_effect = [self.response(), error]

                exit_code, record, stdout, stderr = self.run_main(sdk)

                self.assertEqual(exit_code, 1)
                self.assertEqual(stdout, "第 1 轮建议：D\n")
                self.assertEqual(record["status"], "error")
                self.assertEqual(record["completed_turns"], 1)
                self.assertEqual(record["error"]["type"], "conversation_not_found")
                self.assertEqual(record["error"]["http_status"], 400)
                for detail in ("平台找不到上一轮的响应记录", "第一轮回答已保存", "响应存储"):
                    self.assertIn(detail, stderr)
                self.assert_diagnostics(record, "APIStatusError", "response-first")
                diagnostics = record["error"]["diagnostics"]
                self.assertEqual(diagnostics["api_error_code"], "previous_response_not_found")
                self.assertIn("configured key: [REDACTED]", diagnostics["detail"])
                self.assertIn("other key: [REDACTED]", diagnostics["detail"])
                first_turn, second_turn = record["turns"]
                self.assertEqual(first_turn["status"], "success")
                self.assertEqual(first_turn["action"], "D")
                self.assertEqual(first_turn["response"]["id"], "response-first")
                self.assertEqual(second_turn["status"], "error")
                self.assertEqual(second_turn["error"], record["error"])
                self.assertEqual(client.responses.create.call_count, 2)
                first_request, second_request = [
                    call.kwargs for call in client.responses.create.call_args_list
                ]
                self.assertEqual([item["role"] for item in first_request["input"]], ["system", "user"])
                self.assertEqual(second_request["previous_response_id"], "response-first")
                self.assertEqual(second_request["input"], [
                    {"role": "user", "content": advisor.USER_PROMPT_2},
                ])

    def test_network_failure_logs_redacted_underlying_cause(self):
        for error_name, category in (("APIConnectionError", "connection_error"),
                                     ("APITimeoutError", "timeout")):
            with self.subTest(error_name=error_name):
                sdk, client = self.make_sdk()
                error = getattr(sdk, error_name)("Request failed")
                error.__cause__ = OSError(
                    f"Proxy connection refused: {self.API_KEY}; {self.OTHER_KEY}; "
                    f"Authorization: Bearer {self.BEARER_TOKEN}; "
                    f"socks5://{self.PROXY_USERNAME}:{self.PROXY_PASSWORD}@proxy.example.invalid:1080"
                )
                client.responses.create.side_effect = [self.response(), error]

                exit_code, record, stdout, _ = self.run_main(sdk)

                self.assertEqual(exit_code, 1)
                self.assertEqual(stdout, "第 1 轮建议：D\n")
                self.assertEqual(record["error"]["type"], category)
                self.assert_diagnostics(record, error_name, "Request failed")
                cause, = record["error"]["diagnostics"]["causes"]
                self.assertEqual(cause["exception_type"], "OSError")
                self.assertIn("Proxy connection refused", cause["detail"])
                self.assertIn("Bearer [REDACTED]", cause["detail"])
                self.assertIn("socks5://[REDACTED]@proxy.example.invalid:1080", cause["detail"])
                self.assertEqual(record["turns"][0]["status"], "success")
                self.assertEqual(record["turns"][1]["status"], "error")
                self.assertEqual(client.responses.create.call_count, 2)

    def test_cause_diagnostics_handle_context_suppression_and_cycles(self):
        outer = RuntimeError("outer failure")
        inner = OSError("underlying context")
        outer.__context__ = inner
        self.assertEqual(advisor.exception_diagnostics(outer)["causes"], [
            {"exception_type": "OSError", "detail": "underlying context"},
        ])

        deep_error = RuntimeError("deep failure")
        current = deep_error
        for depth in range(100):
            current.__cause__ = RuntimeError(f"cause {depth}")
            current = current.__cause__
        causes = advisor.exception_diagnostics(deep_error)["causes"]
        self.assertGreater(len(causes), 0)
        self.assertLessEqual(len(causes), 8)
        outer.__suppress_context__ = True
        self.assertFalse(advisor.exception_diagnostics(outer).get("causes"))
        outer.__cause__ = inner
        inner.__cause__ = outer
        self.assertEqual(advisor.exception_diagnostics(outer)["causes"], [
            {"exception_type": "OSError", "detail": "underlying context"},
        ])

    def test_second_turn_interrupt_preserves_first_turn(self):
        sdk, client = self.make_sdk(responses=[self.response(), KeyboardInterrupt()])
        exit_code, record, stdout, _ = self.run_main(sdk)
        self.assertEqual(exit_code, 130)
        self.assertEqual(stdout, "第 1 轮建议：D\n")
        self.assertEqual(record["status"], "cancelled")
        self.assertEqual(record["completed_turns"], 1)
        self.assertEqual(record["turns"][0]["action"], "D")
        self.assertEqual(record["turns"][0]["status"], "success")
        self.assertEqual(record["turns"][1]["status"], "cancelled")
        self.assertEqual(record["turns"][1]["error"]["type"], "KeyboardInterrupt")
        self.assertIsNotNone(record["turns"][1]["finished_at"])
        self.assertEqual(client.responses.create.call_count, 2)

        sdk.OpenAI.assert_called_once()
        client.__exit__.assert_called_once()

    def test_second_invalid_action_retains_both_raw_outputs(self):
        sdk, client = self.make_sdk(responses=[self.response(), self.response("C or D", "response-second")])
        exit_code, record, stdout, _ = self.run_main(sdk)
        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "第 1 轮建议：D\n")
        self.assertEqual([turn["output_text"] for turn in record["turns"]], ["D", "C or D"])
        self.assertEqual(record["turns"][0]["action"], "D")
        self.assertIsNone(record["turns"][1]["action"])
        self.assertEqual(record["turns"][1]["error"]["type"], "AdviceError")
        self.assertEqual(client.responses.create.call_count, 2)

    def test_initial_logging_failure_prevents_all_requests(self):
        sdk, client = self.make_sdk()

        def fail_save(*args, **kwargs):
            raise OSError("simulated full disk")

        exit_code, _, stdout, stderr = self.run_main(sdk, save_hook=fail_save, expected_logs=0)
        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("无法创建运行记录", stderr)
        sdk.OpenAI.assert_not_called()
        client.responses.create.assert_not_called()

    def test_result_logging_failure_prevents_second_request(self):
        sdk, client = self.make_sdk()
        original_save = advisor.save_record
        failed = False

        def fail_once_after_first_result(path, record, api_key=""):
            nonlocal failed
            if not failed and record["completed_turns"] == 1:
                failed = True
                raise OSError("simulated full disk")
            original_save(path, record, api_key)

        exit_code, record, stdout, stderr = self.run_main(sdk, save_hook=fail_once_after_first_result)
        self.assertEqual(exit_code, 1)
        self.assertTrue(failed)
        self.assertEqual(stdout, "")
        self.assertEqual(record["error"]["type"], "logging_error")
        self.assertIn("无法保存对话记录", stderr)
        self.assertEqual(record["turns"][0]["action"], "D")
        client.responses.create.assert_called_once()
        sdk.OpenAI.assert_called_once()
        client.__exit__.assert_called_once()

    def test_invalid_action_is_recorded_without_becoming_advice(self):
        for output in ("D because it has a higher payoff", "", None, "CD"):
            with self.subTest(output=output):
                sdk, client = self.make_sdk(response=SimpleNamespace(status="completed", output_text=output))
                exit_code, record, stdout, _ = self.run_main(sdk)
                self.assertEqual(exit_code, 1)
                self.assertEqual(stdout, "")
                self.assertEqual(record["status"], "error")
                self.assertIsNone(record["action"])
                self.assertEqual(record["output_text"], output)
                self.assertEqual(record["error"]["type"], "AdviceError")
                client.responses.create.assert_called_once()

    def test_incomplete_response_is_rejected_even_with_valid_action(self):
        sdk, client = self.make_sdk(response=SimpleNamespace(status="incomplete", output_text="D"))
        exit_code, record, stdout, _ = self.run_main(sdk)
        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIsNone(record["action"])
        self.assertEqual(record["output_text"], "D")
        self.assertEqual(record["error"]["type"], "AdviceError")
        self.assertEqual(record["completed_turns"], 0)
        client.responses.create.assert_called_once()

    def test_reasoning_exhaustion_explains_output_budget_without_retry(self):
        for details in ({"reason": "max_output_tokens"}, SimpleNamespace(reason="max_output_tokens")):
            with self.subTest(details=type(details).__name__):
                response_data = {
                    "id": "response-incomplete",
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "usage": {
                        "output_tokens": 1024,
                        "output_tokens_details": {"reasoning_tokens": 1024},
                    },
                    "output": [{"type": "reasoning"}],
                }
                response = SimpleNamespace(
                    status="incomplete", output_text="", incomplete_details=details,
                    model_dump=lambda **kwargs: response_data,
                )
                sdk, client = self.make_sdk(response=response)
                exit_code, record, stdout, stderr = self.run_main(sdk)
                self.assertEqual(exit_code, 1)
                self.assertEqual(stdout, "")
                self.assertIsNone(record["action"])
                self.assertEqual(record["completed_turns"], 0)
                self.assertEqual(record["response"], response_data)
                message = record["error"]["message"]
                self.assertIn("max_output_tokens", message)
                self.assertIn("MAX_OUTPUT_TOKENS", message)
                self.assertIn("推理", message)
                self.assertIn(message, stderr)
                client.responses.create.assert_called_once()

    def test_content_filter_does_not_recommend_increasing_token_budget(self):
        for details in ({"reason": "content_filter"}, SimpleNamespace(reason="content_filter")):
            with self.subTest(details=type(details).__name__):
                response = SimpleNamespace(
                    status="incomplete", output_text="D", incomplete_details=details,
                )
                sdk, client = self.make_sdk(response=response)
                exit_code, record, stdout, _ = self.run_main(sdk)
                self.assertEqual(exit_code, 1)
                self.assertEqual(stdout, "")
                self.assertIsNone(record["action"])
                self.assertEqual(record["completed_turns"], 0)
                self.assertIn("content_filter", record["error"]["message"])
                self.assertNotIn("增大", record["error"]["message"])
                client.responses.create.assert_called_once()

    def test_unexpected_exception_has_redacted_diagnostics(self):
        sdk, _ = self.make_sdk(request_error=TypeError(
            f"unexpected SDK payload: {self.API_KEY}; {self.OTHER_KEY}"
        ))
        exit_code, record, stdout, stderr = self.run_main(sdk)
        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["error"]["type"], "TypeError")
        self.assertIn("unexpected SDK payload", stderr)
        self.assert_diagnostics(record, "TypeError", "unexpected SDK payload")


if __name__ == "__main__":
    unittest.main()
