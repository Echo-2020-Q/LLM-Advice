"""Qwen adapter/one-step tests: fake SDK, temporary configuration, no network."""

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
from main import llm_provider as providers


def response(content="C", reason="stop", **message_fields):
    return {
        "id": "chat-current", "model": providers.QWEN_MODEL,
        "choices": [{"index": 0, "finish_reason": reason,
                     "message": {"role": "assistant", "content": content,
                                 "reasoning_content": "offline private reasoning", **message_fields}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


class QwenOneStepTests(unittest.TestCase):
    def setUp(self):
        for name in ("socket.socket.connect", "socket.create_connection"):
            guard = patch(name, side_effect=AssertionError("Network forbidden"))
            guard.start()
            self.addCleanup(guard.stop)

    def make_sdk(self, responses):
        sdk = ModuleType("openai")
        sdk.APIError = type("APIError", (Exception,), {})
        for name in ("APIConnectionError", "APIStatusError", "APITimeoutError"):
            setattr(sdk, name, type(name, (sdk.APIError,), {}))
        clients = []

        def new_client(**options):
            if clients:
                clients[-1].close.assert_called_once_with()
            client = MagicMock()
            client.__enter__.return_value = client
            client.__exit__.side_effect = lambda *args: client.close() and False
            item = responses[len(clients)]
            if isinstance(item, Exception):
                client.chat.completions.create.side_effect = item
            else:
                client.chat.completions.create.return_value = item
            clients.append(client)
            return client

        sdk.OpenAI = MagicMock(side_effect=new_client)
        sdk.DefaultHttpxClient = MagicMock()
        return sdk, clients

    def run_qwen(self, responses, *, single=False, qwen_reasoning_effort=None):
        sdk, clients = self.make_sdk(responses)
        with tempfile.TemporaryDirectory() as directory:
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                patch.object(advisor, "LOG_DIR", Path(directory)),
                patch.object(advisor, "SINGLE_QUESTION", single),
                patch.object(advisor, "load_qwen_config", return_value=("qwen-test-key", "http://qwen.invalid:8080/v1")),
                patch.object(advisor, "load_config", side_effect=AssertionError("GPT credentials forbidden")),
                patch.dict(sys.modules, {"openai": sdk}),
                redirect_stdout(stdout), redirect_stderr(stderr),
            ):
                result = advisor.main(provider="qwen", qwen_reasoning_effort=qwen_reasoning_effort)
            paths = list(Path(directory).glob("*.json"))
            self.assertEqual(len(paths), 1)
            text = paths[0].read_text(encoding="utf-8")
            self.assertNotIn("qwen-test-key", text + stdout.getvalue() + stderr.getvalue())
            record = json.loads(text)
        return result, record, clients, sdk, stdout.getvalue(), stderr.getvalue()

    def test_two_qwen_prompts_use_separate_closed_clients_and_no_history(self):
        code, record, clients, sdk, stdout, _ = self.run_qwen([response("C"), response("D")])
        self.assertEqual(code, 0)
        self.assertEqual(record["completed_turns"], 2)
        self.assertEqual(record["provider"], "qwen")
        self.assertEqual(record["conversation_mode"], "independent_turns")
        self.assertEqual(record["api_interface"], "chat.completions")
        self.assertEqual(record["qwen_reasoning_effort"], "medium")
        self.assertFalse(record["client_options"]["shared_across_turns"])
        self.assertEqual(Path(record["env_file"]).name, "Qwen.env")
        self.assertEqual(len(clients), 2)
        for i, (client, prompt) in enumerate(zip(clients, [advisor.USER_PROMPT, advisor.USER_PROMPT_2])):
            client.close.assert_called_once_with()
            client.responses.create.assert_not_called()
            request = client.chat.completions.create.call_args.kwargs
            self.assertEqual(request, providers.build_qwen_request(advisor.SYSTEM_PROMPT, prompt,
                             max_tokens=advisor.MAX_OUTPUT_TOKENS,
                             reasoning_effort=advisor.QWEN_REASONING_EFFORT))
            self.assertNotIn("previous_response_id", request)
            self.assertNotIn("store", request)
            self.assertEqual(record["turns"][i]["request"], request)
            self.assertEqual(record["turns"][i]["response"]["choices"][0]["message"]["reasoning_content"],
                             "offline private reasoning")
        self.assertIn("第 1 轮建议：C", stdout)
        self.assertIn("第 2 轮建议：D", stdout)
        for call in sdk.DefaultHttpxClient.call_args_list:
            self.assertEqual(call.kwargs, {"trust_env": False, "verify": True})
        for call in sdk.OpenAI.call_args_list:
            self.assertEqual(call.kwargs["max_retries"], 0)

    def test_single_qwen_question_requires_no_saved_response_id(self):
        reply = response()
        reply.pop("id")
        code, record, clients, _, _, _ = self.run_qwen([reply], single=True)
        self.assertEqual(code, 0)
        self.assertEqual(record["completed_turns"], 1)
        self.assertEqual(len(clients), 1)
        clients[0].close.assert_called_once_with()

    def test_incomplete_result_saved_and_client_closed_without_second_request(self):
        code, record, clients, _, stdout, stderr = self.run_qwen([response("D", "length")])
        self.assertEqual(code, 1)
        self.assertEqual(record["completed_turns"], 0)
        self.assertEqual(record["response"]["choices"][0]["finish_reason"], "length")
        self.assertIsNone(record["action"])
        self.assertEqual(len(clients), 1)
        clients[0].close.assert_called_once_with()
        self.assertEqual(stdout, "")
        self.assertIn("max_tokens", stderr)

    def test_exception_closes_client_and_never_retries(self):
        code, record, clients, _, _, stderr = self.run_qwen([RuntimeError("offline-failure qwen-test-key")])
        self.assertEqual(code, 1)
        self.assertEqual(len(clients), 1)
        clients[0].close.assert_called_once_with()
        clients[0].chat.completions.create.assert_called_once()
        self.assertIn("offline-failure", stderr)
        self.assertEqual(record["completed_turns"], 0)

    def test_response_validation_keeps_reasoning_separate_and_checks_finish_reason(self):
        self.assertEqual(providers.validate_chat_response(response(" D \n")), "D")
        for payload in [response("C", "length"), response("C", "content_filter"),
                        response("C", None), response("C", refusal="refused"),
                        response("C", tool_calls=[{"id": "bad"}]), response("<think>C</think>D"),
                        response(None), {"choices": []}]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                providers.validate_chat_response(payload)
        sdk_response = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content="C", reasoning_content="long reasoning"))])
        self.assertEqual(providers.validate_chat_response(sdk_response), "C")
        text, raw = providers.chat_response_record(sdk_response)
        self.assertEqual(text, "C")
        self.assertEqual(raw["choices"][0]["message"]["reasoning_content"], "long reasoning")

    def test_qwen_file_parsing_never_uses_gpt_or_process_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Qwen.env"
            path.write_text('\ufeffexport GPT_API_KEY="offline-qwen" # key\nbase_url="http://192.0.2.1:8080/v1/"\n',
                            encoding="utf-8")
            with patch.object(providers, "provider_env_file", return_value=path), patch.dict(os.environ, {
                "GPT_API_KEY": "wrong", "OPENAI_API_KEY": "wrong", "base_url": "https://wrong.invalid/v1",
            }):
                self.assertEqual(providers.load_qwen_config(), ("offline-qwen", "http://192.0.2.1:8080/v1"))
                for malformed in ["GPT_API_KEY=offline\n", "base_url=http://localhost:8000/v1\n",
                                  "GPT_API_KEY=offline\nbase_url=http://user:password@localhost:8000/v1\n",
                                  "GPT_API_KEY=offline\nbase_url=http://localhost:BAD/v1\n"]:
                    path.write_text(malformed, encoding="utf-8")
                    with self.assertRaises(providers.ConfigError) as failure:
                        providers.load_qwen_config()
                    self.assertNotIn("password", str(failure.exception))
                    self.assertNotIn("offline", str(failure.exception))

    def test_selector_and_cli_route_without_network_or_credentials(self):
        self.assertEqual(providers.normalize_provider(" QWEN "), "qwen")
        self.assertEqual(providers.provider_env_file("qwen").name, "Qwen.env")
        self.assertEqual(providers.provider_env_file("gpt").name, "gpt.env")
        with self.assertRaises(ValueError):
            providers.normalize_provider("other")
        with patch.object(advisor, "main", return_value=7) as main:
            self.assertEqual(advisor.cli(["--provider", "qwen"]), 7)
            main.assert_called_once_with(provider="qwen", qwen_reasoning_effort="medium")
        with patch.object(advisor, "check_response_log") as check, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                advisor.cli(["--provider", "qwen", "--check-response", "anything.json"])
            check.assert_not_called()
        with self.assertRaises(ValueError):
            advisor.build_request("prompt", previous_response_id="old", provider="qwen")

    def test_gpt_diagnostic_override_uses_gpt_transport_when_default_is_qwen(self):
        sdk = ModuleType("openai")
        client = MagicMock()
        client.__enter__.return_value = client
        client.responses.retrieve.return_value = SimpleNamespace(id="resp-offline", status="completed")
        sdk.OpenAI = MagicMock(return_value=client)
        sdk.DefaultHttpxClient = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "source.json"
            log.write_text(json.dumps({
                "base_url": "https://gpt.invalid/v1", "turns": [{
                    "status": "success", "request": {"store": True},
                    "response": {"id": "resp-offline"},
                }],
            }), encoding="utf-8")
            original = log.read_bytes()
            with (
                patch.object(advisor, "PROVIDER", "qwen"),
                patch.object(advisor, "TRUST_ENV", True),
                patch.object(advisor, "QWEN_TRUST_ENV", False),
                patch.object(advisor, "LOG_DIR", Path(directory) / "logs"),
                patch.object(advisor, "load_config", return_value=("offline-gpt-key", "https://gpt.invalid/v1")),
                patch.object(advisor, "load_qwen_config", side_effect=AssertionError("Qwen credentials forbidden")),
                patch.dict(sys.modules, {"openai": sdk}),
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(advisor.cli(["--provider", "gpt", "--check-response", str(log)]), 0)
            self.assertEqual(log.read_bytes(), original)
        sdk.DefaultHttpxClient.assert_called_once_with(trust_env=True, verify=True)
        client.responses.retrieve.assert_called_once_with("resp-offline")
        client.responses.create.assert_not_called()
        client.chat.completions.create.assert_not_called()

    def test_invalid_editable_provider_is_cli_error_before_start(self):
        with patch.object(advisor, "PROVIDER", "typo"), patch.object(advisor, "main") as main:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
                advisor.cli([])
            self.assertEqual(failure.exception.code, 2)
            main.assert_not_called()

    def test_all_qwen_efforts_map_to_extra_body_and_logs(self):
        self.assertEqual(providers.QWEN_REASONING_EFFORTS, ("low", "medium", "xhigh"))
        for level in providers.QWEN_REASONING_EFFORTS:
            with self.subTest(level=level):
                code, record, clients, _, _, _ = self.run_qwen(
                    [response("D")], single=True, qwen_reasoning_effort=f" {level.upper()} ",
                )
                self.assertEqual(code, 0)
                self.assertEqual(record["qwen_reasoning_effort"], level)
                request = clients[0].chat.completions.create.call_args.kwargs
                self.assertEqual(request["extra_body"]["reasoning_effort"], level)
                self.assertNotIn("reasoning_effort", request)
                self.assertEqual(record["turns"][0]["request"], request)
                clients[0].close.assert_called_once_with()
        self.assertEqual(advisor.QWEN_REASONING_EFFORT, "medium", "Overrides must not mutate defaults")

    def test_legacy_helper_omits_effort_but_none_is_not_a_valid_explicit_level(self):
        request = providers.build_qwen_request("system", "user")
        self.assertEqual(request["extra_body"], {"cache_prompt": False, "reasoning_format": "deepseek"})
        for value in (None, "", "high", "max", 1, False):
            with self.subTest(value=value), self.assertRaises(ValueError):
                providers.normalize_qwen_reasoning_effort(value)
        with self.assertRaises(ValueError):
            providers.build_qwen_request("system", "user", reasoning_effort="high")

    def test_reasoning_cli_override_and_validation_are_local_only(self):
        with patch.object(advisor, "main", return_value=7) as main:
            self.assertEqual(advisor.cli(["--provider", "qwen", "--qwen-reasoning-effort", " LOW "]), 7)
            main.assert_called_once_with(provider="qwen", qwen_reasoning_effort="low")
        for args in (
            ["--provider", "qwen", "--qwen-reasoning-effort", "high"],
            ["--provider", "gpt", "--qwen-reasoning-effort", "medium"],
            ["--provider", "qwen", "--qwen-reasoning-effort", "low", "--check-response", "unused.json"],
            ["--provider", "gpt", "--qwen-reasoning-effort", "low", "--check-response", "unused.json"],
        ):
            with self.subTest(args=args), patch.object(advisor, "main") as main:
                with patch.object(advisor, "check_response_log") as check, redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as failure:
                        advisor.cli(args)
                self.assertEqual(failure.exception.code, 2)
                main.assert_not_called()
                check.assert_not_called()
        self.assertEqual(advisor.QWEN_REASONING_EFFORT, "medium")

    def test_invalid_editable_reasoning_fails_before_configuration_or_sdk(self):
        with (
            patch.object(advisor, "QWEN_REASONING_EFFORT", "typo"),
            patch.object(advisor, "load_qwen_config", side_effect=AssertionError("Credentials forbidden")),
            patch.object(advisor, "create_client", side_effect=AssertionError("SDK forbidden")),
            patch.dict(sys.modules, {"openai": None}),
        ):
            with self.assertRaises(ValueError):
                advisor.main(provider="qwen")
            with self.assertRaises(ValueError):
                advisor.run_advisor("unused", "http://unused.invalid/v1", {}, provider="qwen")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
                advisor.cli(["--provider", "qwen"])
            self.assertEqual(failure.exception.code, 2)

    def test_qwen_effort_never_changes_gpt_requests(self):
        original = advisor.build_request("supplied observation", provider="gpt")
        with patch.object(advisor, "QWEN_REASONING_EFFORT", "invalid-for-qwen"):
            self.assertEqual(advisor.build_request("supplied observation", provider="gpt"), original)
        self.assertEqual(original, {
            "model": advisor.MODEL,
            "input": [{"role": "system", "content": advisor.SYSTEM_PROMPT},
                      {"role": "user", "content": "supplied observation"}],
            "max_output_tokens": advisor.MAX_OUTPUT_TOKENS, "store": True,
        })
        with self.assertRaises(ValueError):
            advisor.build_request("observation", provider="gpt", qwen_reasoning_effort="medium")


if __name__ == "__main__":
    unittest.main()
