"""DeepSeek single/two-question tests using fake clients, with network disabled."""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

from main import GPT_onestep_test as advisor
from main import llm_provider as providers


def response(content="C", finish_reason="stop"):
    return {
        "model": providers.DEEPSEEK_MODEL,
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": content,
                                 "reasoning_content": "offline reasoning"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


class DeepSeekOneStepTests(unittest.TestCase):
    def setUp(self):
        for name in ("socket.socket.connect", "socket.create_connection"):
            guard = patch(name, side_effect=AssertionError("Network forbidden"))
            guard.start()
            self.addCleanup(guard.stop)

    def make_sdk(self, replies):
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
            reply = replies[len(clients)]
            if isinstance(reply, Exception):
                client.chat.completions.create.side_effect = reply
            else:
                client.chat.completions.create.return_value = reply
            clients.append(client)
            return client

        sdk.OpenAI = MagicMock(side_effect=new_client)
        sdk.DefaultHttpxClient = MagicMock()
        return sdk, clients

    def run_deepseek(self, replies, *, single=False):
        sdk, clients = self.make_sdk(replies)
        with tempfile.TemporaryDirectory() as directory:
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                patch.object(advisor, "LOG_DIR", Path(directory)),
                patch.object(advisor, "SINGLE_QUESTION", single),
                patch.object(advisor, "TRUST_ENV", True),
                patch.object(advisor, "QWEN_TRUST_ENV", False),
                patch.object(advisor, "load_deepseek_config", return_value=(
                    "offline-deepseek-key", "https://deepseek.invalid/v1")) as load_deepseek,
                patch.object(advisor, "load_config", side_effect=AssertionError("GPT credentials forbidden")),
                patch.object(advisor, "load_qwen_config", side_effect=AssertionError("Qwen credentials forbidden")),
                patch.dict(sys.modules, {"openai": sdk}),
                redirect_stdout(stdout), redirect_stderr(stderr),
            ):
                code = advisor.main(provider="deepseek")
            load_deepseek.assert_called_once_with()
            files = list(Path(directory).glob("*.json"))
            self.assertEqual(len(files), 1)
            saved = files[0].read_text(encoding="utf-8")
            self.assertNotIn("offline-deepseek-key", saved + stdout.getvalue() + stderr.getvalue())
            record = json.loads(saved)
        return code, record, clients, sdk, stdout.getvalue(), stderr.getvalue()

    def test_two_questions_are_independent_and_clients_closed_between_requests(self):
        code, record, clients, sdk, stdout, _ = self.run_deepseek([response("C"), response("D")])
        self.assertEqual(code, 0)
        self.assertEqual(record["completed_turns"], 2)
        self.assertEqual(record["provider"], "deepseek")
        self.assertEqual(record["conversation_mode"], "independent_turns")
        self.assertEqual(record["api_interface"], "chat.completions")
        self.assertIsNone(record["qwen_reasoning_effort"])
        self.assertEqual(Path(record["env_file"]).name, "Deepseek4.1flash")
        self.assertFalse(record["client_options"]["shared_across_turns"])
        self.assertTrue(record["client_options"]["trust_env"])
        self.assertEqual(len(clients), 2)
        for index, (client, prompt) in enumerate(zip(clients, [advisor.USER_PROMPT, advisor.USER_PROMPT_2])):
            client.close.assert_called_once_with()
            client.responses.create.assert_not_called()
            request = client.chat.completions.create.call_args.kwargs
            self.assertEqual(request, {
                "model": providers.DEEPSEEK_MODEL,
                "messages": [{"role": "system", "content": advisor.SYSTEM_PROMPT},
                             {"role": "user", "content": prompt}],
                "max_tokens": advisor.MAX_OUTPUT_TOKENS,
                "stream": False,
            })
            self.assertEqual(record["turns"][index]["request"], request)
            self.assertEqual(record["turns"][index]["response"]["choices"][0]["message"]["reasoning_content"],
                             "offline reasoning")
        self.assertIn("第 1 轮建议：C", stdout)
        self.assertIn("第 2 轮建议：D", stdout)
        for call in sdk.DefaultHttpxClient.call_args_list:
            self.assertEqual(call.kwargs, {"trust_env": True, "verify": True})
        for call in sdk.OpenAI.call_args_list:
            self.assertEqual(call.kwargs["api_key"], "offline-deepseek-key")
            self.assertEqual(call.kwargs["base_url"], "https://deepseek.invalid/v1")
            self.assertEqual(call.kwargs["max_retries"], 0)

    def test_single_question_needs_no_saved_response_id(self):
        code, record, clients, _, _, _ = self.run_deepseek([response()], single=True)
        self.assertEqual(code, 0)
        self.assertEqual(record["planned_turns"], 1)
        self.assertEqual(record["completed_turns"], 1)
        self.assertEqual(len(clients), 1)
        clients[0].close.assert_called_once_with()

    def test_truncated_response_is_preserved_and_second_question_not_sent(self):
        code, record, clients, _, stdout, stderr = self.run_deepseek([response("C", "length")])
        self.assertEqual(code, 1)
        self.assertEqual(record["completed_turns"], 0)
        self.assertEqual(record["turns"][0]["status"], "error")
        self.assertEqual(record["response"]["choices"][0]["finish_reason"], "length")
        self.assertIsNone(record["action"])
        self.assertEqual(len(clients), 1)
        clients[0].close.assert_called_once_with()
        self.assertEqual(stdout, "")
        self.assertIn("max_tokens", stderr)

    def test_failure_after_first_question_preserves_it_and_redacts_error(self):
        code, record, clients, _, stdout, stderr = self.run_deepseek([
            response("D"), RuntimeError("offline failure: offline-deepseek-key"),
        ])
        self.assertEqual(code, 1)
        self.assertEqual(record["completed_turns"], 1)
        self.assertEqual(record["turns"][0]["action"], "D")
        self.assertEqual(record["turns"][1]["status"], "error")
        self.assertEqual(len(clients), 2)
        for client in clients:
            client.close.assert_called_once_with()
            client.chat.completions.create.assert_called_once()
        self.assertIn("第 1 轮建议：D", stdout)
        self.assertNotIn("第 2 轮建议", stdout)
        self.assertIn("[REDACTED]", stderr)

    def test_external_client_is_never_reused(self):
        sdk, clients = self.make_sdk([response("D")])
        external = MagicMock()
        with patch.dict(sys.modules, {"openai": sdk}):
            action = advisor.run_advisor("offline-key", "https://deepseek.invalid/v1", {},
                                         provider="deepseek", client=external)
        self.assertEqual(action, "D")
        self.assertEqual(external.mock_calls, [])
        clients[0].close.assert_called_once_with()

    def test_cli_routes_deepseek_and_rejects_incompatible_options_before_requests(self):
        with patch.object(advisor, "main", return_value=7) as main:
            self.assertEqual(advisor.cli(["--provider", "deepseek"]), 7)
            main.assert_called_once_with(provider="deepseek", qwen_reasoning_effort=None)
        for options in (["--check-response", "unused.json"], ["--qwen-reasoning-effort", "medium"]):
            with self.subTest(options=options), patch.object(advisor, "main") as main:
                with patch.object(advisor, "check_response_log") as check, redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as failure:
                        advisor.cli(["--provider", "deepseek", *options])
                    self.assertEqual(failure.exception.code, 2)
                main.assert_not_called()
                check.assert_not_called()
        with self.assertRaisesRegex(ValueError, "previous_response_id"):
            advisor.build_request("observation", provider="deepseek", previous_response_id="old")
        with self.assertRaisesRegex(ValueError, "Qwen"):
            advisor.build_request("observation", provider="deepseek", qwen_reasoning_effort="medium")

    def test_qwen_settings_do_not_change_deepseek_requests(self):
        expected = advisor.build_request("observation", provider="deepseek")
        with patch.object(advisor, "QWEN_REASONING_EFFORT", "not-a-valid-level"):
            self.assertEqual(advisor.build_request("observation", provider="deepseek"), expected)
        self.assertNotIn("extra_body", expected)
        self.assertNotIn("reasoning_effort", expected)

    def test_http_error_diagnostics_describe_chat_interface(self):
        sdk, _ = self.make_sdk([])
        error = sdk.APIStatusError("offline HTTP failure")
        error.status_code = 400
        sdk, clients = self.make_sdk([error])
        # Make the injected exception a member of this fake SDK's exception hierarchy.
        sdk.APIStatusError = type(error)
        with patch.dict(sys.modules, {"openai": sdk}):
            with self.assertRaises(advisor.RunError) as failure:
                advisor.run_advisor("offline-key", "https://deepseek.invalid/v1", {}, provider="deepseek")
        self.assertIn("DeepSeek", str(failure.exception))
        self.assertIn("Chat Completions", str(failure.exception))
        self.assertNotIn("previous_response_id", str(failure.exception))
        self.assertEqual(failure.exception.http_status, 400)
        clients[0].close.assert_called_once_with()

    def test_config_with_only_gpt_key_uses_4router_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Deepseek4.1flash"
            path.write_text('\ufeffexport GPT_API_KEY="offline-deepseek-key" # token\n', encoding="utf-8")
            with patch.object(providers, "provider_env_file", return_value=path) as select:
                self.assertEqual(providers.load_deepseek_config(),
                                 ("offline-deepseek-key", "https://4router.net/v1"))
            select.assert_called_once_with("deepseek")

    def test_config_uses_deepseek_key_and_explicit_base_url_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Deepseek4.1flash"
            for url_key in ("base_url", "DEEPSEEK_BASE_URL", "GPT_BASE_URL", "OPENAI_BASE_URL"):
                with self.subTest(url_key=url_key):
                    path.write_text(
                        "GPT_API_KEY=offline-old-key\nDEEPSEEK_API_KEY=offline-selected-key\n"
                        f'{url_key}="https://override.invalid/v1/"\n', encoding="utf-8")
                    with patch.object(providers, "provider_env_file", return_value=path):
                        self.assertEqual(providers.load_deepseek_config(),
                                         ("offline-selected-key", "https://override.invalid/v1"))

    def test_config_never_falls_back_to_environment_or_other_provider_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Deepseek4.1flash"
            (Path(directory) / "gpt.env").write_text("GPT_API_KEY=offline-other-provider\n", encoding="utf-8")
            with (
                patch.object(providers, "provider_env_file", return_value=path) as select,
                patch.dict(os.environ, {
                    "GPT_API_KEY": "offline-process-key", "DEEPSEEK_API_KEY": "offline-process-key",
                    "OPENAI_API_KEY": "offline-process-key", "base_url": "https://wrong.invalid/v1",
                    "DEEPSEEK_BASE_URL": "https://wrong.invalid/v1",
                }),
            ):
                with self.assertRaises(providers.ConfigError):
                    providers.load_deepseek_config()
                path.write_text("# no key\n", encoding="utf-8")
                with self.assertRaises(providers.ConfigError):
                    providers.load_deepseek_config()
                path.write_text("GPT_API_KEY=offline-file-key\n", encoding="utf-8")
                self.assertEqual(providers.load_deepseek_config(),
                                 ("offline-file-key", "https://4router.net/v1"))
            self.assertEqual([call.args for call in select.call_args_list], [("deepseek",)] * 3)

    def test_invalid_configuration_errors_do_not_expose_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Deepseek4.1flash"
            malformed = (
                "base_url=https://override.invalid/v1\n",
                "GPT_API_KEY=offline-secret\nbase_url=https://user:offline-password@override.invalid/v1\n",
                "GPT_API_KEY=offline-secret\nbase_url=https://override.invalid:offline-password/v1\n",
                'GPT_API_KEY="offline-secret\n',
                'GPT_API_KEY="offline-secret with-whitespace"\n',
            )
            for content in malformed:
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with patch.object(providers, "provider_env_file", return_value=path):
                        with self.assertRaises(providers.ConfigError) as failure:
                            providers.load_deepseek_config()
                    self.assertNotIn("offline-secret", str(failure.exception))
                    self.assertNotIn("offline-password", str(failure.exception))


if __name__ == "__main__":
    unittest.main()
