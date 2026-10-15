"""Offline regression coverage for one-step advice and error diagnostics."""

import io
import json
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

    def make_sdk(self, *, response=None, constructor_error=None, request_error=None):
        """Provide the SDK interface without importing it or making requests."""
        sdk = ModuleType("openai")
        sdk.APIError = type("APIError", (Exception,), {})
        for name in ("APIConnectionError", "APIStatusError", "APITimeoutError"):
            setattr(sdk, name, type(name, (sdk.APIError,), {}))
        client = MagicMock()
        client.__enter__.return_value = client
        client.responses.create.return_value = response or SimpleNamespace(
            id="response-test", model=advisor.MODEL,
            status="completed", output_text="D",
        )
        client.responses.create.side_effect = request_error
        sdk.OpenAI = MagicMock(return_value=client, side_effect=constructor_error)
        return sdk, client

    def run_main(self, sdk):
        stdout, stderr = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as temporary:
            log_dir = Path(temporary) / "logs"
            with (
                patch.object(advisor, "LOG_DIR", log_dir),
                patch.object(advisor, "load_config", return_value=(self.API_KEY, "https://example.invalid/v1")),
                patch.dict(sys.modules, {"openai": sdk}),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = advisor.main()
            paths = list(log_dir.glob("*.json"))
            self.assertEqual(len(paths), 1)
            self.assertFalse(list(log_dir.glob("*.tmp")))
            raw_log = paths[0].read_text(encoding="utf-8")
            record = json.loads(raw_log)
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

    def test_success_preserves_single_letter_output_and_single_request(self):
        sdk, client = self.make_sdk()
        exit_code, record, stdout, stderr = self.run_main(sdk)
        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, "D\n")
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["action"], "D")
        self.assertEqual(record["output_text"], "D")
        self.assertIsNone(record["error"])
        self.assertIn("运行记录", stderr)
        client.responses.create.assert_called_once_with(**advisor.build_request(advisor.USER_PROMPT))
        self.assertEqual(sdk.OpenAI.call_args.kwargs["max_retries"], 0)

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
        sdk, _ = self.make_sdk(response=SimpleNamespace(status="incomplete", output_text="D"))
        exit_code, record, stdout, _ = self.run_main(sdk)
        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIsNone(record["action"])
        self.assertEqual(record["output_text"], "D")
        self.assertEqual(record["error"]["type"], "AdviceError")

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
