"""Offline coverage for inspecting stored Responses without generating advice."""

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


class ResponseDiagnosticTests(unittest.TestCase):
    API_KEY = "configured-diagnostic-secret"
    OTHER_KEY = "sk-unconfigured-diagnostic-secret"
    BASE_URL = "https://example.invalid/v1"
    RESPONSE_ID = "response-existing-first"

    def setUp(self):
        for target in ("socket.socket.connect", "socket.create_connection"):
            guard = patch(target, side_effect=AssertionError("Network access is forbidden in offline tests"))
            guard.start()
            self.addCleanup(guard.stop)

    def make_sdk(self):
        sdk = ModuleType("openai")
        sdk.APIError = type("APIError", (Exception,), {})
        for name in ("APIConnectionError", "APIStatusError", "APITimeoutError"):
            setattr(sdk, name, type(name, (sdk.APIError,), {}))
        client = MagicMock()
        client.__enter__.return_value = client
        client.responses.create.side_effect = AssertionError("Diagnostics must never create a response")
        client.responses.retrieve.return_value = SimpleNamespace(
            id=self.RESPONSE_ID, status="completed", model="gpt-5.5", store=True,
            created_at=1234567890, output_text="PRIVATE_RETRIEVED_ANSWER",
        )
        sdk.OpenAI = MagicMock(return_value=client)
        sdk.DefaultHttpxClient = MagicMock()
        return sdk, client

    def source_record(self, *, response_id=RESPONSE_ID, base_url=BASE_URL):
        return {
            "run_id": "existing-conversation",
            "base_url": base_url,
            "status": "error",
            "turns": [{
                "turn": 1,
                "status": "success",
                "request": {"model": "gpt-5.5", "store": True},
                "response": {"id": response_id, "status": "completed", "store": True},
                "action": "C",
            }, {
                "turn": 2,
                "status": "error",
                "response": None,
            }],
        }

    def run_check(self, sdk, *, source=None, save_hook=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source_path = directory / "source.json"
            original = json.dumps(source if source is not None else self.source_record()).encode("utf-8")
            source_path.write_bytes(original)
            log_dir = directory / "logs"
            with (
                patch.object(advisor, "LOG_DIR", log_dir),
                patch.object(advisor, "load_config", return_value=(self.API_KEY, self.BASE_URL)) as config,
                patch.dict(sys.modules, {"openai": sdk}),
                patch.object(advisor, "save_record", side_effect=save_hook or advisor.save_record),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = advisor.check_response_log(source_path)
            self.assertEqual(source_path.read_bytes(), original, "The original conversation log must remain untouched")
            paths = list((log_dir / "response_checks").glob("*.json"))
            records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
            all_output = stdout.getvalue() + stderr.getvalue() + json.dumps(records)
            self.assertNotIn(self.API_KEY, all_output)
            self.assertNotIn(self.OTHER_KEY, all_output)
            self.assertNotIn("PRIVATE_RETRIEVED_ANSWER", all_output)
            self.assertFalse(list(log_dir.rglob("*.tmp")))
        return exit_code, records, stdout.getvalue(), stderr.getvalue(), config

    def test_found_response_is_retrieved_once_without_generation(self):
        sdk, client = self.make_sdk()
        exit_code, records, _, _, _ = self.run_check(sdk)
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "found")
        client.responses.retrieve.assert_called_once_with(self.RESPONSE_ID)
        client.responses.create.assert_not_called()
        sdk.OpenAI.assert_called_once()
        self.assertEqual(sdk.OpenAI.call_args.kwargs["max_retries"], 0)
        client.__exit__.assert_called_once()

    def test_missing_response_id_does_not_load_credentials_or_create_client(self):
        for response_id in (None, "", "   ", "response/unsafe", "response\nunsafe"):
            with self.subTest(response_id=response_id):
                sdk, client = self.make_sdk()
                exit_code, _, _, _, config = self.run_check(
                    sdk, source=self.source_record(response_id=response_id),
                )
                self.assertNotEqual(exit_code, 0)
                config.assert_not_called()
                sdk.OpenAI.assert_not_called()
                client.responses.retrieve.assert_not_called()
                client.responses.create.assert_not_called()

    def test_partial_source_log_does_not_load_credentials_or_create_client(self):
        sources = ({}, {"turns": []}, {"turns": [None]}, {
            "base_url": self.BASE_URL,
            "turns": [{"status": "error", "response": {"id": self.RESPONSE_ID}}],
        })
        for source in sources:
            with self.subTest(source=source):
                sdk, client = self.make_sdk()
                exit_code, _, _, _, config = self.run_check(sdk, source=source)
                self.assertNotEqual(exit_code, 0)
                config.assert_not_called()
                sdk.OpenAI.assert_not_called()
                client.responses.retrieve.assert_not_called()
                client.responses.create.assert_not_called()

    def test_trailing_slash_on_saved_base_url_is_accepted(self):
        sdk, client = self.make_sdk()
        exit_code, records, _, _, _ = self.run_check(
            sdk, source=self.source_record(base_url=self.BASE_URL + "/"),
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(records[0]["status"], "found")
        client.responses.retrieve.assert_called_once_with(self.RESPONSE_ID)
        client.responses.create.assert_not_called()

    def test_different_retrieved_id_is_not_reported_as_found(self):
        sdk, client = self.make_sdk()
        client.responses.retrieve.return_value.id = "a-different-response"
        exit_code, records, _, _, _ = self.run_check(sdk)
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(len(records), 1)
        self.assertNotEqual(records[0]["status"], "found")
        client.responses.retrieve.assert_called_once_with(self.RESPONSE_ID)
        client.responses.create.assert_not_called()

    def test_query_plan_is_saved_before_retrieval(self):
        sdk, client = self.make_sdk()
        response = client.responses.retrieve.return_value

        def retrieve(response_id):
            paths = list((advisor.LOG_DIR / "response_checks").glob("*.json"))
            self.assertEqual(len(paths), 1)
            saved = json.loads(paths[0].read_text(encoding="utf-8"))
            self.assertIn(self.RESPONSE_ID, json.dumps(saved))
            self.assertNotEqual(saved["status"], "found")
            return response

        client.responses.retrieve.side_effect = retrieve
        exit_code, _, _, _, _ = self.run_check(sdk)
        self.assertEqual(exit_code, 0)
        client.responses.retrieve.assert_called_once_with(self.RESPONSE_ID)

    def test_different_base_url_prevents_all_requests(self):
        sdk, client = self.make_sdk()
        exit_code, _, _, _, _ = self.run_check(
            sdk, source=self.source_record(base_url="https://other.invalid/v1"),
        )
        self.assertNotEqual(exit_code, 0)
        sdk.OpenAI.assert_not_called()
        client.responses.retrieve.assert_not_called()
        client.responses.create.assert_not_called()

    def test_http_404_is_saved_without_retry_or_generation(self):
        sdk, client = self.make_sdk()
        error = sdk.APIStatusError(
            f"Stored response not found: {self.API_KEY}; {self.OTHER_KEY}"
        )
        error.status_code = 404
        error.code = "response_not_found"
        error.request_id = "diagnostic-request-id"
        client.responses.retrieve.side_effect = error
        exit_code, records, _, _, _ = self.run_check(sdk)
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(len(records), 1)
        self.assertNotEqual(records[0]["status"], "found")
        detail = records[0]["error"]["diagnostics"]["detail"]
        self.assertIn("Stored response not found", detail)
        self.assertIn("[REDACTED]", detail)
        client.responses.retrieve.assert_called_once_with(self.RESPONSE_ID)
        client.responses.create.assert_not_called()
        client.__exit__.assert_called_once()

    def test_connection_failure_is_saved_with_redacted_underlying_cause(self):
        sdk, client = self.make_sdk()
        error = sdk.APIConnectionError("Connection failed")
        error.__cause__ = OSError(f"TLS negotiation failed: {self.API_KEY}; {self.OTHER_KEY}")
        client.responses.retrieve.side_effect = error
        exit_code, records, _, _, _ = self.run_check(sdk)
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(len(records), 1)
        diagnostics = records[0]["error"]["diagnostics"]
        self.assertEqual(diagnostics["exception_type"], "APIConnectionError")
        cause, = diagnostics["causes"]
        self.assertIn("TLS negotiation failed", cause["detail"])
        self.assertIn("[REDACTED]", cause["detail"])
        client.responses.retrieve.assert_called_once_with(self.RESPONSE_ID)
        client.responses.create.assert_not_called()
        client.__exit__.assert_called_once()

    def test_initial_logging_failure_prevents_sdk_construction(self):
        sdk, client = self.make_sdk()

        def cannot_save(*args, **kwargs):
            raise OSError("simulated full disk")

        exit_code, records, _, _, _ = self.run_check(sdk, save_hook=cannot_save)
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(records, [])
        sdk.OpenAI.assert_not_called()
        client.responses.retrieve.assert_not_called()
        client.responses.create.assert_not_called()

    def test_cli_check_dispatch_never_runs_advice(self):
        with (
            patch.object(advisor, "main") as main,
            patch.object(advisor, "check_response_log", return_value=7) as check,
        ):
            self.assertEqual(advisor.cli(["--check-response", "existing.json"]), 7)
        main.assert_not_called()
        check.assert_called_once()
        self.assertEqual(str(check.call_args.args[0]), "existing.json")

    def test_cli_help_never_runs_advice_or_check(self):
        with (
            patch.object(advisor, "main") as main,
            patch.object(advisor, "check_response_log") as check,
            redirect_stdout(io.StringIO()),
        ):
            try:
                exit_code = advisor.cli(["--help"])
            except SystemExit as exc:
                exit_code = exc.code
        self.assertEqual(exit_code, 0)
        main.assert_not_called()
        check.assert_not_called()


if __name__ == "__main__":
    unittest.main()
