"""Offline validation and recovery tests; no SDK or credentials required."""
import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import run_trials as runner


class TrialTests(unittest.TestCase):
    def test_sdk_compatibility_without_network(self):
        def create(*, model, messages, max_tokens, output_config):
            raise AssertionError("Preflight must never call the SDK method.")
        cfg = runner.read_config(runner.ROOT / "config.example.json")
        trial = runner.load_trials(runner.ROOT / "examples/demo_trials.csv")[0]
        request = runner.build_request(trial, cfg, "prompt")
        self.assertNotIn("temperature", request)
        runner.validate_sdk_request(create, request)
        cfg["temperature"] = 0
        legacy_request = runner.build_request(trial, cfg, "prompt")
        self.assertEqual(legacy_request["temperature"], 0)
        with self.assertRaisesRegex(ValueError, "temperature"):
            runner.validate_sdk_request(create, legacy_request)

    def test_local_type_error_stops_without_retry(self):
        from unittest.mock import Mock
        cfg = runner.read_config(runner.ROOT / "config.example.json")
        trials = runner.load_trials(runner.ROOT / "examples/demo_trials.csv")[:2]
        create = Mock(side_effect=TypeError("unsupported argument"))
        client = SimpleNamespace(messages=SimpleNamespace(create=create))
        with tempfile.TemporaryDirectory(dir=runner.ROOT) as folder, \
                patch.object(runner.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "Local SDK/programming error"):
                runner.execute(Path(folder), {"trials": trials}, "hash", trials, cfg, "prompt", client)
            self.assertEqual(create.call_count, 1)
            sleep.assert_not_called()
            self.assertEqual(len((Path(folder) / "failures.jsonl").read_text().splitlines()), 1)

    def args(self, output):
        return argparse.Namespace(mode="mock", trials="examples/demo_trials.csv",
                                  config="config.example.json", prompt="prompt.txt",
                                  output=str(output), limit=2)

    def test_invalid_probabilities(self):
        for probabilities in ((True, 0, 0), (float("nan"), 0, 1), (-0.1, 0.1, 1),
                              (0.2, 0.2, 0.2), (float("inf"), 0, 0)):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                runner.validate_response(dict(zip(runner.PROBS, probabilities), explanation="test"), .001)
        runner.validate_response(dict(zip(runner.PROBS, (.1, .2, .7)), explanation="test"), .001)

    def test_resume_and_settings_guard(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT) as folder:
            output = Path(folder)
            args = self.args(output)
            self.assertEqual(runner.run(args), 0)
            before = (output / "results.csv").read_bytes()
            self.assertEqual(runner.run(args), 0)
            self.assertEqual(before, (output / "results.csv").read_bytes())
            settings = json.loads((output / "settings.json").read_text())
            settings["prompt"] = "changed"
            (output / "settings.json").write_text(json.dumps(settings))
            with self.assertRaisesRegex(ValueError, "changed"):
                runner.run(args)
            self.assertFalse((output / ".lock").exists())

    def test_retry_and_failure_recovery(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT) as folder:
            args = self.args(Path(folder))
            with patch.object(runner, "validate_response", side_effect=ValueError("bad response")), \
                    patch.object(runner.time, "sleep"):
                self.assertEqual(runner.run(args), 1)
            self.assertEqual(len((Path(folder) / "failures.jsonl").read_text().splitlines()), 6)
            self.assertEqual(runner.run(args), 0)
            with (Path(folder) / "results.csv").open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)

    def test_request_order_and_independence(self):
        cfg = runner.read_config(runner.ROOT / "config.example.json")
        trials = runner.load_trials(runner.ROOT / "examples/demo_trials.csv")
        first = runner.build_request(trials[0], cfg, "prompt")
        second = runner.build_request(trials[1], cfg, "prompt")
        for request in (first, second):
            self.assertEqual(len(request["messages"]), 1)
            self.assertEqual([b["text"] for b in request["messages"][0]["content"] if b["type"] == "text"],
                             ["Image A:", "Image B:", "Image C:", "prompt"])

    def test_live_adapter_with_fake_client(self):
        cfg = runner.read_config(runner.ROOT / "config.example.json")
        cfg["model"] = "test-pinned-model"
        trials = runner.load_trials(runner.ROOT / "examples/demo_trials.csv")[:1]
        value = dict(zip(runner.PROBS, (.2, .3, .5)), explanation="C differs from A and B.")
        response = SimpleNamespace(stop_reason="end_turn", model="test-pinned-model", _request_id="req-test",
                                   content=[SimpleNamespace(type="text", text=json.dumps(value))],
                                   usage=SimpleNamespace(input_tokens=100, output_tokens=50),
                                   model_dump=lambda **kwargs: {"content": value})
        from unittest.mock import Mock
        client = SimpleNamespace(messages=SimpleNamespace(create=Mock(return_value=response)))
        settings = {"trials": trials}
        with tempfile.TemporaryDirectory(dir=runner.ROOT) as folder:
            self.assertEqual(runner.execute(Path(folder), settings, "hash", trials, cfg, "prompt", client), 0)
            self.assertEqual(client.messages.create.call_count, 1)
            with (Path(folder) / "results.csv").open(newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["mode"], "live")
            self.assertEqual(row["request_id"], "req-test")

    def test_output_boundary(self):
        with self.assertRaises(ValueError):
            runner.writable_path("../outside-vlm")


if __name__ == "__main__":
    unittest.main()
