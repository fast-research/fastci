from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CompleteMockCITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="fastci-integration-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.output = Path(cls.temporary.name) / "run"
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                "-u",
                str(root / "examples/mock_ci.py"),
                "--out",
                str(cls.output),
            ],
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode:
            raise AssertionError(result.stdout + "\n" + result.stderr)
        cls.summary = json.loads((cls.output / "summary.json").read_text())
        cls.runs = {row["name"]: row for row in cls.summary["scenarios"]}

    def test_complete_lifecycle_uses_real_cli_and_observed_traces(self):
        self.assertEqual(self.summary["scenario_count"], 11)
        self.assertEqual(self.summary["catalog_size"], 13)
        self.assertEqual(self.summary["subprocess_attempts"], 97)
        self.assertTrue(all(self.summary["checks"].values()))
        commands = json.loads((self.output / "commands.json").read_text())
        self.assertEqual(
            {c["argv"][3] for c in commands},
            {"plan", "run", "ingest", "prepare-history", "replay"},
        )
        trace = Path(
            self.runs["bootstrap"]["outcomes"]["smoke_fast"]["attempts"][0][
                "trace_files"
            ][0]["path"]
        )
        data = json.loads(trace.read_text())
        self.assertTrue(data["profiles"][0]["samples"])
        self.assertTrue(
            any(frame["name"].endswith("forward") for frame in data["shared"]["frames"])
        )
        # A rejected MR never entered the fixture's stable branch.
        current = (self.output / "project/framework/distributed.py").read_text()
        self.assertIn("return sum(values) + 0", current)

    def test_history_drives_order_and_reductions_really_change_execution(self):
        row = self.runs["model_a"]
        self.assertEqual(row["order"][0], "distributed_tp")
        self.assertGreater(row["risk"]["distributed_tp"]["instability"], 0)
        self.assertEqual(row["risk"]["distributed_tp"]["alpha"], 0)
        self.assertIn("smoke_slow", row["affected"])
        self.assertNotIn("smoke_slow", row["order"])
        compiled = self.runs["compile_regression"]["outcomes"]["compiled_train"]
        self.assertEqual([a["status"] for a in compiled["attempts"]], ["fail", "pass"])
        self.assertEqual(compiled["status"], "fail")
        repaired = self.runs["compile_repair"]["outcomes"]["compiled_train"]
        self.assertEqual(repaired["workload_action"], "original")
        self.assertEqual(len(repaired["attempts"]), 1)
        self.assertEqual(repaired["status"], "pass")

    def test_graph_errors_and_failfast_preserve_missing_observations(self):
        row = self.runs["config_and_new_test"]
        self.assertIn("checkpoint_roundtrip", row["trace_errors"])
        self.assertIn("checkpoint_roundtrip", row["preserved_tests"])
        self.assertIn("new_cache", row["updated_traces"])
        skipped = self.runs["rollback_failfast"]["outcomes"]
        self.assertEqual(skipped["smoke_fast"]["status"], "skipped")
        self.assertEqual(skipped["smoke_fast"]["attempts"], [])
        history = json.loads((self.output / "observations.json").read_text())
        self.assertEqual(history[-1]["outcomes"], {"distributed_tp": "fail"})

    def test_mock_metrics_and_actual_reservation_intervals(self):
        self.assertIsNone(self.summary["measured_gpu_seconds"])
        observed_overlap = False
        for row in self.runs.values():
            active = []
            for outcome in row["outcomes"].values():
                if outcome["attempts"]:
                    active.append(outcome)
            for i, left in enumerate(active):
                for right in active[i + 1 :]:
                    overlaps = max(
                        left["started_seconds"], right["started_seconds"]
                    ) < min(left["finished_seconds"], right["finished_seconds"])
                    if overlaps:
                        observed_overlap = True
                        self.assertFalse(set(left["devices"]) & set(right["devices"]))
        self.assertTrue(observed_overlap)
        replay = json.loads((self.output / "replay.json").read_text())
        self.assertEqual(replay["resource_mode"], "mock")
        for row in replay["rows"]:
            self.assertNotIn("serial_prefix_gpu_seconds", row)
            self.assertIn("serial_prefix_mock_device_seconds", row)


if __name__ == "__main__":
    unittest.main()
