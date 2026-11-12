from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

from fastci.graph import Evidence, Graph
from fastci.megatron.common import seal
from fastci.megatron.planner import plan
from fastci.megatron.replay_selection import BenchmarkSelector
from fastci.megatron.runner import run
from fastci.model import TestSpec
from fastci.risk import RiskHistory
from fastci.selection import select
from fastci.source import Change, SourceIndex


class MegatronTests(unittest.TestCase):
    def setUp(self):
        self.catalog = seal(
            {
                "revision": "head",
                "config": {
                    "cases": ["a", "b", "c"],
                    "hardware": "fixture-gpu",
                    "environment_id": "fixture",
                    "python": sys.executable,
                },
                "tests": [
                    {
                        "id": name,
                        "case": name,
                        "validation_key": "same",
                        "fingerprint": name,
                        "launcher": "torchrun",
                    }
                    for name in ("a", "b", "c")
                ],
            }
        )
        graph = Graph(
            {
                "a": Evidence({"train.step"}, seconds=3),
                "b": Evidence({"train.step"}, seconds=1),
                "c": Evidence({"train.other"}, seconds=2),
            },
            "base",
        )
        self.evidence = seal(
            {
                "revision": "base",
                "environment_id": "fixture",
                "hardware": "fixture-gpu",
                "runtime_fingerprint": "runtime",
                "complete_baseline": True,
                "graph": graph.to_json(),
                "fingerprints": {"a": "a", "b": "b", "c": "old recipe"},
            }
        )

    def selection(self, mode, evidence):
        history = RiskHistory(recent={"c": [1]})
        index = SourceIndex({})
        change = Change("base", "head", ["train.py"], ["train.step"])
        with (
            patch(
                "fastci.megatron.planner.analyze", return_value=(change, index, index)
            ),
            patch("fastci.megatron.planner.git", return_value="base-tree"),
        ):
            return plan(".", self.catalog, "base", evidence, mode, history)

    def test_shadow_runs_baseline_and_protects_changed_recipe(self):
        result = self.selection("shadow", self.evidence)
        self.assertEqual(result["proposed"], ["c", "b"])
        self.assertEqual(result["execute"], ["a", "b", "c"])
        self.assertEqual(result["selection"]["groups"][0]["representative"], "b")

    def test_enforce_rechecks_runtime_before_omitting_jobs(self):
        result = self.selection("enforce", self.evidence)
        self.assertEqual(result["execute"], ["c", "b"])
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("fcntl.flock"),
            patch(
                "fastci.megatron.runner.open",
                mock_open(),
                create=True,
            ),
            patch(
                "fastci.megatron.runner.run_case",
                side_effect=lambda repo, revision, row, *_: {
                    "test": row["id"],
                    "status": "passed",
                    "gpu_seconds": 1,
                },
            ),
        ):
            observed = run(
                ".",
                result,
                Path(directory) / "run",
                environment_probe=lambda _: {
                    "fingerprint": "different",
                    "yq": "/usr/bin/yq",
                },
            )
        self.assertEqual([row["test"] for row in observed["outcomes"]], ["a", "b", "c"])
        self.assertIn(
            "runtime_environment_changed", observed["plan"]["fallback_reasons"]
        )

    def test_file_evidence_recovers_only_the_unresolved_file(self):
        index = SourceIndex(
            {
                "train.py": "def step(): pass",
                "setup.py": "def missed(): pass",
                "elsewhere.py": "def missed(): pass",
            }
        )
        graph = Graph(
            {
                "a": Evidence({"train.step"}),
                "b": Evidence({"setup.other"}),
                "c": Evidence({"other.unrelated"}),
            }
        )
        suite = {name: TestSpec(name) for name in graph.tests}
        change = Change(
            "base",
            "head",
            list(index.sources),
            ["train.step", "setup.missed", "elsewhere.missed"],
        )
        result = select(
            change,
            graph,
            suite,
            (index,),
            list(suite),
            file_jobs={"train.py": {"a"}, "setup.py": {"b"}},
        )
        self.assertEqual(result.retained, ["a", "b"])
        self.assertIn("setup.missed:file_evidence", result.fallbacks)
        self.assertIn("elsewhere.missed:no_expansion_evidence", result.fallbacks)

    def test_benchmark_prunes_context_then_adds_unpruned_file_evidence(self):
        graph = {
            "job_registry": {"first": 0, "second": 1, "other": 2},
            "node_coverage": {"m.observed": [0, 1], "m.other": [2]},
            "callees": {"root": {"m.observed": [0, 1], "m.other": [2]}},
        }
        source = "def observed(): return 1\ndef missed(): return 1\n"
        selector = BenchmarkSelector(graph, ["m.py"])
        decisions, evidence = selector.select(
            ["m.py"], {"m.py": source}, {"m.py": source.replace("1", "2")}
        )
        self.assertEqual(
            decisions,
            {
                "no_cep_no_file": ["first", "second"],
                "no_cep_file": ["first", "other", "second"],
                "cep_no_file": ["first"],
                "cep_file": ["first", "other", "second"],
            },
        )
        self.assertEqual(evidence["file_matches"], ["m.py"])

    def test_reproduce_archived_selection_table(self):
        root = Path(__file__).resolve().parents[1] / "experiments/megatron"
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                [sys.executable, str(root / "reproduce.py"), "--out", directory],
                cwd=directory,
                check=True,
                capture_output=True,
            )
            self.assertEqual(
                (Path(directory) / "paper-table.tsv").read_bytes(),
                (root / "data/expected-table.tsv").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
