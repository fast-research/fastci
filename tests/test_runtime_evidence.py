from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from fastci.cli import main
from fastci.graph import Evidence, Graph, GraphStore
from fastci.model import TestSpec, digest, read_json, suite_json, write_json
from fastci.provenance import validate_ingestion, verify_checkout
from fastci.runner import run_tests
from fastci.selection import select
from fastci.source import Change, SourceIndex, git
from fastci.traces import parse_speedscope
from fastci.workload import Proposal, WorkloadGuard

if __package__:
    from ._git_fixture import commit, init_repo
else:
    from _git_fixture import commit, init_repo


def fake_sampler(root):
    """Protocol fixture only: does not emulate statistical sampling or GPU work."""
    path = root / "sampler"
    path.write_text(
        "#!"
        + sys.executable
        + "\n"
        + """import json, os, sys
from pathlib import Path
output = Path(sys.argv[sys.argv.index('--output') + 1])
data = {'shared': {'frames': [{'file': os.environ['TRACE_SOURCE'], 'name': 'f', 'line': 1}]},
        'profiles': [{'type': 'sampled', 'unit': 'seconds', 'startValue': 0, 'endValue': 99, 'samples': [[0]]}]}
output.write_text(json.dumps(data))
output.with_suffix('.args.json').write_text(json.dumps(sys.argv[1:]))
"""
    )
    path.chmod(0o755)
    return path


def traced_program(assertions):
    return (
        "import os,time\nfrom pathlib import Path\n"
        + assertions
        + """
if os.environ.get('LOCAL_RANK', '0') == '0':
    deadline = time.monotonic() + 5
    directory = Path(os.environ['FASTCI_TRACE_DIR'])
    while not list(directory.glob('*.args.json')) and time.monotonic() < deadline:
        time.sleep(.01)
    assert list(directory.glob('*.args.json')), 'sampler protocol fixture did not start'
"""
    )


class EvidenceIntegrityTests(unittest.TestCase):
    def test_execution_duration_overrides_stale_catalog_estimate(self):
        suite = {
            name: TestSpec(
                name, gpus=1, seconds=seconds, metadata={"task_type": "train"}
            )
            for name, seconds in (("a", 100), ("b", 1))
        }
        graph = Graph(
            {
                "a": Evidence({"m.f"}, seconds=1, timing_source="execution"),
                "b": Evidence({"m.f"}, seconds=10, timing_source="execution"),
            }
        )
        result = select(
            Change("b", "h", ["m.py"], ["m.f"]), graph, suite, (), list(suite)
        )
        self.assertEqual(result.retained, ["a"])

    def test_corrupt_or_wrong_revision_graph_falls_back_with_diagnostics(self):
        with tempfile.TemporaryDirectory() as folder:
            store = GraphStore(folder)
            store.publish(Graph({"test": Evidence({"m.f"})}), "stable-rev", True)
            for content in (
                "{broken",
                json.dumps({"schema": 1, "tests": []}),
                json.dumps({"job_registry": [], "callees": {}}),
                json.dumps(Graph(revision="wrong").to_json()),
            ):
                store.path("requested").write_text(content)
                graph, origin = store.load("requested")
                self.assertEqual(origin, "stable")
                self.assertEqual(graph.revision, "stable-rev")
                self.assertTrue(store.load_errors)
            (Path(folder) / "stable.json").write_text("broken too")
            graph, origin = store.load("requested")
            self.assertEqual(origin, "missing")
            self.assertEqual(graph.tests, {})
            self.assertEqual(len(store.load_errors), 2)

    def test_structurally_invalid_profiles_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.json"
            for value in (
                {"shared": {"frames": [None]}, "profiles": []},
                {"shared": {"frames": [{"name": None}]}, "profiles": []},
                {"shared": {"frames": []}, "profiles": [None]},
            ):
                write_json(path, value)
                with self.assertRaises(ValueError):
                    parse_speedscope([path], "/repo", SourceIndex({}))
            write_json(
                path,
                {
                    "shared": {
                        "frames": [
                            {"name": "native", "file": None},
                            {"name": "f", "file": "/repo/m.py", "line": 1},
                        ]
                    },
                    "profiles": [
                        {
                            "type": "sampled",
                            "unit": "seconds",
                            "startValue": 0,
                            "endValue": 1,
                            "samples": [[0, 1]],
                        }
                    ],
                },
            )
            self.assertEqual(
                parse_speedscope([path], "/repo", SourceIndex({})).nodes, {"m.f"}
            )

    def test_ingestion_binds_revision_plan_snapshot_and_captured_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.json"
            path.write_text("fixture trace bytes")
            snapshot = Graph().to_json()
            plan = {
                "change": {"revision": "rev"},
                "order": ["test"],
                "base_snapshot": {"digest": digest(snapshot)},
            }
            run = {
                "revision": "rev",
                "source_verified": True,
                "plan_digest": digest(plan),
                "outcomes": [
                    {
                        "test": "test",
                        "status": "pass",
                        "attempts": [
                            {
                                "trace_files": [
                                    {
                                        "path": str(path),
                                        "sha256": hashlib.sha256(
                                            path.read_bytes()
                                        ).hexdigest(),
                                    }
                                ]
                            }
                        ],
                    }
                ],
            }
            accepted, errors = validate_ingestion(
                plan, run, snapshot, "rev", {"test": [str(path)]}, Path(folder)
            )
            self.assertEqual(set(accepted), {"test"})
            self.assertEqual(errors, {})
            second = Path(folder) / "second.json"
            second.write_text("second captured trace")
            complete = copy.deepcopy(run)
            complete["outcomes"][0]["attempts"][0]["trace_files"].append(
                {
                    "path": str(second),
                    "sha256": hashlib.sha256(second.read_bytes()).hexdigest(),
                }
            )
            self.assertIn(
                "test",
                validate_ingestion(
                    plan, complete, snapshot, "rev", {"test": [str(path)]}, Path(folder)
                )[1],
            )
            for field, value in (
                ("revision", "wrong"),
                ("plan_digest", "wrong"),
                ("source_verified", False),
            ):
                changed = {**run, field: value}
                with self.assertRaises(ValueError):
                    validate_ingestion(plan, changed, snapshot, "rev", {}, Path(folder))
            with self.assertRaises(ValueError):
                validate_ingestion(
                    plan,
                    run,
                    {**snapshot, "revision": "changed"},
                    "rev",
                    {},
                    Path(folder),
                )
            path.write_text("changed after the run")
            accepted, errors = validate_ingestion(
                plan,
                run,
                snapshot,
                "rev",
                {"test": [str(path)], "skipped": [str(path)]},
                Path(folder),
            )
            self.assertEqual(accepted, {})
            self.assertEqual(set(errors), {"test", "skipped"})
            incomplete = copy.deepcopy(run)
            incomplete["outcomes"][0]["status"] = "timeout"
            self.assertIn(
                "test",
                validate_ingestion(
                    plan,
                    incomplete,
                    snapshot,
                    "rev",
                    {"test": [str(path)]},
                    Path(folder),
                )[1],
            )


class SamplerTransportTests(unittest.TestCase):
    def test_packaged_hook_chains_startup_and_only_samples_rank_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "sitecustomize.py").write_text(
                "import os\nos.environ['APPLICATION_STARTED']='yes'\n"
            )
            target = root / "target.py"
            target.write_text("def f(): return 1\n")
            binary = fake_sampler(root)
            code = traced_program("assert os.environ['APPLICATION_STARTED']=='yes'\n")
            suite = {
                str(rank): TestSpec(
                    str(rank),
                    [sys.executable, "-c", code],
                    env={
                        "PYTHONPATH": str(root),
                        "LOCAL_RANK": str(rank),
                        "FASTCI_PYSPY": str(binary),
                        "TRACE_SOURCE": str(target),
                    },
                )
                for rank in (0, 1)
            }
            results = run_tests(
                list(suite),
                suite,
                {n: Proposal() for n in suite},
                WorkloadGuard(),
                "env",
                "rev",
                root,
                {},
                root / "logs",
                trace_dir=root / "traces",
                trace_rate=37,
                trace_subprocesses=True,
            )
            self.assertTrue(all(o.status == "pass" for o in results))
            zero = results[0].attempts[0]
            self.assertEqual(len(zero["trace_files"]), 1)
            path = Path(zero["trace_files"][0]["path"])
            argv = read_json(path.with_suffix(".args.json"))
            self.assertEqual(argv[argv.index("--rate") + 1], "37")
            self.assertIn("--subprocesses", argv)
            self.assertEqual(results[1].attempts[0]["trace_files"], [])
            self.assertTrue(results[1].attempts[0]["trace_errors"])

    def test_missing_sampler_does_not_change_test_result(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            test = TestSpec(
                "test",
                [sys.executable, "-c", "pass"],
                env={"FASTCI_PYSPY": str(root / "absent")},
            )
            results = run_tests(
                ["test"],
                {"test": test},
                {"test": Proposal()},
                WorkloadGuard(),
                "env",
                "rev",
                root,
                {},
                root / "logs",
                trace_dir=root / "traces",
            )
            self.assertEqual(results[0].status, "pass")
            self.assertEqual(results[0].attempts[0]["trace_files"], [])
            self.assertTrue(results[0].attempts[0]["trace_errors"])


class RuntimePolicyAndPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        init_repo(self.repo, branch="main", name="Audit fixture")
        (self.repo / ".gitignore").write_text("__pycache__/\n")
        (self.repo / "m.py").write_text("def f(): return 1\n")
        self.commit("base")
        self.base = git(self.repo, "rev-parse", "HEAD").strip()
        git(self.repo, "checkout", "-qb", "codex/change")
        (self.repo / "m.py").write_text("def f(): return 2\n")
        self.commit("change")
        self.revision = git(self.repo, "rev-parse", "HEAD").strip()
        self.store = GraphStore(self.root / "graphs")
        self.store.publish(
            Graph({"a": Evidence({"m.f"}), "b": Evidence({"other.f"})}), self.base, True
        )
        self.state, self.config = self.root / "state.json", self.root / "config.json"
        self.plan_path, self.run_path = self.root / "plan.json", self.root / "run.json"
        write_json(self.config, {"mode": "full"})

    def commit(self, message):
        commit(self.repo, message)

    def plan(self, code="pass", env=None):
        suite = {
            n: TestSpec(
                n,
                [sys.executable, "-c", code],
                env=env or {},
                metadata={"task_type": n},
            )
            for n in ("a", "b")
        }
        write_json(self.root / "suite.json", suite_json(suite))
        self.assertEqual(
            main(
                [
                    "plan",
                    "--repo",
                    str(self.repo),
                    "--target",
                    "main",
                    "--suite",
                    str(self.root / "suite.json"),
                    "--graphs",
                    str(self.store.root),
                    "--state",
                    str(self.state),
                    "--config",
                    str(self.config),
                    "--out",
                    str(self.plan_path),
                ]
            ),
            0,
        )

    def execute_plan(self, *args):
        return main(
            [
                "run",
                "--plan",
                str(self.plan_path),
                "--cwd",
                str(self.repo),
                "--state",
                str(self.state),
                "--environment",
                "fixture",
                "--run-id",
                "audit-run",
                "--out",
                str(self.run_path),
                *args,
            ]
        )

    def test_disabled_config_reroutes_existing_plan_without_touching_state(self):
        self.plan("import os;assert os.environ['FASTCI_TRACE']=='0'")
        self.assertEqual(read_json(self.plan_path)["order"], ["a"])
        write_json(self.config, {"mode": "disabled"})
        self.state.write_text("invalid state must not be read when disabled")
        Path(read_json(self.plan_path)["base_snapshot"]["path"]).unlink()
        self.assertEqual(self.execute_plan(), 0)
        result = read_json(self.run_path)
        self.assertEqual(result["effective_mode"], "disabled")
        self.assertEqual([o["test"] for o in result["outcomes"]], ["a", "b"])
        self.assertEqual(
            self.state.read_text(), "invalid state must not be read when disabled"
        )
        self.assertTrue(
            all(not a["trace_files"] for o in result["outcomes"] for a in o["attempts"])
        )
        self.assertEqual(
            main(
                [
                    "sample",
                    "--config",
                    str(self.config),
                    "--pid",
                    "-1",
                    "--out",
                    str(self.root / "unused"),
                ]
            ),
            0,
        )
        self.assertFalse((self.root / "unused").exists())
        self.plan_path = self.root / "disabled-plan.json"
        self.plan()
        self.assertEqual(read_json(self.plan_path)["mode"], "disabled")
        self.assertEqual(read_json(self.plan_path)["order"], ["a", "b"])
        self.assertEqual(
            main(
                [
                    "ingest",
                    "--config",
                    str(self.config),
                    "--snapshot",
                    "missing",
                    "--manifest",
                    "missing",
                    "--repo",
                    "missing",
                    "--revision",
                    "missing",
                    "--remote-root",
                    "/repo",
                    "--graphs",
                    str(self.store.root),
                    "--out",
                    str(self.root / "disabled-ingest.json"),
                ]
            ),
            0,
        )
        self.assertTrue(read_json(self.root / "disabled-ingest.json")["disabled"])

    def test_drain_prevents_new_optimized_plans_but_allows_admitted_run(self):
        self.plan()
        write_json(self.config, {"mode": "drain"})
        self.assertEqual(self.execute_plan("--no-trace"), 0)
        self.assertEqual(
            [o["test"] for o in read_json(self.run_path)["outcomes"]], ["a"]
        )
        before = self.state.read_bytes()
        self.plan_path, self.run_path = (
            self.root / "next-plan.json",
            self.root / "next-run.json",
        )
        self.plan()
        self.assertEqual(read_json(self.plan_path)["mode"], "drain")
        self.assertEqual(self.execute_plan(), 0)
        self.assertEqual(self.state.read_bytes(), before)
        self.assertEqual(
            [o["test"] for o in read_json(self.run_path)["outcomes"]], ["a", "b"]
        )

    def test_stable_publication_requires_merge_and_uses_execution_timing(self):
        binary = fake_sampler(self.root)
        self.plan(
            traced_program("import m\nassert m.f()==2\n"),
            {"FASTCI_PYSPY": str(binary), "TRACE_SOURCE": str(self.repo / "m.py")},
        )
        self.assertEqual(self.execute_plan(), 0)
        data = read_json(self.plan_path)
        args = [
            "ingest",
            "--repo",
            str(self.repo),
            "--revision",
            self.revision,
            "--snapshot",
            data["base_snapshot"]["path"],
            "--manifest",
            str(self.run_path.with_suffix(".traces.json")),
            "--plan",
            str(self.plan_path),
            "--run",
            str(self.run_path),
            "--remote-root",
            str(self.repo),
            "--graphs",
            str(self.store.root),
            "--stable",
            "--stable-ref",
            "main",
            "--out",
            str(self.root / "ingest.json"),
        ]
        self.assertEqual(main(args), 2)
        self.assertFalse(self.store.path(self.revision).exists())
        git(self.repo, "checkout", "-q", "main")
        git(self.repo, "merge", "--ff-only", "codex/change")
        self.assertEqual(main(args), 0)
        graph, _ = self.store.load(self.revision)
        outcome = read_json(self.run_path)["outcomes"][0]
        self.assertEqual(graph.tests["a"].seconds, outcome["attempts"][0]["seconds"])
        self.assertNotEqual(graph.tests["a"].seconds, 99)
        self.assertEqual(graph.tests["a"].timing_source, "execution")
        self.assertEqual(graph.tests["b"], self.store.load(self.base)[0].tests["b"])

    def test_squash_merge_publishes_under_integrated_commit_and_rejects_later_tree(
        self,
    ):
        binary = fake_sampler(self.root)
        self.plan(
            traced_program("import m\nassert m.f()==2\n"),
            {"FASTCI_PYSPY": str(binary), "TRACE_SOURCE": str(self.repo / "m.py")},
        )
        self.assertEqual(self.execute_plan(), 0)
        git(self.repo, "checkout", "-q", "main")
        git(self.repo, "merge", "--squash", "codex/change")
        self.commit("squashed integration")
        integrated = git(self.repo, "rev-parse", "HEAD").strip()
        self.assertNotEqual(integrated, self.revision)
        data = read_json(self.plan_path)
        args = [
            "ingest",
            "--repo",
            str(self.repo),
            "--revision",
            self.revision,
            "--snapshot",
            data["base_snapshot"]["path"],
            "--manifest",
            str(self.run_path.with_suffix(".traces.json")),
            "--plan",
            str(self.plan_path),
            "--run",
            str(self.run_path),
            "--remote-root",
            str(self.repo),
            "--graphs",
            str(self.store.root),
            "--stable",
            "--stable-ref",
            "main",
            "--out",
            str(self.root / "ingest.json"),
        ]
        self.assertEqual(main(args), 0)
        graph, origin = self.store.load(integrated)
        self.assertEqual(origin, "merge_base")
        self.assertEqual(graph.revision, integrated)
        before = (self.store.root / "stable.json").read_bytes()
        (self.repo / "new.py").write_text("def other(): return 1\n")
        self.commit("new untested integration tree")
        self.assertEqual(main(args), 2)
        self.assertEqual((self.store.root / "stable.json").read_bytes(), before)

    def test_non_fast_forward_merge_has_its_own_graph_version(self):
        binary = fake_sampler(self.root)
        self.plan(
            traced_program("import m\nassert m.f()==2\n"),
            {"FASTCI_PYSPY": str(binary), "TRACE_SOURCE": str(self.repo / "m.py")},
        )
        self.assertEqual(self.execute_plan(), 0)
        git(self.repo, "checkout", "-q", "main")
        git(self.repo, "merge", "--no-ff", "-m", "integration merge", "codex/change")
        integrated = git(self.repo, "rev-parse", "HEAD").strip()
        data = read_json(self.plan_path)
        args = [
            "ingest",
            "--repo",
            str(self.repo),
            "--revision",
            self.revision,
            "--snapshot",
            data["base_snapshot"]["path"],
            "--manifest",
            str(self.run_path.with_suffix(".traces.json")),
            "--plan",
            str(self.plan_path),
            "--run",
            str(self.run_path),
            "--remote-root",
            str(self.repo),
            "--graphs",
            str(self.store.root),
            "--stable",
            "--stable-ref",
            "main",
            "--out",
            str(self.root / "ingest.json"),
        ]
        self.assertEqual(main(args), 0)
        graph, origin = self.store.load(integrated)
        self.assertEqual((origin, graph.revision), ("merge_base", integrated))
        self.assertEqual(
            read_json(self.root / "ingest.json")["published_revision"], integrated
        )

    def test_source_change_between_paired_attempts_cannot_approve_reduction(self):
        test = TestSpec(
            "mutator",
            [
                sys.executable,
                "-c",
                "from pathlib import Path;Path('m.py').write_text('changed')",
            ],
            bindings={"size": {"kind": "env", "key": "SIZE"}},
        )
        guard = WorkloadGuard()
        with self.assertRaises(ValueError):
            run_tests(
                ["mutator"],
                {"mutator": test},
                {"mutator": Proposal({"size": 2}, {}, "policy")},
                guard,
                "env",
                self.revision,
                self.repo,
                {},
                self.root / "logs",
                verify_source=lambda: verify_checkout(self.repo, self.revision),
            )
        self.assertEqual(guard.entries, {})


if __name__ == "__main__":
    unittest.main()
