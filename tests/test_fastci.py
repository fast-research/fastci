from __future__ import annotations

import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

from fastci.cli import main
from fastci.engine import plan
from fastci.graph import Evidence, Graph, GraphStore
from fastci.history_csv import prepare_history
from fastci.model import TestSpec, read_json, suite_json, write_json
from fastci.replay import scheduling_replay
from fastci.risk import RiskHistory
from fastci.runner import report, run_tests
from fastci.selection import select
from fastci.source import Change, SourceIndex, analyze, changed_entities, git
from fastci.traces import parse_speedscope
from fastci.workload import Proposal, WorkloadGuard, invocation, propose

if __package__:
    from ._git_fixture import commit, init_repo
else:
    from _git_fixture import commit, init_repo


def spec(name: str, seconds: float = 10, **kwargs) -> TestSpec:
    return TestSpec(
        name, seconds=seconds, metadata={"task_type": "distributed"}, **kwargs
    )


def evidence(*edges: tuple[str, str], nodes=()) -> Evidence:
    return Evidence(set(nodes) | {n for edge in edges for n in edge}, set(edges), 10)


class GraphSelectionTests(unittest.TestCase):
    def setUp(self):
        self.index = SourceIndex(
            {"m.py": "def f():\n return 1\n\ndef g():\n return f()\n"}
        )
        self.change = Change("base", "head", ["m.py"], ["m.f"])
        self.graph = Graph(
            {"a": evidence(("m.g", "m.f")), "b": evidence(("m.g", "m.f"))}
        )
        self.suite = {
            "a": spec("a", 20, gpus=2, hardware="gpu"),
            "b": spec("b", 10, gpus=2, hardware="gpu"),
        }

    def selected(self, **kwargs):
        return select(
            self.change,
            self.graph,
            self.suite,
            (self.index,),
            list(self.suite),
            **kwargs,
        )

    def test_lowest_gpu_cost_representative(self):
        selected = self.selected()
        self.assertEqual(selected.affected, ["a", "b"])
        self.assertEqual(selected.retained, ["b"])

    def test_metadata_and_hardware_prevent_pruning(self):
        for field, value in (
            ("metadata", {"task_type": "bitwise"}),
            ("hardware", "other"),
            ("gpus", 1),
        ):
            with self.subTest(field=field):
                original = getattr(self.suite["a"], field)
                setattr(self.suite["a"], field, value)
                self.assertEqual(self.selected().retained, ["a", "b"])
                setattr(self.suite["a"], field, original)

    def test_unknown_scope_cannot_prove_equivalence(self):
        self.suite["a"].metadata = {}
        self.suite["b"].metadata = {}
        self.assertEqual(self.selected().retained, ["a", "b"])

    def test_exact_edges_not_just_bfs_levels(self):
        self.graph.tests["a"] = evidence(("m.f", "x"), ("m.f", "y"), ("x", "z"))
        self.graph.tests["b"] = evidence(("m.f", "x"), ("m.f", "y"), ("y", "z"))
        self.assertEqual(self.selected().retained, ["a", "b"])

    def test_k3_is_default_and_k2_is_ablation(self):
        common = [("m.f", "x"), ("x", "y")]
        self.graph.tests["a"] = evidence(*common, ("y", "z"))
        self.graph.tests["b"] = evidence(*common, ("y", "w"))
        self.assertEqual(self.selected().retained, ["a", "b"])
        self.assertEqual(self.selected(depth=2).retained, ["b"])

    def test_pruning_is_per_modified_entity_then_union(self):
        self.graph.tests["a"].nodes.add("m.g2")
        self.change.entities.append("m.g2")
        self.assertEqual(self.selected().retained, ["a", "b"])

    def test_sibling_expansion_not_fabricated_context(self):
        self.index = SourceIndex(
            {"m.py": "class C:\n def missed(self): pass\n def run(self): pass\n"}
        )
        self.change.entities = ["m.C.missed"]
        for e in self.graph.tests.values():
            e.nodes = {"m.C.run"}
        self.assertEqual(self.selected().retained, ["a", "b"])

    def test_static_caller_expansion(self):
        self.change.entities = ["m.f"]
        self.graph.tests["a"].nodes = {"m.g"}
        self.graph.tests["b"].nodes = {"unrelated"}
        self.assertEqual(self.selected().retained, ["a"])

    def test_unknown_entity_and_untraced_test_fallback(self):
        self.change.entities = ["new.f"]
        self.assertEqual(self.selected().retained, ["a", "b"])
        self.change.entities = ["m.f"]
        self.graph.tests.pop("a")
        self.assertEqual(self.selected().retained, ["a", "b"])

    def test_nonpython_uses_existing_baseline_rules(self):
        self.change.files = ["job.yaml"]
        self.change.entities = []
        result = select(self.change, self.graph, self.suite, (), ["a"])
        self.assertEqual(result.retained, ["a"])

    def test_disabled_and_select_ablation(self):
        history = RiskHistory(recent={"a": [1], "b": [0]})
        result = plan(
            self.change,
            self.graph,
            self.suite,
            (self.index,),
            history,
            {},
            mode="disabled",
        )
        self.assertEqual(result["order"], ["a", "b"])
        self.assertFalse(any(p["overrides"] for p in result["proposals"].values()))
        result = plan(
            self.change,
            self.graph,
            self.suite,
            (self.index,),
            history,
            {},
            mode="select",
        )
        self.assertEqual(result["order"], ["b"])

    def test_graph_update_replaces_only_successful_tests(self):
        newer = self.graph.replacing({"a": evidence(nodes=["new.node"])}, "head")
        self.assertNotIn("m.f", newer.tests["a"].nodes)
        self.assertIn("m.f", newer.tests["b"].nodes)
        self.assertIn("m.f", self.graph.tests["a"].nodes)

    def test_versioned_store_falls_back_stable(self):
        with tempfile.TemporaryDirectory() as root:
            store = GraphStore(root)
            store.publish(self.graph, "stable-rev", True)
            store.publish(Graph(), "specific")
            self.assertEqual(store.load("specific")[1], "merge_base")
            graph, origin = store.load("absent")
            self.assertEqual(origin, "stable")
            self.assertEqual(graph.revision, "stable-rev")

    def test_legacy_graph_import_preserves_isolated_nodes(self):
        graph = Graph.from_json(
            {
                "job_registry": {"test": 4},
                "execution_times": {"test": 5},
                "node_coverage": {"root": [4], "m.C.f": [4]},
                "callees": {"root": {"m.C.f": [4]}},
            }
        )
        self.assertEqual(graph.tests["test"].nodes, {"m.C.f"})
        self.assertEqual(graph.tests["test"].edges, set())
        self.assertEqual(Graph.from_json(graph.to_json()).tests, graph.tests)


class SourceTests(unittest.TestCase):
    def test_globals_imports_and_class_body_expand_all_file_entities(self):
        for before, after in (
            ("X=1", "X=2"),
            ("import a", "import b"),
            ("class C:\n x=1", "class C:\n x=2"),
        ):
            suffix = "\ndef used(): return 1\ndef unused(): return 2\n"
            old, new = (
                SourceIndex({"m.py": before + suffix}),
                SourceIndex({"m.py": after + suffix}),
            )
            self.assertEqual(changed_entities("m.py", old, new), {"m.used", "m.unused"})

    def test_nested_scope_and_decorator_mapping(self):
        old = SourceIndex(
            {
                "m.py": "class C:\n @a\n async def f(self):\n  def inner():\n   return 1\n  return inner()\n"
            }
        )
        new = SourceIndex({"m.py": old.sources["m.py"].replace("return 1", "return 2")})
        self.assertEqual(changed_entities("m.py", old, new), {"m.C.f.inner"})
        new = SourceIndex({"m.py": old.sources["m.py"].replace("@a", "@b")})
        self.assertEqual(changed_entities("m.py", old, new), {"m.C.f"})

    def test_alias_static_call_and_deleted_function(self):
        index = SourceIndex(
            {
                "m.py": "def f(): pass\n",
                "n.py": "from m import f as alias\ndef g(): alias()\n",
            }
        )
        self.assertIn("n.g", index.expand("m.f"))
        new = SourceIndex({"m.py": ""})
        self.assertEqual(changed_entities("m.py", index, new), {"m.f"})

    def test_git_uses_merge_base_old_and_new_without_checkout(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            init_repo(root)
            (root / "m.py").write_text("def removed(): return 1\n")
            commit(root, "base")
            base = git(root, "rev-parse", "HEAD").strip()
            (root / "m.py").rename(root / "n.py")
            (root / "n.py").write_text("def added(): return 2\n")
            commit(root, "rename")
            head = git(root, "rev-parse", "HEAD").strip()
            (root / "n.py").write_text("invalid local working copy")
            change, old, new = analyze(root, "HEAD", base)
            self.assertEqual(change.merge_base, base)
            self.assertEqual(set(change.entities), {"m.removed", "n.added"})
            self.assertEqual(git(root, "rev-parse", "HEAD").strip(), head)
            self.assertIn("n.added", new.functions)


class TraceTests(unittest.TestCase):
    def trace(self):
        return {
            "shared": {
                "frames": [
                    {"file": "/repo/m.py", "name": "run", "line": 3},
                    {"file": "/external/x.py", "name": "wrapper", "line": 1},
                    {"file": "/repo/m.py", "name": "leaf", "line": 5},
                    {"file": "/repo-evil/m.py", "name": "fake", "line": 5},
                ]
            },
            "profiles": [
                {
                    "type": "sampled",
                    "unit": "milliseconds",
                    "startValue": 0,
                    "endValue": 2000,
                    "samples": [[0, 1, 2, 3]],
                    "weights": [1],
                }
            ],
        }

    def test_class_resolution_filtering_and_seconds(self):
        index = SourceIndex(
            {
                "m.py": "class C:\n def run(self):\n  return self.leaf()\n def leaf(self):\n  pass\n"
            }
        )
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trace.json"
            write_json(path, self.trace())
            e = parse_speedscope([path], "/repo", index)
            self.assertEqual(e.nodes, {"m.C.run", "m.C.leaf"})
            self.assertEqual(e.edges, {("m.C.run", "m.C.leaf")})
            self.assertEqual(e.seconds, 2)

    def test_invalid_empty_or_partial_trace_does_not_replace(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / "trace.json"
            for mutate in (
                lambda d: d.update(profiles=[]),
                lambda d: d["profiles"][0].update(samples=[[99]]),
                lambda d: d["profiles"][0].update(samples=[]),
            ):
                data = self.trace()
                mutate(data)
                write_json(p, data)
                with self.assertRaises(ValueError):
                    parse_speedscope([p], "/repo", SourceIndex({}))

    def test_overlapping_profiles_do_not_sum_wall_time(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / "trace.json"
            data = self.trace()
            data["profiles"].append(copy.deepcopy(data["profiles"][0]))
            write_json(p, data)
            self.assertEqual(parse_speedscope([p], "/repo", SourceIndex({})).seconds, 2)


class RiskTests(unittest.TestCase):
    def test_paper_equations_by_hand(self):
        h = RiskHistory(associations={"m.py": {"a": [10, 5]}}, recent={"a": [1, 0, 0]})
        r = h.score("a", ["m.py"])
        instability = (0.8**2 / (0.8**2 + 0.8 + 1)) * 0.85
        self.assertAlmostEqual(r["code"], 0.5)
        self.assertAlmostEqual(r["alpha"], 10 / 15)
        self.assertAlmostEqual(r["total"], (10 / 15) * 0.5 + (5 / 15) * instability)

    def test_no_positive_association_uses_only_instability(self):
        h = RiskHistory(associations={"m.py": {"a": [100, 0]}}, recent={"a": [1]})
        self.assertEqual(h.score("a", ["m.py"])["total"], 1)
        self.assertEqual(h.score("new", ["m.py"])["total"], 0)

    def test_pass_attenuation_window_and_ties(self):
        h = RiskHistory(recent={"a": [1, 0, 0, 0, 0]})
        self.assertAlmostEqual(
            h.score("a", [])["instability"],
            0.8**4 / sum(0.8**i for i in range(5)) * 0.6,
        )
        self.assertEqual(RiskHistory().order(["z", "a"], [])[0], ["z", "a"])

    def test_update_only_observed_once_and_no_premature_mutation(self):
        h = RiskHistory()
        h.score("a", ["m.py"])
        self.assertEqual(h.associations, {})
        h.observe(
            "run", ["m.py", "m.py"], {"a": "fail", "b": "skipped", "c": "timeout"}
        )
        self.assertEqual(h.associations, {"m.py": {"a": [1, 1]}})
        with self.assertRaises(ValueError):
            h.observe("run", [], {})

    def test_replay_no_future_leakage_and_nonfailure_warmup(self):
        rows = [
            {
                "sequence": 2,
                "run_id": "2",
                "files": ["m.py"],
                "tests": ["a", "b"],
                "outcomes": {"a": "pass", "b": "fail"},
            },
            {
                "sequence": 1,
                "run_id": "1",
                "files": ["m.py"],
                "tests": ["a", "b"],
                "outcomes": {"a": "pass", "b": "pass"},
            },
        ]
        result = scheduling_replay(rows, 1)
        risk = result["rows"][1]
        self.assertEqual(risk["order"], ["a", "b"])
        self.assertEqual(risk["scores"]["b"]["total"], 0)
        self.assertIsNone(risk["serial_prefix_gpu_seconds"])
        rows.append({**rows[0], "run_id": "3", "sequence": 3})
        self.assertEqual(scheduling_replay(rows, 1)["rows"][3]["order"], ["b", "a"])


class WorkloadTests(unittest.TestCase):
    def setUp(self):
        self.test = TestSpec(
            "a",
            command=["python", "train.py", "--layers", "12"],
            metadata={
                "task_type": "distributed",
                "validation_dimensions": ["parallelism"],
                "unrelated_dimensions": ["layers", "logging"],
            },
            workload={
                "layers": 12,
                "sequence": 8192,
                "parallelism": 8,
                "logging": True,
            },
            bindings={
                "layers": {"kind": "arg", "key": "--layers"},
                "sequence": {"kind": "env", "key": "SEQ"},
                "parallelism": {"kind": "env", "key": "TP"},
                "logging": {"kind": "env", "key": "LOGGING"},
            },
        )
        self.rules = {
            "reductions": {
                "layers": 2,
                "sequence": 128,
                "parallelism": 1,
                "logging": False,
            }
        }

    def test_explicitly_unrelated_only_and_runner_overrides(self):
        proposal = propose(self.test, Graph(), self.rules)
        self.assertEqual(proposal.overrides, {"layers": 2, "logging": False})
        argv, env = invocation(self.test, proposal.overrides)
        self.assertEqual(argv, ["python", "train.py", "--layers", "2"])
        self.assertEqual(env, {"LOGGING": "false"})
        self.assertEqual(self.test.command[-1], "12")

    def test_runtime_path_protects_metadata_mistake(self):
        graph = Graph({"a": evidence(nodes=["m.check_layers"])})
        self.rules["path_dimensions"] = {"*.check_layers": ["layers"]}
        self.assertNotIn("layers", propose(self.test, graph, self.rules).overrides)

    def test_initial_periodic_validation_and_revocation(self):
        p = propose(self.test, Graph(), self.rules)
        guard = WorkloadGuard(interval=3)
        self.assertEqual(guard.action("a", p, "env"), "paired")
        self.assertTrue(guard.compare("a", p, "env", "pass", "pass", "v1"))
        for _ in range(2):
            self.assertEqual(guard.action("a", p, "env"), "reduced")
            guard.used("a", p, "env")
        self.assertEqual(guard.action("a", p, "env"), "paired")
        self.assertFalse(guard.compare("a", p, "env", "fail", "pass", "v2"))
        self.assertEqual(guard.action("a", p, "env"), "original")
        self.assertEqual(guard.action("a", p, "different-env"), "paired")

    def test_errors_are_never_promoted_and_config_changes_revalidate(self):
        p = propose(self.test, Graph(), self.rules)
        guard = WorkloadGuard()
        self.assertFalse(guard.compare("a", p, "env", "timeout", "timeout", "v"))
        self.test.command.append("--new-validation")
        p2 = propose(self.test, Graph(), self.rules)
        self.assertEqual(guard.action("a", p2, "env"), "paired")


class RunnerTests(unittest.TestCase):
    def test_mock_resource_mode_never_reports_measured_gpu_work(self):
        from fastci.runner import Outcome

        outcome = Outcome(
            "fixture",
            "pass",
            [{"seconds": 3, "gpus": 2}],
            started_seconds=1,
            finished_seconds=4,
        )
        result = report([outcome], "mock")
        self.assertIsNone(result["measured_gpu_seconds"])
        self.assertEqual(result["mock_device_seconds"], 6)
        self.assertEqual(result["wall_seconds"], 4)
        with self.assertRaises(ValueError):
            report([], "unknown")

    def test_paired_mismatch_restores_baseline_and_counts_both_attempts(self):
        with tempfile.TemporaryDirectory() as folder:
            test = TestSpec(
                "a",
                [
                    sys.executable,
                    "-c",
                    "import os,sys;sys.exit(1 if os.getenv('SMALL') else 0)",
                ],
                bindings={"size": {"kind": "env", "key": "SMALL"}},
            )
            proposal = Proposal({"size": 1}, {}, "policy")
            guard = WorkloadGuard()
            outcomes = run_tests(
                ["a"],
                {"a": test},
                {"a": proposal},
                guard,
                "env",
                "v",
                folder,
                {},
                Path(folder) / "logs",
            )
            self.assertEqual(outcomes[0].status, "pass")
            self.assertEqual(
                [a["status"] for a in outcomes[0].attempts], ["pass", "fail"]
            )
            self.assertEqual(guard.action("a", proposal, "env"), "original")
            self.assertEqual(report(outcomes)["paired_tests"], 1)
            self.assertGreater(report(outcomes)["attempt_seconds"], 0)

    def test_resource_preflight_and_no_overallocation(self):
        with tempfile.TemporaryDirectory() as folder:
            # Synthetic devices exercise allocation only; these are CPU commands.
            command = [
                sys.executable,
                "-c",
                "import os,time;print(os.environ['CUDA_VISIBLE_DEVICES']);time.sleep(.03)",
            ]
            suite = {n: TestSpec(n, command, 1, "fake") for n in ("a", "b")}
            proposals = {n: Proposal() for n in suite}
            with self.assertRaises(ValueError):
                run_tests(
                    list(suite),
                    suite,
                    proposals,
                    WorkloadGuard(),
                    "e",
                    "v",
                    folder,
                    {},
                    Path(folder) / "bad",
                )
            outcomes = run_tests(
                list(suite),
                suite,
                proposals,
                WorkloadGuard(),
                "e",
                "v",
                folder,
                {"fake": ["synthetic-0"]},
                Path(folder) / "logs",
                workers=2,
            )
            self.assertTrue(all(o.status == "pass" for o in outcomes))
            self.assertGreater(outcomes[1].queued_seconds, outcomes[0].queued_seconds)
            self.assertEqual(
                Path(outcomes[0].attempts[0]["log"]).read_text().strip(), "synthetic-0"
            )

    def test_timeout_and_stop_pending_only(self):
        with tempfile.TemporaryDirectory() as folder:
            suite = {
                "a": TestSpec(
                    "a",
                    [sys.executable, "-c", "import time;time.sleep(10)"],
                    timeout=0.05,
                ),
                "b": TestSpec("b", [sys.executable, "-c", "print('not run')"]),
            }
            out = run_tests(
                list(suite),
                suite,
                {n: Proposal() for n in suite},
                WorkloadGuard(),
                "e",
                "v",
                folder,
                {},
                Path(folder) / "logs",
                workers=1,
                stop_on_failure=True,
            )
            self.assertEqual([o.status for o in out], ["timeout", "skipped"])


class CLITests(unittest.TestCase):
    def test_portable_csv_order_retries_and_unobserved_outcomes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "history.csv"
            rows = [
                {"run_id": "run2", "sequence": 2, "test": "a", "status": "pass"},
                {"run_id": "run1", "sequence": 1, "test": "a", "status": "fail"},
                {"run_id": "run1", "sequence": 1, "test": "a", "status": "pass"},
                {"run_id": "run1", "sequence": 1, "test": "b", "status": "timeout"},
                {"run_id": "run1", "sequence": 1, "test": "unknown", "status": "pass"},
            ]
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "run_id",
                        "sequence",
                        "test",
                        "status",
                        "files",
                        "seconds",
                    ],
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {**row, "files": json.dumps(["model.py"]), "seconds": "3"}
                    )
            records, diagnostics = prepare_history(
                path, {"a": spec("a", gpus=2), "b": spec("b")}
            )
            self.assertEqual([r["run_id"] for r in records], ["run1", "run2"])
            self.assertEqual(records[0]["outcomes"], {"a": "fail"})
            self.assertEqual(records[0]["seconds"], {"a": 3})
            self.assertEqual(records[0]["gpus"], {"a": 2})
            self.assertEqual(diagnostics["duplicate_test_observations"], 1)
            self.assertEqual(diagnostics["unobserved_outcomes"], 1)
            self.assertEqual(diagnostics["outside_suite"], 1)

    def test_portable_csv_rejects_inconsistent_metadata_and_invalid_cost(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "history.csv"
            for field, value in (
                ("sequence", "2"),
                ("files", "{}"),
                ("seconds", "nan"),
                ("status", "typo"),
            ):
                with self.subTest(field=field):
                    with path.open("w", newline="") as handle:
                        writer = csv.DictWriter(
                            handle,
                            fieldnames=[
                                "run_id",
                                "sequence",
                                "test",
                                "status",
                                "files",
                                "seconds",
                            ],
                        )
                        writer.writeheader()
                        row = {
                            "run_id": "r",
                            "sequence": 1,
                            "test": "a",
                            "status": "pass",
                            "files": '["m.py"]',
                            "seconds": "1",
                        }
                        writer.writerow(row)
                        writer.writerow({**row, field: value})
                    with self.assertRaises(ValueError):
                        prepare_history(path, {"a": spec("a")})

    def test_public_csv_and_json_examples_replay_identically(self):
        examples = Path(__file__).resolve().parents[1] / "examples"
        records, _ = prepare_history(
            examples / "history.csv",
            {
                "smoke": spec("smoke", gpus=1),
                "distributed": spec("distributed", gpus=2),
            },
        )
        self.assertEqual(
            scheduling_replay(records, 0),
            scheduling_replay(read_json(examples / "history.json"), 0),
        )

    def test_run_rejects_dirty_checkout_before_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            repo = root / "repo"
            repo.mkdir()
            init_repo(repo)
            (repo / "m.py").write_text("def f(): return 1\n")
            commit(repo, "base")
            base = git(repo, "rev-parse", "HEAD").strip()
            (repo / "m.py").write_text("def f(): return 2\n")
            commit(repo, "source")
            write_json(
                root / "suite.json",
                suite_json({"a": spec("a", command=[sys.executable, "-c", "pass"])}),
            )
            plan_args = [
                "plan",
                "--repo",
                str(repo),
                "--target",
                base,
                "--suite",
                str(root / "suite.json"),
                "--graphs",
                str(root / "graphs"),
                "--state",
                str(root / "state.json"),
                "--out",
                str(root / "plan.json"),
            ]
            self.assertEqual(main(plan_args), 0)
            (repo / "m.py").write_text("def f(): return 3\n")
            self.assertEqual(
                main(
                    [
                        "run",
                        "--plan",
                        str(root / "plan.json"),
                        "--cwd",
                        str(repo),
                        "--state",
                        str(root / "state.json"),
                        "--run-id",
                        "dirty",
                        "--environment",
                        "fixture",
                        "--out",
                        str(root / "run.json"),
                    ]
                ),
                2,
            )
            self.assertFalse((root / "run.json").exists())

    def test_plan_execute_history_and_frozen_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            repo = root / "repo"
            repo.mkdir()
            init_repo(repo)
            (repo / "m.py").write_text("def f(): return 1\n")
            commit(repo, "base")
            base = git(repo, "rev-parse", "HEAD").strip()
            (repo / "m.py").write_text("def f(): return 2\n")
            commit(repo, "head")
            write_json(
                root / "suite.json",
                suite_json(
                    {"a": spec("a", command=[sys.executable, "-c", "print('ok')"])}
                ),
            )
            GraphStore(root / "graphs").publish(
                Graph({"a": evidence(nodes=["m.f"])}), base
            )
            plan_path = root / "plan.json"
            args = [
                "plan",
                "--repo",
                str(repo),
                "--target",
                base,
                "--suite",
                str(root / "suite.json"),
                "--graphs",
                str(root / "graphs"),
                "--state",
                str(root / "state.json"),
                "--out",
                str(plan_path),
            ]
            self.assertEqual(main(args), 0)
            data = read_json(plan_path)
            self.assertEqual(data["order"], ["a"])
            self.assertEqual(data["graph_origin"], "merge_base")
            self.assertTrue(Path(data["base_snapshot"]["path"]).exists())
            run_args = [
                "run",
                "--plan",
                str(plan_path),
                "--cwd",
                str(repo),
                "--state",
                str(root / "state.json"),
                "--run-id",
                "run1",
                "--environment",
                "fixture",
                "--out",
                str(root / "result.json"),
            ]
            self.assertEqual(main(run_args), 0)
            self.assertEqual(
                read_json(root / "state.json")["history"]["recent"], {"a": [0]}
            )
            self.assertEqual(
                main(run_args), 2
            )  # duplicates cannot leak into future risk

    def test_import_and_ingest_preserves_failed_fragment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            repo = root / "repo"
            repo.mkdir()
            init_repo(repo)
            (repo / "m.py").write_text("def f(): return 1\n")
            commit(repo, "base")
            graph = Graph(
                {"a": evidence(nodes=["old.a"]), "b": evidence(nodes=["old.b"])}
            )
            write_json(root / "snapshot.json", graph.to_json())
            trace = {
                "shared": {"frames": [{"file": "/code/m.py", "name": "f", "line": 1}]},
                "profiles": [
                    {
                        "type": "sampled",
                        "unit": "seconds",
                        "startValue": 0,
                        "endValue": 1,
                        "samples": [[0]],
                    }
                ],
            }
            write_json(root / "a.speedscope", trace)
            write_json(
                root / "manifest.json",
                {"a": ["a.speedscope"], "b": ["missing.speedscope"]},
            )
            self.assertEqual(
                main(
                    [
                        "ingest",
                        "--snapshot",
                        str(root / "snapshot.json"),
                        "--manifest",
                        str(root / "manifest.json"),
                        "--repo",
                        str(repo),
                        "--revision",
                        "HEAD",
                        "--remote-root",
                        "/code",
                        "--graphs",
                        str(root / "graphs"),
                        "--out",
                        str(root / "ingest.json"),
                    ]
                ),
                1,
            )
            graph, origin = GraphStore(root / "graphs").load(
                git(repo, "rev-parse", "HEAD").strip()
            )
            self.assertEqual(graph.tests["a"].nodes, {"m.f"})
            self.assertEqual(graph.tests["b"].nodes, {"old.b"})


if __name__ == "__main__":
    unittest.main()
