"""Exercise the complete FastCI CLI against a multi-revision CPU mock CI.

Real: Git branches, source diffs, test assertions, subprocess concurrency, observed
Python call stacks, graph replacement, history, and workload validation.
Mock: GPU devices, historical cost estimates, deterministic trace gaps, and faults.
No private source, GPU, py-spy installation, or network connection is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastci.graph import (  # noqa: E402 - executable from a source checkout
    Graph,
    GraphStore,
)
from fastci.model import TestSpec, read_json, suite_json, write_json  # noqa: E402
from fastci.source import git  # noqa: E402

if __package__:
    from ._fixture_git import commit_fixture, init_fixture
else:
    from _fixture_git import commit_fixture, init_fixture


def require(condition, message):
    if not condition:
        raise RuntimeError("Mock CI invariant failed: " + message)


def catalog(include_new=False):
    definitions = [
        ("smoke_slow", "smoke", 1, "mock_compute", 9),
        ("smoke_fast", "smoke", 1, "mock_compute", 3),
        ("distributed_dp", "data_parallel", 2, "mock_compute", 7),
        ("distributed_tp", "tensor_parallel", 2, "mock_compute", 7),
        ("long_context", "long_sequence", 1, "mock_memory", 8),
        ("checkpoint_roundtrip", "checkpoint", 1, "mock_memory", 6),
        ("shape_guard", "shape_alignment", 1, "mock_compute", 5),
        ("compiled_train", "compile", 1, "mock_compute", 6),
        ("mixed_precision", "precision", 1, "mock_compute", 4),
        ("evaluation", "evaluation", 1, "mock_memory", 4),
        ("loader_cpu", "loader", 0, "cpu", 1),
        ("metrics_cpu", "metrics", 0, "cpu", 1),
    ]
    if include_new:
        definitions.append(("new_cache", "cache", 0, "cpu", 1))
    result = {}
    for name, task, devices, hardware, seconds in definitions:
        metadata = {
            "task_type": task,
            "validation_dimensions": ["parallelism"],
            "unrelated_dimensions": ["layers", "sequence", "logging"],
        }
        if name == "checkpoint_roundtrip":
            metadata["validation_dimensions"].append("layers")
        result[name] = TestSpec(
            name,
            [sys.executable, "mock_worker.py", "--test", name],
            devices,
            hardware,
            seconds,
            metadata=metadata,
            workload={
                "layers": 4,
                "sequence": 1024,
                "logging": True,
                "parallelism": devices,
            }
            if devices
            else {"parallelism": 1},
            bindings={
                "layers": {"kind": "env", "key": "LAYERS"},
                "sequence": {"kind": "env", "key": "SEQUENCE"},
                "logging": {"kind": "env", "key": "LOGGING"},
            },
            env={"MOCK_WORK_SECONDS": ".150" if name == "smoke_slow" else ".025"},
            timeout=20,
        )
    return result


class MockCI:
    def __init__(self, output):
        self.output, self.repo = output, output / "project"
        self.graphs, self.state = output / "graphs", output / "state.json"
        self.traces = output / "traces"
        self.scenarios, self.observations, self.commands = [], [], []
        self.suite = catalog()
        self.checks = {}

    def invoke(self, args, expected=0, run_id="setup"):
        env = {
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "FASTCI_TRACE": "0",
            "MOCK_TRACE_ROOT": str(self.traces),
            "MOCK_RUN_ID": run_id,
        }
        process = subprocess.run(
            [sys.executable, "-m", "fastci", *map(str, args)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.commands.append(
            {
                "argv": ["python", "-m", "fastci", *map(str, args)],
                "exit_code": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
            }
        )
        require(
            process.returncode == expected,
            f"{' '.join(map(str, args[:2]))}: expected {expected}, got {process.returncode}\n{process.stderr}\n{process.stdout}",
        )

    def setup(self):
        require(not self.output.exists(), "choose a fresh output directory")
        self.output.mkdir(parents=True)
        shutil.copytree(
            Path(__file__).with_name("mock_project"),
            self.repo,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        init_fixture(self.repo, branch="main", name="FastCI mock fixture")
        self.commit("Bootstrap synthetic CPU framework")
        write_json(
            self.output / "pools.json",
            {
                "mock_compute": ["mock-c0", "mock-c1"],
                "mock_memory": ["mock-m0", "mock-m1"],
            },
        )
        write_json(
            self.output / "rules.json",
            {
                "reductions": {"layers": 2, "sequence": 16, "logging": False},
                "path_dimensions": {
                    "framework.model.validate_long_sequence": ["sequence"]
                },
            },
        )
        write_json(self.state, {"schema": 1, "guard": {"entries": {}, "interval": 2}})

    def commit(self, message):
        commit_fixture(self.repo, message)

    def branch(self, name, file, before, after):
        git(self.repo, "checkout", "-q", "main")
        git(self.repo, "checkout", "-qb", "codex/mock-" + name)
        path = self.repo / file
        text = path.read_text()
        require(text.count(before) == 1, "unambiguous source edit: " + name)
        path.write_text(text.replace(before, after))
        self.commit("Mock MR: " + name)

    def scenario(
        self,
        name,
        expected_pass=True,
        mode="full",
        baseline=None,
        malformed=None,
        stop=False,
        merge=True,
    ):
        folder = self.output / "runs" / name
        folder.mkdir(parents=True)
        suite_file = folder / "suite.json"
        write_json(suite_file, suite_json(self.suite))
        args = [
            "plan",
            "--repo",
            self.repo,
            "--source",
            "HEAD",
            "--target",
            "main",
            "--suite",
            suite_file,
            "--graphs",
            self.graphs,
            "--state",
            self.state,
            "--rules",
            self.output / "rules.json",
            "--mode",
            mode,
            "--out",
            folder / "plan.json",
        ]
        if baseline is not None:
            write_json(folder / "baseline.json", baseline)
            args.extend(["--baseline", folder / "baseline.json"])
        self.invoke(args)
        planned = read_json(folder / "plan.json")
        before_state = read_json(self.state)
        run_args = [
            "run",
            "--plan",
            folder / "plan.json",
            "--cwd",
            self.repo,
            "--state",
            self.state,
            "--environment",
            "cpu-mock-framework-v1",
            "--run-id",
            name,
            "--pools",
            self.output / "pools.json",
            "--resource-mode",
            "mock",
            "--workers",
            1 if stop else 4,
            "--out",
            folder / "run.json",
        ]
        run_args.extend(["--trace-dir", self.traces / name])
        if stop:
            run_args.append("--stop-on-failure")
        self.invoke(run_args, 0 if expected_pass else 1, name)
        result = read_json(folder / "run.json")
        require(result["all_passed"] is expected_pass, name + ": authoritative result")
        require(
            result["measured_gpu_seconds"] is None
            and result["resource_mode"] == "mock",
            "no GPU measurement claims",
        )
        executed = [o for o in result["outcomes"] if o["attempts"]]
        skipped = [o["test"] for o in result["outcomes"] if not o["attempts"]]
        if skipped:
            recent_before = before_state.get("history", {}).get("recent", {})
            recent_after = read_json(self.state)["history"]["recent"]
            for test in skipped:
                require(
                    recent_before.get(test) == recent_after.get(test),
                    "skipped test must not update history",
                )
        ingest, base = self.ingest_evidence(
            name,
            folder,
            planned["change"]["revision"],
            executed,
            malformed,
            expected_pass,
            merge,
        )
        updated = set(ingest["updated"])
        concurrency = verify_reservations(
            result["outcomes"], read_json(self.output / "pools.json"), self.suite
        )
        outcomes = {o["test"]: o for o in result["outcomes"]}
        self.observations.append(
            {
                "run_id": name,
                "sequence": len(self.observations),
                "files": planned["change"]["files"],
                "tests": [o["test"] for o in executed],
                "outcomes": {o["test"]: o["status"] for o in executed},
                "seconds": {
                    o["test"]: sum(a["seconds"] for a in o["attempts"])
                    for o in executed
                },
                "gpus": {o["test"]: self.suite[o["test"]].gpus for o in executed},
            }
        )
        row = {
            "name": name,
            "catalog": len(self.suite),
            "affected": planned["affected"],
            "order": planned["order"],
            "entities": planned["change"]["entities"],
            "graph_origin": planned["graph_origin"],
            "risk": planned["scores"],
            "groups": planned["selection"]["groups"],
            "fallbacks": planned["selection"]["fallbacks"],
            "outcomes": outcomes,
            "expected_pass": expected_pass,
            "observed_pass": result["all_passed"],
            "skipped": skipped,
            "updated_traces": ingest["updated"],
            "trace_errors": ingest["preserved_on_error"],
            "preserved_tests": sorted(set(base.tests) - updated),
            "max_concurrency": concurrency,
            "wall_seconds": result["wall_seconds"],
            "mock_device_seconds": result["mock_device_seconds"],
            "attempts": sum(len(o["attempts"]) for o in executed),
            "artifact_directory": str(folder.relative_to(self.output)),
        }
        self.scenarios.append(row)
        print(
            f"{name:22} {len(planned['affected']):2} -> {len(planned['order']):2} tests; "
            f"{row['attempts']:2} attempts; {'PASS' if result['all_passed'] else 'EXPECTED FAILURE'}"
        )
        return planned, result, row

    def ingest_evidence(
        self, name, folder, revision, executed, malformed, expected_pass, merge
    ):
        manifest = {}
        for outcome in executed:
            paths = [
                Path(item["path"])
                for attempt in outcome["attempts"]
                for item in attempt["trace_files"]
            ]
            require(
                len(paths) == len(outcome["attempts"]),
                "one observed-call trace per attempt",
            )
            manifest[outcome["test"]] = [str(path) for path in paths]
        if malformed:
            require(malformed in manifest, "fault injection targets an executed test")
            (folder / "malformed.speedscope").write_text('{"broken": true}')
            manifest[malformed] = [str(folder / "malformed.speedscope")]
        write_json(folder / "manifest.json", manifest)
        ingest_args = [
            "ingest",
            "--snapshot",
            folder / "plan.base-graph.json",
            "--manifest",
            folder / "manifest.json",
            "--repo",
            self.repo,
            "--revision",
            "HEAD",
            "--remote-root",
            "/mock/source",
            "--graphs",
            self.graphs,
            "--out",
            folder / "ingest.json",
            "--plan",
            folder / "plan.json",
            "--run",
            folder / "run.json",
        ]
        if expected_pass and merge and name != "bootstrap":
            branch = git(self.repo, "branch", "--show-current").strip()
            git(self.repo, "checkout", "-q", "main")
            git(self.repo, "merge", "--ff-only", branch)
        if expected_pass:
            ingest_args.extend(["--stable", "--stable-ref", "main"])
        self.invoke(ingest_args, 1 if malformed else 0)
        ingest = read_json(folder / "ingest.json")
        base = Graph.from_json(read_json(folder / "plan.base-graph.json"))
        current, _ = GraphStore(self.graphs).load(revision)
        updated = set(ingest["updated"])
        for test, fragment in base.tests.items():
            if test not in updated:
                require(
                    current.tests[test] == fragment,
                    name + ": preserve untouched/failed trace evidence",
                )
        if malformed:
            require(
                malformed in ingest["preserved_on_error"], "malformed trace is reported"
            )
        return ingest, base

    def check(self, name, condition):
        require(condition, name)
        self.checks[name] = True

    def finish(self):
        git(self.repo, "checkout", "-q", "main")
        write_json(self.output / "observations.json", self.observations)
        self.invoke(
            [
                "replay",
                "--input",
                self.output / "observations.json",
                "--warmup",
                1,
                "--resource-mode",
                "mock",
                "--out",
                self.output / "replay.json",
            ]
        )
        # Exercise the public CSV adapter against the same actual observations.
        with (self.output / "observations.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["run_id", "sequence", "test", "status", "files", "seconds"],
            )
            writer.writeheader()
            for record in self.observations:
                for test in record["tests"]:
                    writer.writerow(
                        {
                            "run_id": record["run_id"],
                            "sequence": record["sequence"],
                            "test": test,
                            "status": record["outcomes"][test],
                            "files": json.dumps(record["files"]),
                            "seconds": record["seconds"][test],
                        }
                    )
        write_json(self.output / "suite.json", suite_json(self.suite))
        self.invoke(
            [
                "prepare-history",
                "--csv",
                self.output / "observations.csv",
                "--suite",
                self.output / "suite.json",
                "--out",
                self.output / "csv-history.json",
            ]
        )
        self.invoke(
            [
                "replay",
                "--input",
                self.output / "csv-history.json",
                "--warmup",
                1,
                "--resource-mode",
                "mock",
                "--out",
                self.output / "csv-replay.json",
            ]
        )
        self.check(
            "csv_and_json_replays_agree",
            read_json(self.output / "replay.json")
            == read_json(self.output / "csv-replay.json"),
        )
        self.check(
            "parallel_execution_observed",
            max(r["max_concurrency"] for r in self.scenarios) > 1,
        )
        summary = {
            "kind": "end_to_end_cpu_mock",
            "resource_mode": "mock",
            "scenario_count": len(self.scenarios),
            "catalog_size": len(self.suite),
            "checks": self.checks,
            "scenarios": self.scenarios,
            "test_executions": sum(
                len(r["outcomes"]) - len(r["skipped"]) for r in self.scenarios
            ),
            "subprocess_attempts": sum(r["attempts"] for r in self.scenarios),
            "measured_gpu_seconds": None,
            "limits": [
                "No GPU or real cluster is exercised.",
                "Call instrumentation substitutes for statistical sampling.",
                "Missing frames, malformed traces, and regressions are deliberate fixture faults.",
                "This workflow does not measure paper performance or prove no missed regressions.",
            ],
        }
        write_json(self.output / "summary.json", summary)
        write_json(self.output / "commands.json", self.commands)
        lines = [
            "# Complete CPU mock run",
            "",
            "All device reservations are mock slots. GPU performance is not measured.",
            "",
            "| Scenario | Affected | Retained | Attempts | Outcome | Graph |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
        for row in self.scenarios:
            lines.append(
                f"| {row['name']} | {len(row['affected'])} | {len(row['order'])} | {row['attempts']} | "
                f"{'pass' if row['observed_pass'] else 'expected failure'} | {row['graph_origin']} |"
            )
        lines.extend(
            ["", "## Verified invariants", ""] + [f"- {name}" for name in self.checks]
        )
        (self.output / "report.md").write_text("\n".join(lines) + "\n")
        print(
            f"Verified {len(self.checks)} invariants; report: {self.output / 'report.md'}"
        )
        return summary


def verify_reservations(outcomes, pools, suite):
    """Check actual dispatch intervals, not expected/fabricated completion times."""
    events = []
    for outcome in outcomes:
        if not outcome["attempts"]:
            continue
        name = outcome["test"]
        devices = outcome["devices"]
        require(len(devices) == suite[name].gpus, "requested device count")
        require(
            set(devices).issubset(pools.get(suite[name].hardware, [])),
            "hardware pool isolation",
        )
        start, end = outcome["started_seconds"], outcome["finished_seconds"]
        require(start is not None and end >= start, "dispatch timestamps")
        events.extend([(start, 1, name, devices), (end, -1, name, devices)])
    busy, active, maximum = set(), 0, 0
    for _, change, name, devices in sorted(events):
        if change > 0:
            require(
                not busy.intersection(devices),
                "no concurrent double allocation: " + name,
            )
            busy.update(devices)
        else:
            require(set(devices).issubset(busy), "release only reserved devices")
            busy.difference_update(devices)
        active += change
        maximum = max(maximum, active)
    require(not busy and active == 0, "all reservations released")
    return maximum


def run_mock(output):
    ci = MockCI(output.resolve())
    ci.setup()
    ci.scenario("bootstrap", mode="baseline")
    graph, _ = GraphStore(ci.graphs).load(git(ci.repo, "rev-parse", "main").strip())
    ci.check(
        "bootstrap_traces_are_observed_calls",
        bool(graph.covering("framework.model.Encoder.forward")),
    )
    ci.check(
        "deliberate_sampling_gaps",
        not graph.covering("framework.model.Encoder.validate_config")
        and not graph.covering("framework.ops.unobserved_activation"),
    )

    ci.branch(
        "failed_tp",
        "framework/distributed.py",
        "return sum(values) + 0",
        "return sum(values) + 1",
    )
    ci.scenario("failed_tp", expected_pass=False)
    ci.branch(
        "model_a", "framework/model.py", "return result\n", "return list(result)\n"
    )
    p, r, _ = ci.scenario("model_a")
    outcomes = {o["test"]: o for o in r["outcomes"]}
    ci.check("risk_learns_from_actual_prior_failure", p["order"][0] == "distributed_tp")
    ci.check(
        "lowest_cost_equivalent_test_retained",
        "smoke_fast" in p["order"]
        and "smoke_slow" in p["affected"]
        and "smoke_slow" not in p["order"],
    )
    ci.check(
        "distinct_validation_scopes_retained",
        {"distributed_dp", "distributed_tp", "mixed_precision"}.issubset(p["order"]),
    )
    ci.check(
        "runtime_path_protects_long_sequence",
        "sequence" not in p["proposals"]["long_context"]["overrides"],
    )
    ci.check(
        "initial_unsafe_reduction_revoked",
        outcomes["shape_guard"]["comparison_agreed"] is False
        and outcomes["shape_guard"]["status"] == "pass",
    )

    # Deliberately remove only the exact-version materialization; stable remains.
    GraphStore(ci.graphs).path(git(ci.repo, "rev-parse", "main").strip()).unlink()
    ci.branch(
        "model_b", "framework/model.py", "return list(result)", "return result[:]"
    )
    p, r, _ = ci.scenario("model_b")
    outcomes = {o["test"]: o for o in r["outcomes"]}
    ci.check("stable_graph_fallback", p["graph_origin"] == "stable")
    ci.check(
        "accepted_workload_reused",
        outcomes["compiled_train"]["workload_action"] == "reduced",
    )
    ci.check(
        "revoked_workload_restored",
        outcomes["shape_guard"]["workload_action"] == "original",
    )

    ci.branch(
        "compile_regression",
        "framework/compiler.py",
        "large_model_bias = 0",
        "large_model_bias = 1",
    )
    _, r, _ = ci.scenario("compile_regression", expected_pass=False)
    compiled = next(o for o in r["outcomes"] if o["test"] == "compiled_train")
    ci.check(
        "periodic_pair_catches_large_only_regression",
        compiled["workload_action"] == "paired"
        and [a["status"] for a in compiled["attempts"]] == ["fail", "pass"]
        and compiled["status"] == "fail",
    )
    ci.branch(
        "compile_repair",
        "framework/compiler.py",
        "return value + (large_model_bias if layers >= 4 else 0)",
        "return (large_model_bias if layers >= 4 else 0) + value",
    )
    _, r, _ = ci.scenario("compile_repair")
    ci.check(
        "periodic_revocation_persists",
        r["outcomes"][0]["workload_action"] == "original",
    )

    ci.branch(
        "class_expansion", "framework/model.py", "assert width > 0", "assert width >= 1"
    )
    p, _, _ = ci.scenario("class_expansion")
    ci.check(
        "within_class_missing_evidence_expansion",
        p["change"]["entities"] == ["framework.model.Encoder.validate_config"]
        and "smoke_slow" in p["order"]
        and "smoke_fast" in p["order"]
        and not p["selection"]["fallbacks"],
    )
    ci.branch(
        "caller_expansion",
        "framework/ops.py",
        "return max(0, value)",
        "return max(value, 0)",
    )
    p, _, _ = ci.scenario("caller_expansion")
    ci.check(
        "static_caller_missing_evidence_expansion",
        p["change"]["entities"] == ["framework.ops.unobserved_activation"]
        and bool(p["order"])
        and not p["selection"]["fallbacks"],
    )

    ci.suite = catalog(include_new=True)
    ci.branch(
        "config_and_new_test",
        "ci-settings.json",
        '"retry_limit": 1',
        '"retry_limit": 2',
    )
    p, _, row = ci.scenario(
        "config_and_new_test",
        baseline=["smoke_fast", "checkpoint_roundtrip", "loader_cpu"],
        malformed="checkpoint_roundtrip",
    )
    ci.check(
        "config_routes_to_existing_rules",
        "non_python:existing_ci_rules" in p["selection"]["fallbacks"],
    )
    ci.check(
        "untraced_new_test_retained",
        "untraced_tests:retain" in p["selection"]["fallbacks"]
        and "new_cache" in p["order"],
    )
    ci.check(
        "malformed_trace_preserves_previous_evidence",
        "checkpoint_roundtrip" in row["trace_errors"]
        and "checkpoint_roundtrip" in row["preserved_tests"],
    )

    ci.branch(
        "unknown_entity", "framework/ops.py", "return value\n", "return value + 0\n"
    )
    p, _, _ = ci.scenario("unknown_entity")
    ci.check(
        "no_evidence_falls_back_to_full_baseline",
        len(p["order"]) == len(ci.suite)
        and any(
            reason.endswith(":no_expansion_evidence")
            for reason in p["selection"]["fallbacks"]
        ),
    )
    ci.branch(
        "rollback_failfast",
        "framework/distributed.py",
        "return sum(values) + 0",
        "return sum(values) + 1",
    )
    p, r, row = ci.scenario(
        "rollback_failfast",
        expected_pass=False,
        mode="baseline",
        baseline=["distributed_tp", "smoke_fast", "loader_cpu"],
        stop=True,
    )
    ci.check(
        "baseline_rollback_disables_optimizations",
        p["order"] == ["distributed_tp", "smoke_fast", "loader_cpu"]
        and all(not value["overrides"] for value in p["proposals"].values()),
    )
    ci.check(
        "failfast_skips_pending_without_observing_passes",
        row["skipped"] == ["smoke_fast", "loader_cpu"],
    )
    return ci.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run_mock(parser.parse_args().out)
