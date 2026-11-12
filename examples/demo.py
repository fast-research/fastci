"""Run a complete CPU-only fixture, including deliberately unsafe reduction rollback.

All sampled stacks, durations, historical failures, and validation labels here
are synthetic fixtures. Test subprocesses are actually executed. Nothing in this
demo measures LLM training speedups or validates GPU workloads.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastci.cli import main  # noqa: E402 - executable from a source checkout
from fastci.graph import Graph, GraphStore  # noqa: E402
from fastci.model import TestSpec, read_json, suite_json, write_json  # noqa: E402
from fastci.risk import RiskHistory  # noqa: E402
from fastci.source import SourceIndex, git  # noqa: E402
from fastci.traces import parse_speedscope  # noqa: E402

if __package__:
    from ._fixture_git import commit_fixture, init_fixture
else:
    from _fixture_git import commit_fixture, init_fixture


def call(*args: str) -> None:
    if main(list(args)) != 0:
        raise RuntimeError(f"Command failed: {args[0]}")


def fixture_suite():
    # Equal-cost CPU representatives keep suite order.
    names = ["smoke_fast", "smoke_slow", "long_context", "guarded"]
    suite = {}
    for name in names:
        kind = "smoke" if name.startswith("smoke") else name
        suite[name] = TestSpec(
            name,
            [sys.executable, "smoke.py"],
            seconds=20 if name == "smoke_slow" else 10,
            metadata={
                "task_type": kind,
                "unrelated_dimensions": ["layers", "sequence", "logging"],
            },
            env={"KIND": kind},
            workload={"layers": 12, "sequence": 8192, "logging": True},
            bindings={
                "layers": {"kind": "env", "key": "LAYERS"},
                "sequence": {"kind": "env", "key": "SEQUENCE"},
                "logging": {"kind": "env", "key": "LOGGING"},
            },
        )
    return suite


def fixture_evidence(destination, repo, model, suite):
    frames = [
        {"file": "/fixture/smoke.py", "name": "run", "line": 7},
        {"file": "/fixture/model.py", "name": "forward", "line": 2},
        {"file": "/fixture/model.py", "name": "validate_long_sequence", "line": 5},
    ]
    trace_manifest, fragments = {}, {}
    index = SourceIndex(
        {"model.py": model, "smoke.py": (repo / "smoke.py").read_text()}
    )
    for name in suite:
        data = {
            "shared": {"frames": frames},
            "profiles": [
                {
                    "type": "sampled",
                    "unit": "seconds",
                    "startValue": 0,
                    "endValue": 10,
                    "samples": [[0, 1]] + ([[0, 2]] if name == "long_context" else []),
                }
            ],
        }
        path = destination / f"{name}.speedscope"
        write_json(path, data)
        trace_manifest[name] = [path.name]
        fragments[name] = parse_speedscope([path], "/fixture", index)
    return trace_manifest, fragments


def demo(destination: Path) -> dict:
    if destination.exists():
        raise ValueError(
            "Use a new destination directory to preserve prior demo evidence"
        )
    destination.mkdir(parents=True)
    repo = destination / "repo"
    repo.mkdir()
    init_fixture(repo)
    model = "def forward(width):\n    return width * 2\n\ndef validate_long_sequence(sequence):\n    assert sequence >= 8192\n"
    (repo / "model.py").write_text(model)
    (repo / "smoke.py").write_text("""import os
from model import forward, validate_long_sequence

def run():
    layers = int(os.environ.get("LAYERS", "12"))
    sequence = int(os.environ.get("SEQUENCE", "8192"))
    assert forward(4) == 12
    if os.environ.get("KIND") == "long_context":
        validate_long_sequence(sequence)
    if os.environ.get("KIND") == "guarded":
        assert layers == 12, "original model shape is part of the actual test"

if __name__ == "__main__":
    run()
""")
    commit_fixture(repo, "fixture base")
    base = git(repo, "rev-parse", "HEAD").strip()
    suite = fixture_suite()
    write_json(destination / "suite.json", suite_json(suite))
    rules = read_json(Path(__file__).with_name("rules.json"))
    write_json(destination / "rules.json", rules)
    trace_manifest, fragments = fixture_evidence(destination, repo, model, suite)
    GraphStore(destination / "graphs").publish(Graph(fragments), base, stable=True)
    write_json(destination / "manifest.json", trace_manifest)
    history = RiskHistory(recent={"long_context": [1]})
    write_json(destination / "state.json", {"schema": 1, "history": asdict(history)})
    (repo / "model.py").write_text(model.replace("width * 2", "width * 3"))
    commit_fixture(repo, "fixture MR changes forward")
    call(
        "plan",
        "--repo",
        str(repo),
        "--target",
        base,
        "--suite",
        str(destination / "suite.json"),
        "--graphs",
        str(destination / "graphs"),
        "--state",
        str(destination / "state.json"),
        "--rules",
        str(destination / "rules.json"),
        "--out",
        str(destination / "plan.json"),
    )
    call(
        "run",
        "--plan",
        str(destination / "plan.json"),
        "--cwd",
        str(repo),
        "--state",
        str(destination / "state.json"),
        "--environment",
        "cpu-fixture-v1",
        "--run-id",
        "fixture-mr-1",
        "--out",
        str(destination / "run.json"),
    )
    planned = read_json(destination / "plan.json")
    # Post-run trace manifest contains executed tests only; unexecuted fragments
    # must survive from the frozen base snapshot.
    write_json(
        destination / "manifest.json", {n: trace_manifest[n] for n in planned["order"]}
    )
    call(
        "ingest",
        "--snapshot",
        str(destination / "plan.base-graph.json"),
        "--manifest",
        str(destination / "manifest.json"),
        "--repo",
        str(repo),
        "--revision",
        "HEAD",
        "--remote-root",
        "/fixture",
        "--graphs",
        str(destination / "graphs"),
        "--out",
        str(destination / "ingest.json"),
    )
    planned, result = (
        read_json(destination / "plan.json"),
        read_json(destination / "run.json"),
    )
    guarded = next(o for o in result["outcomes"] if o["test"] == "guarded")
    assert planned["order"][0] == "long_context"
    assert "smoke_slow" not in planned["order"]
    assert "sequence" not in planned["proposals"]["long_context"]["overrides"]
    assert guarded["comparison_agreed"] is False and guarded["status"] == "pass"
    summary = {
        "kind": "CPU fixture; synthetic trace/history; actual subprocess execution",
        "affected": planned["affected"],
        "order": planned["order"],
        "all_passed_under_authoritative_workload": result["all_passed"],
        "unsafe_reduction_revoked": True,
        "original_and_reduced_attempts": sum(
            len(o["attempts"]) for o in result["outcomes"]
        ),
        "graph_update": read_json(destination / "ingest.json"),
        "measured_gpu_seconds": result["measured_gpu_seconds"],
    }
    write_json(destination / "summary.json", summary)
    print(f"Demo verified; inspect {destination / 'summary.json'}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    demo(args.out.resolve())
