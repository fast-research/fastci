from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from .engine import plan
from .graph import Graph, GraphStore
from .model import digest, load_suite, read_json, suite_from_json, write_json
from .policy import BASELINE_MODES, BYPASS_MODES, MODES, configured_mode
from .provenance import verify_checkout
from .repository import changed_paths, merge_base, resolve_commit
from .risk import RiskHistory
from .source import Change, analyze
from .workload import Proposal, WorkloadGuard


def load_state(path: str | Path) -> tuple[RiskHistory, WorkloadGuard]:
    data = read_json(path) if Path(path).exists() else {}
    return RiskHistory(**data.get("history", {})), WorkloadGuard(
        **data.get("guard", {})
    )


def save_state(path: str | Path, history: RiskHistory, guard: WorkloadGuard) -> None:
    write_json(path, {"schema": 1, "history": asdict(history), "guard": asdict(guard)})


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Paper-aligned FastCI reference implementation"
    )
    commands = root.add_subparsers(dest="action", required=True)
    p = commands.add_parser(
        "plan", help="Analyze immutable Git versions and save a reviewable plan"
    )
    p.add_argument("--repo", required=True)
    p.add_argument("--source", default="HEAD")
    p.add_argument("--target", required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--graphs", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--rules")
    p.add_argument(
        "--baseline", help="JSON list from existing CI rules; defaults to whole suite"
    )
    p.add_argument("--mode", choices=MODES, default="full")
    p.add_argument(
        "--config",
        help="JSON configuration; its mode field controls admission and rollback",
    )
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--out", required=True)
    p = commands.add_parser(
        "run", help="Execute a plan against the exact clean source revision"
    )
    p.add_argument("--plan", required=True)
    p.add_argument(
        "--config",
        help="Re-read configuration before execution; defaults to the plan's config",
    )
    p.add_argument("--cwd", required=True)
    p.add_argument("--state", required=True)
    p.add_argument(
        "--environment",
        required=True,
        help="Stable environment/image identity for paired validation",
    )
    p.add_argument("--run-id", required=True)
    p.add_argument(
        "--pools", help="JSON hardware label -> list of local CUDA device IDs"
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--resource-mode",
        choices=["local", "mock"],
        default="local",
        help="Label CPU simulations of device pools explicitly; mock mode reports no measured GPU work",
    )
    p.add_argument("--stop-on-failure", action="store_true")
    p.add_argument(
        "--trace-dir", help="Fresh per-run trace directory; defaults beside output logs"
    )
    p.add_argument("--trace-rate", type=int, default=50)
    p.add_argument("--trace-subprocesses", action="store_true")
    p.add_argument(
        "--no-trace", action="store_true", help="Disable automatic py-spy injection"
    )
    p.add_argument("--out", required=True)
    p = commands.add_parser(
        "import-graph", help="Convert legacy graph JSON into a versioned store"
    )
    p.add_argument("--input", required=True)
    p.add_argument("--graphs", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--stable", action="store_true")
    p = commands.add_parser(
        "ingest", help="Replace successful trace fragments in a frozen base snapshot"
    )
    p.add_argument("--config", help="disabled mode bypasses graph updates")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--plan", help="Plan used to select this run; required with --run")
    p.add_argument(
        "--run", help="Verified execution output binding trace hashes to a revision"
    )
    p.add_argument(
        "--manifest",
        required=True,
        help="JSON mapping test names to Speedscope file lists",
    )
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--remote-root", required=True)
    p.add_argument("--graphs", required=True)
    p.add_argument(
        "--stable", action="store_true", help="Publish stable graph after merge"
    )
    p.add_argument(
        "--stable-ref", help="Branch/ref that already contains the tested revision"
    )
    p.add_argument("--out", required=True)
    p = commands.add_parser(
        "replay", help="Chronological risk-order replay; no GPU submission"
    )
    p.add_argument("--input", required=True)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--resource-mode", choices=["local", "mock"], default="local")
    p.add_argument("--out", required=True)
    p = commands.add_parser(
        "prepare-history",
        help="Convert portable CI result CSV into chronological replay records",
    )
    p.add_argument("--csv", required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--out", required=True)
    p = commands.add_parser(
        "sample", help="Attach py-spy to an existing rank-0 Python PID"
    )
    p.add_argument("--config")
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--rate", type=int, default=50)
    p.add_argument("--subprocesses", action="store_true")
    p.add_argument("--out", required=True)
    return root


def _plan(args: argparse.Namespace) -> int:
    mode = configured_mode(args.config, args.mode)
    bypassed = mode in BYPASS_MODES
    if mode in BASELINE_MODES:
        revision = resolve_commit(args.repo, args.source)
        base = merge_base(args.repo, revision, args.target)
        files = changed_paths(args.repo, base, revision, detect_renames=True)
        change, indexes = Change(base, revision, files, []), ()
    else:
        change, old, new = analyze(args.repo, args.source, args.target)
        indexes = (old, new)
    store = GraphStore(args.graphs)
    graph, origin = (Graph(), mode) if bypassed else store.load(change.merge_base)
    history = RiskHistory() if bypassed else load_state(args.state)[0]
    result = plan(
        change,
        graph,
        load_suite(args.suite),
        indexes,
        history,
        read_json(args.rules) if args.rules else {},
        read_json(args.baseline) if args.baseline else None,
        mode,
        args.depth,
    )
    result["config"] = str(Path(args.config).resolve()) if args.config else None
    result["graph_origin"] = origin
    result["graph_load_errors"] = store.load_errors
    # Keep immutable run snapshot even if stable graph advances before ingest.
    snapshot = Path(args.out).resolve().with_suffix(".base-graph.json")
    frozen = graph.to_json()
    write_json(snapshot, frozen)
    result["base_snapshot"] = {"path": str(snapshot), "digest": digest(frozen)}
    result["history_digest"] = digest(asdict(history))
    write_json(args.out, result)
    print(
        f"Plan: {len(result['affected'])} affected -> {len(result['order'])} retained; graph={origin}"
    )
    return 0


def _run(args: argparse.Namespace) -> int:
    from .runner import report, run_tests

    data = read_json(args.plan)
    config_path = args.config or data.get("config")
    latest_mode = configured_mode(config_path, data["mode"])
    effective_mode = "disabled" if latest_mode == "disabled" else data["mode"]
    if latest_mode not in (data["mode"], "disabled", "drain"):
        raise ValueError("Configuration changed; generate a fresh plan")
    bypassed = effective_mode in BYPASS_MODES
    history, guard = (
        (RiskHistory(), WorkloadGuard()) if bypassed else load_state(args.state)
    )
    if args.run_id in history.seen:
        raise ValueError("Run ID already observed")
    if not bypassed and digest(asdict(history)) != data["history_digest"]:
        raise ValueError("Risk history changed after planning; generate a fresh plan")
    actual = data["change"]["revision"]
    verify_checkout(args.cwd, actual)
    if (
        not bypassed
        and digest(read_json(data["base_snapshot"]["path"]))
        != data["base_snapshot"]["digest"]
    ):
        raise ValueError("The plan's frozen base snapshot has changed")
    suite = suite_from_json(data["suite"])
    proposals = {name: Proposal(**row) for name, row in data["proposals"].items()}
    order = data["baseline"] if bypassed else data["order"]
    if bypassed:
        proposals = {name: Proposal() for name in order}
    outcomes = run_tests(
        order,
        suite,
        proposals,
        guard,
        args.environment,
        actual,
        args.cwd,
        read_json(args.pools) if args.pools else {},
        Path(args.out).resolve().with_suffix(".logs"),
        args.workers,
        args.stop_on_failure,
        trace_dir=args.trace_dir or Path(args.out).resolve().with_suffix(".traces"),
        trace_enabled=not bypassed
        and not args.no_trace
        and args.resource_mode != "mock",
        trace_rate=args.trace_rate,
        trace_subprocesses=args.trace_subprocesses,
        verify_source=lambda: verify_checkout(args.cwd, actual),
        trace_config=config_path,
    )
    verify_checkout(args.cwd, actual)
    result = report(outcomes, args.resource_mode)
    result.update(
        run_id=args.run_id,
        revision=actual,
        environment=args.environment,
        source_verified=True,
        effective_mode=effective_mode,
        plan_digest=digest(data),
        kind="mock_execution" if args.resource_mode == "mock" else "local_execution",
    )
    # Persist execution evidence before advancing single-writer state.
    write_json(args.out, result)
    write_json(
        Path(args.out).with_suffix(".traces.json"),
        {
            outcome.test: [
                item["path"]
                for attempt in outcome.attempts
                for item in attempt["trace_files"]
            ]
            for outcome in outcomes
            if any(attempt["trace_files"] for attempt in outcome.attempts)
        },
    )
    if not bypassed:
        history.observe(
            args.run_id, data["change"]["files"], {o.test: o.status for o in outcomes}
        )
        save_state(args.state, history, guard)
    metric = (
        "mock_device_seconds"
        if args.resource_mode == "mock"
        else "measured_gpu_seconds"
    )
    print(
        f"Run: {len(outcomes)} tests, passed={result['all_passed']}; {metric}={result[metric]:.3f}"
    )
    return 0 if result["all_passed"] else 1


def _import_graph(args: argparse.Namespace) -> int:
    graph = Graph.from_json(read_json(args.input))
    GraphStore(args.graphs).publish(graph, args.revision, args.stable)
    print(f"Imported graph: {len(graph.tests)} tests")
    return 0


def _ingest(args: argparse.Namespace) -> int:
    from .ingestion import ingest

    plan_data = read_json(args.plan) if args.plan and not args.config else None
    config_path = args.config or (
        plan_data.get("config") if plan_data is not None else None
    )
    if configured_mode(config_path) == "disabled":
        write_json(
            args.out, {"disabled": True, "updated": [], "preserved_on_error": {}}
        )
        return 0
    if args.plan and plan_data is None:
        plan_data = read_json(args.plan)
    result = ingest(
        repo=args.repo,
        ref=args.revision,
        remote_root=args.remote_root,
        snapshot=read_json(args.snapshot),
        manifest_path=Path(args.manifest).resolve(),
        store=GraphStore(args.graphs),
        plan=plan_data,
        run=read_json(args.run) if args.run else None,
        stable=args.stable,
        stable_ref=args.stable_ref,
    )
    write_json(args.out, result)
    updated, errors = result["updated"], result["preserved_on_error"]
    print(f"Ingested {len(updated)} tests; preserved {len(errors)} failed fragments")
    return 0 if not errors else 1


def _replay(args: argparse.Namespace) -> int:
    from .replay import scheduling_replay

    write_json(
        args.out,
        scheduling_replay(read_json(args.input), args.warmup, args.resource_mode),
    )
    return 0


def _prepare_history(args: argparse.Namespace) -> int:
    from .history_csv import prepare_history

    records, diagnostics = prepare_history(args.csv, load_suite(args.suite))
    write_json(args.out, records)
    write_json(Path(args.out).with_suffix(".diagnostics.json"), diagnostics)
    print(
        f"Prepared {len(records)} runs; {diagnostics['duplicate_test_observations']} duplicate observations omitted"
    )
    return 0


def _sample(args: argparse.Namespace) -> int:
    if configured_mode(args.config) == "disabled":
        print("FastCI sampling disabled by configuration")
        return 0
    from .sampler import sampler_command

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    return subprocess.call(
        sampler_command(args.pid, args.out, args.rate, args.subprocesses)
    )


def dispatch(args: argparse.Namespace) -> int:
    handlers = {
        "plan": _plan,
        "run": _run,
        "import-graph": _import_graph,
        "ingest": _ingest,
        "replay": _replay,
        "prepare-history": _prepare_history,
        "sample": _sample,
    }
    return handlers[args.action](args)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return dispatch(args)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"FastCI: {exc}", file=sys.stderr)
        return 2
