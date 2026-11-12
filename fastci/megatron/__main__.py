from __future__ import annotations

import argparse
import subprocess
from dataclasses import asdict
from pathlib import Path

from ..model import read_json, write_json
from ..risk import RiskHistory
from .catalog import catalog, config
from .common import MODES
from .evidence import evaluate, ingest, observe
from .planner import plan
from .runner import probe, run


def parser():
    parser = argparse.ArgumentParser(
        description="Native Megatron-LM functional CI with FastCI"
    )
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("check", "catalog", "plan", "run", "ingest", "evaluate", "observe"):
        command = commands.add_parser(action)
        command.add_argument("--out", required=True, type=Path)
        if action in {"catalog", "plan", "run"}:
            command.add_argument("--repo", required=True, type=Path)
        if action in {"check", "catalog", "plan"}:
            command.add_argument("--config", required=True, type=Path)
        if action in {"catalog", "plan"}:
            command.add_argument("--source", default="HEAD")
        if action == "plan":
            command.add_argument("--target", required=True)
            command.add_argument("--graph", type=Path)
            command.add_argument("--mode", choices=MODES, default="shadow")
        if action in {"plan", "observe"}:
            command.add_argument("--history", type=Path)
        if action == "run":
            command.add_argument("--plan", required=True, type=Path)
            command.add_argument("--dry-run", action="store_true")
        if action in {"ingest", "evaluate", "observe"}:
            command.add_argument("--run", required=True, type=Path)
    return parser


def _history(path):
    return RiskHistory(**read_json(path)) if path else RiskHistory()


def _plan(args):
    cat = catalog(args.repo, config(args.config), args.source)
    history = _history(args.history)
    try:
        evidence = read_json(args.graph) if args.graph else None
    except (OSError, ValueError) as error:
        print(f"Graph unavailable: {error}")
        evidence = None
    return plan(args.repo, cat, args.target, evidence, args.mode, history)


def _artifact(args):
    if args.action == "plan":
        return _plan(args)
    if args.action == "check":
        return probe(config(args.config))
    if args.action == "catalog":
        return catalog(args.repo, config(args.config), args.source)
    if args.action == "ingest":
        return ingest(args.run)
    result = read_json(args.run)
    if args.action == "observe":
        return asdict(observe(result, _history(args.history)))
    return evaluate(result)


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    try:
        if args.action == "run":
            result = run(args.repo, read_json(args.plan), args.out, args.dry_run)
            print(args.out / "run.json")
            return 0 if args.dry_run or result["all_passed"] else 1
        write_json(args.out, _artifact(args))
        print(args.out)
        return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        root.exit(1, f"fastci-megatron: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
