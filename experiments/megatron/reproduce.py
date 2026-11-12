#!/usr/bin/env python3
"""Recalculate Megatron's fixed-snapshot selection table on CPU."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from subprocess import CalledProcessError

if __package__:
    from . import replay_report
else:
    import replay_report

DATA = Path(__file__).resolve().parent / "data"


def read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def benchmark_selector(repo, data):
    from fastci.megatron.replay_selection import BenchmarkSelector
    from fastci.model import digest
    from fastci.repository import changed_paths, sources_at, tracked_paths

    frozen = read_gzip(DATA / "call-graph.json.gz")
    if frozen["reference"] != data["reference"]:
        raise ValueError("Graph and line oracle reference revisions differ")
    if frozen["job_registry"].keys() != data["unsampled_gpu_seconds"].keys():
        raise ValueError("Graph and line oracle test catalogs differ")
    reference_paths = tracked_paths(repo, frozen["reference"])
    selector = BenchmarkSelector(frozen, reference_paths)

    @lru_cache(maxsize=2)
    def sources(sha):
        return sources_at(repo, sha)

    def select(commit):
        sha, parent = commit["commit"], commit["parent"]
        files = changed_paths(repo, parent, sha)
        if digest(files) != commit["changed_files_digest"]:
            raise ValueError(
                f"Changed paths differ from the archived first-parent diff on {sha}"
            )
        after, before = sources(sha), sources(parent)
        decisions, diagnostics = selector.select(files, before, after)
        if decisions.keys() != replay_report.FASTCI_STRATEGIES.keys():
            raise ValueError(f"Unexpected FastCI variants on {sha}")
        for method, selected in decisions.items():
            expected = set(commit["selections"][method])
            if set(selected) != expected:
                raise ValueError(
                    f"Selection mismatch on {sha} for {method}: "
                    f"extra={sorted(set(selected) - expected)}, missing={sorted(expected - set(selected))}"
                )
        return decisions, diagnostics

    return select


# Each worker retains two adjacent source snapshots and one frozen graph.
_worker_selector = None


def initialize_selector(repo, data):
    global _worker_selector
    _worker_selector = benchmark_selector(repo, data)


def select_commit(commit):
    return _worker_selector(commit)


def _report_progress(decisions, count):
    for i, decision in enumerate(decisions, 1):
        if i % 10 == 0:
            print(f"Verified FastCI selections for {i}/{count} commits", flush=True)
        yield decision


def verify_selections(repo, data, commits, jobs):
    if jobs == 1 or len(commits) == 1:
        yield from _report_progress(
            map(benchmark_selector(repo, data), commits), len(commits)
        )
        return
    inputs = {key: data[key] for key in ("reference", "unsampled_gpu_seconds")}
    with ProcessPoolExecutor(
        max_workers=jobs, initializer=initialize_selector, initargs=(repo, inputs)
    ) as pool:
        decisions = pool.map(select_commit, commits, chunksize=4)
        yield from _report_progress(decisions, len(commits))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--repo",
        type=Path,
        help="recompute and verify the four FastCI selection variants using upstream Git",
    )
    parser.add_argument(
        "--commit", help="limit selection verification to this commit or unique prefix"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(os.cpu_count() or 1, 4),
        help="CPU processes for recomputing selections",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.commit and not args.repo:
        parser.error("--commit requires --repo")
    data = read_gzip(DATA / "mainline.json.gz")
    commits = data["commits"]
    costs = data["unsampled_gpu_seconds"]
    if (
        data["schema"] != 2
        or len(commits) != 100
        or len(costs) != 47
        or any(
            row["parent"] != previous["commit"]
            for row, previous in zip(commits, commits[1:])
        )
    ):
        raise ValueError("Unexpected Megatron benchmark size or schema")
    recomputed = None
    if args.repo:
        selected = (
            [row for row in commits if row["commit"].startswith(args.commit)]
            if args.commit
            else commits
        )
        if args.commit and len(selected) != 1:
            parser.error("--commit must identify exactly one archived commit")
        try:
            recomputed = replay_report.build_report(
                data, selected, verify_selections(args.repo, data, selected, args.jobs)
            )
        except (OSError, ValueError, CalledProcessError) as error:
            parser.exit(1, f"FastCI selection replay failed: {error}\n")

    report = (
        recomputed
        if recomputed is not None and not args.commit
        else replay_report.build_report(data, commits)
    )
    paper = replay_report.format_table(report)
    if paper != (DATA / "expected-table.tsv").read_text(encoding="utf-8"):
        raise ValueError("Recomputed table differs from the archived benchmark")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "paper-table.tsv").write_text(paper, encoding="utf-8")
    scores = {
        **report,
        "results": [
            {
                key: row[key]
                for key in (
                    "commit",
                    "parent",
                    "cohort",
                    "upstream_pr",
                    "affected",
                    "scores",
                )
            }
            for row in report["results"]
        ],
    }
    (args.out / "results.json").write_text(
        json.dumps(scores, indent=2) + "\n", encoding="utf-8"
    )
    print(paper, end="")
    if recomputed is not None:
        output = args.out / "recomputed-selector.json"
        output.write_text(json.dumps(recomputed, indent=2) + "\n", encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
