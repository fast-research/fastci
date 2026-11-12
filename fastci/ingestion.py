"""Build and publish graph snapshots from sampled execution evidence."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .graph import Graph, GraphStore
from .model import read_json
from .provenance import execution_seconds, validate_ingestion
from .repository import resolve_commit, sources_at, tree_id
from .source import SourceIndex
from .traces import parse_speedscope


def _integrated_revision(repo, revision, run, stable_ref):
    if run is None or not stable_ref:
        raise ValueError("Stable publication requires --plan, --run, and --stable-ref")
    if not run["all_passed"]:
        raise ValueError("A failed CI run cannot become the stable graph")
    integrated = resolve_commit(repo, stable_ref)
    if tree_id(repo, revision) != tree_id(repo, integrated):
        raise ValueError(
            "Integrated source tree differs from the tested tree; test the integration revision first"
        )
    return integrated


def ingest(
    *,
    repo,
    ref,
    remote_root,
    snapshot,
    manifest_path: Path,
    store: GraphStore,
    plan=None,
    run=None,
    stable=False,
    stable_ref=None,
) -> dict:
    if (plan is None) != (run is None):
        raise ValueError("Provide both --plan and --run for verified ingestion")
    revision = resolve_commit(repo, ref)
    graph = Graph.from_json(snapshot)
    manifest = read_json(manifest_path)
    if plan is not None:
        accepted, errors = validate_ingestion(
            plan, run, snapshot, revision, manifest, manifest_path.parent
        )
        outcomes = {row["test"]: row for row in run["outcomes"]}
    else:
        accepted = {
            test: [manifest_path.parent / path for path in paths]
            for test, paths in manifest.items()
        }
        errors, outcomes = {}, {}

    integrated = (
        _integrated_revision(repo, revision, run, stable_ref) if stable else None
    )
    index = SourceIndex(sources_at(repo, revision))
    fragments = {}
    for test, paths in accepted.items():
        try:
            fragment = parse_speedscope(paths, remote_root, index)
            if run is not None:
                fragment = replace(
                    fragment,
                    seconds=execution_seconds(outcomes[test]),
                    timing_source="execution",
                )
            fragments[test] = fragment
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors[test] = str(exc)

    if fragments or run is not None:
        updated = graph.replacing(fragments, revision)
        store.publish(updated, revision)
        if integrated:
            store.publish(updated, integrated, stable=True)
    return {
        "revision": revision,
        "updated": sorted(fragments),
        "preserved_on_error": errors,
        "provenance": "verified" if run is not None else "manual_unverified",
        "published_revision": integrated or revision,
    }
