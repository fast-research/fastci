from __future__ import annotations

import math
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from .model import read_json, write_json


def _reachable(adjacency, node, depth):
    seen, frontier = {node}, {node}
    for _ in range(depth):
        frontier = {
            neighbor for current in frontier for neighbor in adjacency.get(current, ())
        } - seen
        if not frontier:
            break
        seen.update(frontier)
    return seen


@dataclass
class Evidence:
    nodes: set[str] = field(default_factory=set)
    edges: set[tuple[str, str]] = field(default_factory=set)
    seconds: float | None = None
    timing_source: str = "trace"

    def __post_init__(self):
        if self.seconds is not None and (
            not math.isfinite(self.seconds) or self.seconds < 0
        ):
            raise ValueError("Evidence duration must be finite and nonnegative")


@dataclass
class Graph:
    # Test-specific fragments make replacement semantics explicit.
    tests: dict[str, Evidence] = field(default_factory=dict)
    revision: str | None = None

    def covering(self, node: str) -> set[str]:
        return {name for name, evidence in self.tests.items() if node in evidence.nodes}

    def coverage_by_node(self) -> dict[str, set[str]]:
        coverage = defaultdict(set)
        for name, evidence in self.tests.items():
            for node in evidence.nodes:
                coverage[node].add(name)
        return dict(coverage)

    def context(self, node: str, test: str, depth: int = 3) -> tuple:
        if depth < 0:
            raise ValueError("Context depth must be nonnegative")
        evidence = self.tests[test]
        if node not in evidence.nodes:
            raise ValueError("Cannot infer a context for an unobserved entity")
        forward, backward = defaultdict(set), defaultdict(set)
        for caller, callee in evidence.edges:
            forward[caller].add(callee)
            backward[callee].add(caller)
        vertices = _reachable(forward, node, depth) | _reachable(backward, node, depth)
        edges = {(a, b) for a, b in evidence.edges if a in vertices and b in vertices}
        return tuple(sorted(vertices)), tuple(sorted(edges))

    def replacing(self, fragments: dict[str, Evidence], revision: str) -> Graph:
        result = deepcopy(self)
        result.tests.update(deepcopy(fragments))
        result.revision = revision
        return result

    def to_json(self) -> dict:
        return {
            "schema": 1,
            "revision": self.revision,
            "tests": {
                name: {
                    "nodes": sorted(e.nodes),
                    "edges": sorted(e.edges),
                    "seconds": e.seconds,
                    "timing_source": e.timing_source,
                }
                for name, e in sorted(self.tests.items())
            },
        }

    @classmethod
    def from_json(cls, data: dict) -> Graph:
        if not isinstance(data, dict):
            raise ValueError("Graph must be a JSON object")
        if "job_registry" in data:
            return cls._from_legacy(data)
        if data.get("schema") != 1:
            raise ValueError("Unknown graph schema")
        tests = {}
        if not isinstance(data.get("tests"), dict):
            raise ValueError("Graph tests must be a mapping")
        for name, row in data["tests"].items():
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("nodes"), list)
                or not isinstance(row.get("edges"), list)
            ):
                raise ValueError("Invalid graph fragment")
            if not all(isinstance(node, str) for node in row["nodes"]):
                raise ValueError("Graph nodes must be strings")
            if not all(
                isinstance(edge, (list, tuple))
                and len(edge) == 2
                and all(isinstance(n, str) for n in edge)
                for edge in row["edges"]
            ):
                raise ValueError("Graph edges must contain two string endpoints")
            edges = {tuple(edge) for edge in row["edges"]}
            if any(len(edge) != 2 for edge in edges):
                raise ValueError("Graph edges must have two endpoints")
            nodes = set(row["nodes"])
            if any(a not in nodes or b not in nodes for a, b in edges):
                raise ValueError("Graph edges refer to missing nodes")
            tests[name] = Evidence(
                nodes, edges, row.get("seconds"), row.get("timing_source", "trace")
            )
        return cls(tests, data.get("revision"))

    @classmethod
    def _from_legacy(cls, data: dict) -> Graph:
        # Compatibility with the legacy integer-indexed graph schema.
        if not all(
            isinstance(data.get(key, {}), dict)
            for key in ("job_registry", "node_coverage", "callees", "execution_times")
        ):
            raise ValueError("Invalid legacy graph mappings")
        if not all(isinstance(children, dict) for children in data["callees"].values()):
            raise ValueError("Invalid legacy graph adjacency")
        names = {int(i): name for name, i in data["job_registry"].items()}
        tests = {
            name: Evidence(seconds=data.get("execution_times", {}).get(name))
            for name in names.values()
        }
        for node, ids in data["node_coverage"].items():
            for i in ids:
                if node != "root":
                    tests[names[int(i)]].nodes.add(node)
        for a, children in data["callees"].items():
            for b, ids in children.items():
                if a == "root" or b == "root":
                    continue
                for i in ids:
                    e = tests[names[int(i)]]
                    e.edges.add((a, b))
                    e.nodes.update((a, b))
        return cls(tests)


class GraphStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.load_errors: list[str] = []

    def path(self, revision: str) -> Path:
        # Hash instead of embedding refs/branch names in filesystem paths.
        from .model import digest

        return self.root / (digest(revision) + ".json")

    def load(self, merge_base: str) -> tuple[Graph, str]:
        self.load_errors = []
        for path, origin in (
            (self.path(merge_base), "merge_base"),
            (self.root / "stable.json", "stable"),
        ):
            if path.exists():
                try:
                    graph = Graph.from_json(read_json(path))
                    if origin == "merge_base" and graph.revision != merge_base:
                        raise ValueError(
                            "Graph revision does not match the requested merge base"
                        )
                    return graph, origin
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    self.load_errors.append(f"{origin}: {exc}")
        return Graph(), "missing"

    def publish(self, graph: Graph, revision: str, stable: bool = False) -> None:
        graph = deepcopy(graph)
        graph.revision = revision
        write_json(self.path(revision), graph.to_json())
        if stable:
            write_json(self.root / "stable.json", graph.to_json())
