from __future__ import annotations

import ast
import difflib
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from .repository import changed_paths, merge_base, resolve_commit
from .repository import git as git
from .repository import sources_at as sources_at


def entity(path: str, qualified: str) -> str:
    return path.removesuffix(".py").replace("/", ".") + "." + qualified


def jobs_by_file(paths, coverage):
    """Map sampled qualified names to the longest known Python module prefix."""
    modules = {
        path.removesuffix(".py").replace("/", "."): path
        for path in paths
        if path.endswith(".py")
    }
    result = defaultdict(set)
    for node, jobs in coverage.items():
        prefix = node.rpartition(".")[0]
        while prefix:
            if prefix in modules:
                result[modules[prefix]].update(jobs)
                break
            prefix = prefix.rpartition(".")[0]
    return result


@dataclass
class Function:
    path: str
    name: str
    start: int
    end: int
    owner: str | None
    calls: set[str] = field(default_factory=set)
    references: set[str] = field(default_factory=set)

    @property
    def key(self) -> str:
        return entity(self.path, self.name)


class SourceIndex:
    def __init__(self, sources: dict[str, str]):
        self.sources = sources
        self.functions: dict[str, Function] = {}
        self.definitions: list[Function] = []
        self.errors: list[str] = []
        for path, text in sources.items():
            try:
                tree = ast.parse(text)
            except (SyntaxError, ValueError):
                self.errors.append(path)
                continue
            self._visit(path, tree, [], None)
        self.by_file: dict[str, list[Function]] = {}
        for function in self.definitions:
            self.by_file.setdefault(function.path, []).append(function)

    def _visit(
        self, path: str, node: ast.AST, scope: list[str], owner: str | None
    ) -> None:
        if isinstance(node, ast.ClassDef):
            scope = scope + [node.name]
            owner = ".".join(scope)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualified = ".".join(scope + [node.name])
            calls = {
                call.func.id if isinstance(call.func, ast.Name) else call.func.attr
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
                and isinstance(call.func, (ast.Name, ast.Attribute))
            }
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            f = Function(
                path, qualified, start, node.end_lineno or node.lineno, owner, calls
            )
            f.references = {
                item.id if isinstance(item, ast.Name) else item.attr
                for item in ast.walk(node)
                if isinstance(item, (ast.Name, ast.Attribute))
                and isinstance(item.ctx, ast.Load)
            }
            self.definitions.append(f)
            if f.key in self.functions:
                f.calls.update(self.functions[f.key].calls)
                f.references.update(self.functions[f.key].references)
            self.functions[f.key] = f
            scope = scope + [node.name]
            # A nested function is not a sibling method of its enclosing class.
            owner = None
        for child in ast.iter_child_nodes(node):
            self._visit(path, child, scope, owner)

    def in_file(self, path: str) -> set[str]:
        return {f.key for f in self.by_file.get(path, [])}

    def at_line(self, path: str, line: int) -> Function | None:
        candidates = [f for f in self.by_file.get(path, []) if f.start <= line <= f.end]
        return min(candidates, key=lambda f: f.end - f.start) if candidates else None

    @cached_property
    def _expansion_index(self):
        # Build the reverse name references and per-file alias links once. The
        # same revision may expand hundreds of unobserved entities in one plan.
        references: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for key, function in self.functions.items():
            for name in function.calls | function.references:
                references[name][function.path].add(key)
        links = {}
        for path, source in self.sources.items():
            if path in self.errors:
                continue
            imports: dict[str, set[str]] = defaultdict(set)
            assignments: dict[str, set[str]] = defaultdict(set)
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.asname:
                            imports[alias.name].add(alias.asname)
                elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                    value = node.value
                    name = (
                        value.id
                        if isinstance(value, ast.Name)
                        else (value.attr if isinstance(value, ast.Attribute) else None)
                    )
                    if name is not None:
                        targets = (
                            node.targets
                            if isinstance(node, ast.Assign)
                            else [node.target]
                        )
                        assignments[name].update(
                            t.id for t in targets if isinstance(t, ast.Name)
                        )
            if imports or assignments:
                links[path] = imports, assignments
        return references, links

    def expand(self, key: str) -> set[str]:
        f = self.functions.get(key)
        if f is None:
            return set()
        if f.owner:
            return {
                k
                for k, other in self.functions.items()
                if other.path == f.path and other.owner == f.owner
            }
        # Name-based resolution is conservative for aliases and attribute dispatch.
        short = f.name.rsplit(".", 1)[-1]
        references, links = self._expansion_index
        result = set().union(*references.get(short, {}).values())
        for path, (imports, assignments) in links.items():
            aliases = {short, *imports.get(short, ())}
            pending = list(aliases)
            while pending:
                name = pending.pop()
                new = assignments.get(name, set()) - aliases
                aliases.update(new)
                pending.extend(new)
            for alias in aliases - {short}:
                result.update(references.get(alias, {}).get(path, ()))
        return result - {key}


@dataclass
class Change:
    merge_base: str
    revision: str
    files: list[str]
    entities: list[str]
    unresolved: list[str] = field(default_factory=list)


def changed_entities(path: str, old: SourceIndex, new: SourceIndex) -> set[str]:
    before = old.sources.get(path, "").splitlines()
    after = new.sources.get(path, "").splitlines()
    selected: set[str] = set()
    global_change = False
    for tag, i, j, k, new_end in difflib.SequenceMatcher(
        None, before, after, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            continue
        for index, lines, start, end in ((old, before, i, j), (new, after, k, new_end)):
            for line in range(start, end):
                if not lines[line].strip() or lines[line].lstrip().startswith("#"):
                    continue
                function = index.at_line(path, line + 1)
                if function:
                    selected.add(function.key)
                else:
                    global_change = True
    if global_change:
        # All entities in BOTH versions, including removed functions.
        selected.update(old.in_file(path) | new.in_file(path))
    return selected


def analyze(
    repo: str | Path,
    source_ref: str,
    target_ref: str,
    *,
    index_at: Callable[[str], SourceIndex] | None = None,
) -> tuple[Change, SourceIndex, SourceIndex]:
    revision = resolve_commit(repo, source_ref)
    base = merge_base(repo, revision, target_ref)
    files = changed_paths(repo, base, revision)
    # Rename represented as old deletion + new addition, preserving base evidence.
    if index_at is None:
        old = SourceIndex(sources_at(repo, base))
        new = old if revision == base else SourceIndex(sources_at(repo, revision))
    else:
        # A newest-first replay has already indexed this revision as the
        # previous commit's base. Access it before loading the next parent.
        new, old = index_at(revision), index_at(base)
    entities, unresolved = set(), []
    for path in files:
        if not path.endswith(".py"):
            continue
        if path in old.errors or path in new.errors:
            unresolved.append(path + ":parse_error")
            continue
        found = changed_entities(path, old, new)
        entities.update(found)
        if not found and old.sources.get(path) != new.sources.get(path):
            # Includes files with only globals. Comment-only changes also use
            # baseline conservatively rather than claiming an empty safe set.
            unresolved.append(path + ":no_mapped_entities")
    return Change(base, revision, files, sorted(entities), unresolved), old, new
