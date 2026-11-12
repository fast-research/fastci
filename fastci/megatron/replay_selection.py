"""Selection rules used by the archived Megatron-LM benchmark."""

from __future__ import annotations

import ast
from collections import defaultdict
from functools import cached_property, lru_cache

from ..source import entity, jobs_by_file


def changed_functions(files, before, after):
    modified, structural = {}, {}
    for path in files:
        if not path.endswith(".py"):
            structural[path] = "non_python"
        elif path not in before or path not in after:
            structural[path] = "added_or_deleted_module"
        elif before[path] != after[path]:
            try:
                old, old_structure = _functions(before[path])
                new, new_structure = _functions(after[path])
            except SyntaxError:
                structural[path] = "unparsed_module"
                continue
            if old_structure != new_structure:
                structural[path] = "module_structure"
            names = sorted(
                name
                for name in old.keys() | new.keys()
                if old.get(name) != new.get(name)
            )
            if names:
                modified[path] = names
    return modified, structural


def _functions(text):
    tree = ast.parse(text)
    functions = {}

    class Visitor(ast.NodeVisitor):
        scope: tuple[str, ...] = ()

        def _visit_scope(self, node):
            previous = self.scope
            self.scope += (node.name,)
            self.generic_visit(node)
            self.scope = previous

        visit_ClassDef = _visit_scope

        def visit_FunctionDef(self, node):
            functions[".".join((*self.scope, node.name))] = ast.dump(node)
            self._visit_scope(node)

        visit_AsyncFunctionDef = visit_FunctionDef

    class Skeleton(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node.body = [ast.Pass()]
            return node

        visit_AsyncFunctionDef = visit_FunctionDef

    Visitor().visit(tree)
    return functions, ast.dump(Skeleton().visit(tree))


class _CallSites(ast.NodeVisitor):
    def __init__(self):
        self.imports = {}
        self.calls = defaultdict(set)
        self.siblings = defaultdict(set)
        self.classes = ()
        self.caller = None

    def visit_ImportFrom(self, node):
        if node.module:
            for alias in node.names:
                self.imports[alias.asname or alias.name] = node.module

    def visit_Import(self, node):
        for alias in node.names:
            self.imports[alias.asname or alias.name] = alias.name

    def visit_ClassDef(self, node):
        previous = self.classes
        self.classes += (node.name,)
        self.generic_visit(node)
        self.classes = previous

    def visit_FunctionDef(self, node):
        previous = self.caller
        self.caller = ".".join((*self.classes, node.name))
        self.siblings[".".join(self.classes)].add(self.caller)
        self.generic_visit(node)
        self.caller = previous

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        if self.caller is not None:
            if isinstance(node.func, ast.Name):
                self.calls[node.func.id].add(self.caller)
            elif isinstance(node.func, ast.Attribute):
                self.calls[node.func.attr].add(self.caller)
        self.generic_visit(node)


@lru_cache(maxsize=4096)
def _call_sites(text):
    sites = _CallSites()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return sites, {}, set()
    sites.visit(tree)
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name
    definitions = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions.add(node.name)
        elif isinstance(node, ast.Assign):
            definitions.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
    return sites, aliases, definitions


class _Source:
    def __init__(self, sources):
        self.sources = sources

    @cached_property
    def importing(self):
        index = defaultdict(set)
        for path, text in self.sources.items():
            dirs = path.split("/")[:-1]
            if any(
                part.startswith(".") or part in {"__pycache__", "node_modules"}
                for part in dirs
            ):
                continue
            sites, _, _ = _call_sites(text)
            for name, module in sites.imports.items():
                index[name, module].add(path)
        return index

    def expansions(self, path, name):
        if "." in name:
            sites, _, _ = _call_sites(self.sources.get(path, ""))
            return {
                entity(path, sibling)
                for sibling in sites.siblings.get(name.rsplit(".", 1)[0], ())
            }
        module = path.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
        package = module.rpartition(".")[0]
        files = {path} | self.importing.get((name, module), set())
        if package:
            files.update(self.importing.get((name, package), ()))
        result = set()
        for caller_path in files:
            sites, aliases, definitions = _call_sites(self.sources.get(caller_path, ""))
            if caller_path != path and name in definitions:
                continue
            called_as = {
                name,
                *(alias for alias, original in aliases.items() if original == name),
            }
            for spelling in called_as:
                result.update(
                    entity(caller_path, caller)
                    for caller in sites.calls.get(spelling, ())
                )
        return result


class BenchmarkSelector:
    """Replay the benchmark's sampled function, file, and two-hop CEP rules."""

    def __init__(self, graph, reference_paths):
        self.jobs = {index: name for name, index in graph["job_registry"].items()}
        self.coverage = {node: set(ids) for node, ids in graph["node_coverage"].items()}
        self.forward = graph["callees"]
        self.backward = defaultdict(dict)
        for caller, children in self.forward.items():
            for callee, ids in children.items():
                self.backward[callee][caller] = ids
        self.by_file = jobs_by_file(reference_paths, self.coverage)
        self.representatives = lru_cache(maxsize=None)(self._representatives)

    def _representatives(self, node):
        groups = {}
        # The benchmark takes the first member in integer job-set traversal.
        for job in self.coverage[node]:
            signature = tuple(
                self._levels(edges, node, job)
                for edges in (self.forward, self.backward)
            )
            groups.setdefault(signature, job)
        return set(groups.values())

    @staticmethod
    def _levels(edges, node, job):
        seen, frontier, levels = {node}, {node}, []
        for _ in range(2):
            frontier = {
                child
                for parent in frontier
                for child, ids in edges.get(parent, {}).items()
                if job in ids
            } - seen
            if not frontier:
                break
            levels.append(frozenset(frontier))
            seen.update(frontier)
        return tuple(levels)

    def select(self, files, before, after):
        modified, structural = changed_functions(files, before, after)
        head = _Source(after)
        unpruned, pruned, missing = set(), set(), set(structural)
        for path, names in modified.items():
            for name in names:
                node = entity(path, name)
                observed = (
                    {node}
                    if node in self.coverage
                    else head.expansions(path, name) & self.coverage.keys()
                )
                candidates = set().union(*(self.coverage[n] for n in observed))
                unpruned.update(candidates)
                pruned.update(set().union(*(self.representatives(n) for n in observed)))
                if not candidates:
                    missing.add(path)
        file_jobs = set().union(*(self.by_file.get(path, set()) for path in missing))
        decisions = {
            "no_cep_no_file": unpruned,
            "no_cep_file": unpruned | file_jobs,
            "cep_no_file": pruned,
            "cep_file": pruned | file_jobs,
        }
        return {
            name: sorted(self.jobs[i] for i in ids) for name, ids in decisions.items()
        }, {
            "functions": modified,
            "fallback": [f"{reason}:{path}" for path, reason in structural.items()],
            "file_matches": sorted(missing & self.by_file.keys()),
            "unmatched_files": sorted(missing - self.by_file.keys()),
        }
