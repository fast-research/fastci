# Design and method alignment

This document defines how the implementation interprets FastCI's four core techniques. It describes behavior and explicit choices for details that the method leaves open.

## 1. Evidence-guided test selection

The runtime graph records function nodes, directed caller/callee edges, and the tests that observed each node and edge. Internally, the graph stores a fragment per test so that evidence replacement has clear semantics.

Source analysis compares immutable Git objects at the merge base and source revision. Python entities use repository-relative dotted names, such as `package.module.Class.method`. Function-scoped changes map to the narrowest matching function in either version. Changes outside function scope, including imports and module variables, map to all functions and methods in that file. Deleted entities and both sides of a rename retain their version-specific identities.

When an entity has direct evidence, select its observed tests. For missing evidence:

- A method expands to other methods in its class.
- A function expands to statically identified callers, including import aliases.

Static attribute calls are matched conservatively by name. This may overselect and does not establish runtime evidence for reflection or dynamically generated calls. By default, a missing expansion retains the configured baseline. The Megatron-LM adapter can use its observed file-to-test mapping to recover candidates for the affected file and keeps the union of available evidence across the change. It falls back to the configured baseline when the change has no scoped evidence. Tests with no usable graph fragment remain selected. Parse failures and unmapped changes receive explicit fallback reasons.

Static expansion includes import/assignment aliases, alias chains, and functions referencing the target as a callback. Reference-based recovery is deliberately conservative; a matching reference need not be an actual invocation.

Non-Python changes reuse a caller-supplied baseline from existing CI rules. Mixed changes take the union with Python selections. The baseline defaults to the full catalog.

Implementation files include `repository.py`, `source.py`, and `selection.py`.

## 2. Context-aware pruning

For each modified entity and directly observed test, obtain callers within K reverse-directed hops and callees within K forward-directed hops. The default is **K=3**. Take the union of those vertices, including the center, and retain all test-specific directed edges whose endpoints are in that union.

Two tests are equivalent for an entity only when the exact vertex and edge sets match and their validation metadata, hardware label, and GPU count match. Comparing only BFS levels would lose topology and is deliberately avoided. Missing `task_type` means validation scope is unknown, so distinct tests cannot be declared equivalent.

Equivalence also compares actual workload values for validation dimensions and dimensions not explicitly labeled unrelated. An unknown declared dimension value or unresolved resource requirement prevents equivalence. Repeated definitions of the same source entity, such as property getters/setters, retain separate source spans while sharing the entity identity.

Choose the minimum estimated GPU-cost representative in each group:

```text
cost(test) = historical_execution_seconds × required_GPU_count
```

Prefer verified execution duration from the graph; otherwise use catalog time and then trace-derived approximation. For a paired run, the authoritative workload attempt supplies the future execution estimate, while both attempts contribute to run-cost accounting. Unknown costs sort after known costs; ties retain catalog order. CPU tests have zero GPU cost. Prune independently per modified entity, then union the representatives.

If a modified entity is absent from the graph, its context is unknown. Tests discovered through expansion remain selected; equality of a sibling's context is not used as proof of equivalence for the missing entity.

Code: `graph.py`, `selection.py`.

## 3. Risk-aware ordering

For each changed file `f` and test `t`, retain the number of prior observed executions `n[f,t]` and failures `e[f,t]`. The code-change signal is:

```text
R_code(t) = max over changed files f of e[f,t] / n[f,t]
```

Unobserved pairs contribute zero. Let `f*` attain the maximum positive rate:

```text
alpha(t) = n[f*,t] / (n[f*,t] + lambda)
lambda = 5
```

Without a positive code-change rate, set `alpha=0`. Equal positive file rates choose the first file in lexical order, making an otherwise unspecified tie deterministic.

The instability signal uses up to **W=5** recent outcomes, oldest to newest, with failure encoded as one:

```text
weighted_failure_rate = sum(gamma ** (L-i) * outcome[i]) / sum(gamma ** (L-i))
gamma = 0.8
```

Attenuate this rate by 0.85 after at least two consecutive passes, or by 0.6 after at least four. No history gives zero. The final risk is:

```text
R(t) = alpha(t) * R_code(t) + (1 - alpha(t)) * R_instability(t)
```

Sort descending with stable test-order ties. Scoring does not mutate history. Only observed terminal pass/fail outcomes contribute; skipped, timed-out, or unavailable results do not become passes.

Replay processes unique sequence values in increasing order. Each run is scored before updating history, including during the default 30-run warmup. All-pass runs remain part of the history. This is sequential offline replay, not simulation of overlapping production event times.

Code: `risk.py`, `replay.py`.

## 4. Workload-optimized execution

A policy proposes concrete configuration overrides. Eligibility requires the dimension to be explicitly listed as unrelated, to have a binding to the existing runner, and to be absent from protected metadata or observed-path dimensions. Unknown dimensions stay unchanged. Function names alone never prove that a dimension is unrelated.

Before first acceptance, execute the original and reduced workloads on the same source checkout, environment, and device allocation. Approve only matching terminal pass/fail outcomes. Both failing means the test still fails. A mismatch uses the original result and revokes that reduction. Timeouts or executor errors cannot approve it.

The default recheck interval is 10 executions of that test, an implementation parameter rather than a claimed method constant. Accepted reductions are periodically compared again. A revoked policy remains disabled until its configuration, test specification, or environment fingerprint changes. Reuse a stable, meaningful environment identity when invoking the runner.

Configuration overrides do not modify test code. Argument bindings support separate-value flags and `key=value`; environment bindings are passed to the existing command. Measured cost includes both attempts of paired validation.

Every occurrence of a bound argument is replaced, including mixed repeated flag styles. Source identity and tracked-file cleanliness are checked around each attempt, preventing a source-mutating baseline from approving a comparison on different code.

Code: `workload.py`, `runner.py`.

## Evidence lifecycle

The sampler defaults to 50 Hz and Speedscope output. Repository-external frames are filtered, paths normalized, and class-qualified names resolved from the matching source AST when possible. Consecutive retained frames produce directed edges. Multiple profiles and trace files contribute one test fragment. Overlapping profile durations are not summed as independent wall-clock work.

The graph associated with the merge base is preferred; a stable graph is the fallback. Planning writes a frozen base snapshot. Ingestion builds and validates each supplied fragment before replacing that test's old evidence. Untested or unsuccessfully traced tests retain prior fragments. The result is stored under the tested revision and can be promoted to stable after integration.

Corrupt or wrong-version snapshots fall back with diagnostics rather than silently supplying evidence. Normal execution injects a packaged rank-0 hook, preserves application startup, captures per-attempt trace hashes, and emits a manifest. Verified ingestion checks the run/plan/revision, base snapshot digest, complete captured trace set, and terminal outcomes. Timed-out, modified, or unrelated traces cannot replace evidence. Stable promotion checks the integrated tree against the tested tree and publishes under the integration commit, including content-equivalent merge/squash IDs. A differing integration tree requires another test run. Unverified manual import is explicitly labeled and cannot promote stable evidence.

The single `mode` field in an optional configuration implements drain and disable behavior at CLI boundaries. Drain allows admitted plans to finish while new plans use baseline CI. Disable reroutes existing plans, suppresses sampling, bypasses corrupt optimizer state, and prevents FastCI history/graph writes. It does not terminate external services or already executing jobs.

Only requested trace fragments are processed. Filesystem updates are atomic, but state commands require a single writer; distributed locking and event coordination belong to the CI controller.

Implementation files include `sampler.py`, `traces.py`, `ingestion.py`, `provenance.py`, and `graph.py`.

## Infrastructure and evaluation boundaries

The local executor scans the risk-ordered queue and dispatches fitting tasks to homogeneous device pools. This packing policy is infrastructure, not a separate research contribution. It does not implement distributed topology selection, containers, or CI dependency graphs.

Replay reports first-failing-test rank and optional serial-prefix GPU work through the failure. Those metrics do not measure cluster feedback latency, actual cancellation savings, test-selection recall, or real training speedups. Sampling, static expansion, context equality, and periodic outcome comparison also do not provide a formal no-regression guarantee.
