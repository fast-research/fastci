# Data formats

JSON files use UTF-8. Generated plans, state, and traces can contain source paths and test metadata; they are runtime artifacts rather than repository fixtures. All checked-in examples are synthetic.

## Test catalog and reduction policy

The catalog has `schema: 1` and a `tests` array in baseline execution order. Required per-test field: `name`. Optional fields and defaults:

| Field | Default | Meaning |
| --- | --- | --- |
| `command` | `[]` | Runner argv; required for execution |
| `gpus` | `0` | Nonnegative integer device count |
| `hardware` | `"cpu"` | Resource-pool label |
| `seconds` | `null` | Historical execution duration estimate |
| `metadata` | `{}` | Validation semantics; `task_type` enables equivalence checks |
| `workload` | `{}` | Original dimension values |
| `bindings` | `{}` | Per-dimension `kind`, `key`, and optional argument `style` |
| `env` | `{}` | String environment overrides |
| `timeout` | `3600` | Positive per-attempt timeout in seconds |

`metadata.resources_unresolved: true` prevents execution and marks resource cost unknown. Device visibility belongs to the resource pool and cannot be set in `env` or workload bindings.

Policies contain `reductions`, a map from dimensions to reduced values, and optional `path_dimensions`, a map from entity-name glob patterns to protected dimensions. See the full catalog example in the [README](../README.md).

## Native runtime graph

```json
{
  "schema": 1,
  "revision": "base-revision",
  "tests": {
    "smoke": {
      "nodes": ["model.forward", "test_smoke.run"],
      "edges": [["test_smoke.run", "model.forward"]],
      "seconds": 10
    }
  }
}
```

Each edge endpoint must occur in `nodes`. Isolated nodes are retained. Revision filenames in a graph store are hashes of revision identifiers; `stable.json` is the optional fallback. `import-graph` also accepts the legacy `job_registry`, `node_coverage`, `callees`, and optional `execution_times` representation.

Fragments optionally contain `timing_source`: `execution` identifies duration from verified execution results; the default `trace` is only an approximation. Verified execution duration takes priority over a catalog estimate when choosing representatives.

## Trace manifest

```json
{
  "smoke": ["traces/smoke-rank0.speedscope"],
  "distributed": ["traces/distributed-rank0.speedscope"]
}
```

Paths resolve relative to the manifest. Supply sampled Speedscope profiles and set `--remote-root` to the repository root used by the traced environment. The source repository and revision supplied to `ingest` must correspond to those traces.

For verified ingestion, pass `--plan` and `--run`. Each supplied test must have completed execution, and all captured files for that test must match their recorded SHA-256 values. A subset of tests is allowed; omitting files within a supplied test is not. `run` writes this manifest automatically as `*.traces.json`. Stable publication additionally requires `--stable-ref` and an identical tested/integrated source tree. The ingestion report distinguishes tested `revision` from `published_revision` when merge or squash changes commit identity.

## Replay JSON

```json
[
  {
    "run_id": "run-001",
    "sequence": 1,
    "files": ["model.py"],
    "tests": ["smoke", "distributed"],
    "outcomes": {"smoke": "pass", "distributed": "fail"},
    "seconds": {"smoke": 2, "distributed": 5},
    "gpus": {"smoke": 1, "distributed": 2}
  }
]
```

Run IDs and sequence values must be unique. `tests` defines the original order and requires a complete `pass`/`fail` outcome for every member. `seconds` and `gpus` are optional; incomplete prefix cost evidence produces `null`, rather than an invented estimate. Example: [history.json](../examples/history.json).

## Portable result CSV

`prepare-history` consumes one observation per row. Required columns:

| Column | Meaning |
| --- | --- |
| `run_id` | Run identifier |
| `sequence` | Integer chronological position of that run |
| `test` | Test name from the supplied catalog |
| `status` | `pass`, `fail`, `skipped`, `timeout`, `error`, or `cancelled` |
| `files` | JSON array of changed repository-relative paths |

An optional `seconds` column gives a finite, nonnegative execution duration. The catalog supplies device counts. All rows in a run must agree on sequence and changed files. The first terminal pass/fail row for a test is retained; later retry observations are counted and omitted. Row order within a run defines baseline test order.

Tests outside the supplied catalog and rows without a terminal pass/fail outcome are counted and omitted. Malformed rows, inconsistent run metadata, and duplicate sequences across different runs raise errors. A sidecar `*.diagnostics.json` reports omissions. Runs containing only unobserved outcomes cannot contribute replay records. Example: [history.csv](../examples/history.csv).

## Plans, outcomes, and state

`plan` produces an inspectable JSON document containing source revisions, affected and retained sets, execution order, selection reasons, representative groups, risk components, workload proposals, and the catalog. Its base-graph sidecar preserves the evidence used for that decision. A history digest prevents execution against a risk state that has advanced since planning.

`run` outputs per-test outcomes and per-attempt status, exit code, duration, GPU count, workload overrides, and log location. `comparison_agreed` identifies paired-validation agreement. The aggregate cost includes all attempts, including original/reduced pairs.

Executed outcomes include `started_seconds` and `finished_seconds`, relative to the run's dispatch clock. These bound the interval during which resources are reserved; skipped tests have null times. Aggregate `wall_seconds` differs from summed attempt time when jobs overlap. With `--resource-mode mock`, aggregate `measured_gpu_seconds` is null and `mock_device_seconds` reports CPU mock-slot work. Mock replay uses `serial_prefix_mock_device_seconds` instead of the GPU prefix metric.

Attempts include `trace_files` with path/hash pairs and `trace_errors` diagnostics. `source_verified`, `plan_digest`, and `effective_mode` bind execution to the plan and rollback mode. `reserved_gpu_seconds` includes the whole reservation interval (including between paired attempts); mock runs use `mock_reserved_device_seconds`. These duration-based accounting fields are not GPU utilization measurements.

State contains prior file/test associations, recent outcomes, observed run IDs, and workload-guard entries. Use unique run IDs and a single writer. Preserve the state alongside the graph store when continuing a CI history.
