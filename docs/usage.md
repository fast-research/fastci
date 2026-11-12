# FastCI CLI guide

Use Python 3.10 or later, Git, and a POSIX environment. Install from the FastCI checkout with `python -m pip install -e .`. For automatic runtime sampling, run `python -m pip install -e '.[trace]'` and ensure py-spy can attach to the target process.

## Workflow

```text
Git changes + base runtime graph + test catalog + prior outcomes
                         |
                  Evidence selection
                         |
                  Context pruning
                         |
                    Risk ordering
                         |
          Workload overrides and paired validation
                         |
                  Existing CI runner
                         |
        Observed results + per-test trace replacement
```

Generate a plan before execution. Planning reads immutable Git versions and does not advance history. After execution, record observed pass/fail outcomes and update the versioned graph using successfully parsed traces. Preserve evidence for tests that did not run or could not be traced.

## Describe your test suite

Provide a JSON catalog in the original execution order:

```json
{
  "schema": 1,
  "tests": [{
    "name": "distributed_smoke",
    "command": ["python", "train_test.py", "--layers", "12"],
    "gpus": 2,
    "hardware": "gpu",
    "seconds": 120,
    "timeout": 600,
    "metadata": {
      "task_type": "distributed_execution",
      "validation_dimensions": ["parallelism"],
      "unrelated_dimensions": ["layers"]
    },
    "workload": {"parallelism": 2, "layers": 12},
    "bindings": {"layers": {"kind": "arg", "key": "--layers"}}
  }]
}
```

- `command` is an argv array, executed relative to the source checkout. Shell expansion is not implicit.
- `gpus` and `hardware` define the required resource pool. CPU tests use zero GPUs.
- `seconds` is an initial execution-time estimate. Verified execution measurements in the graph take precedence; otherwise catalog estimates precede trace-duration approximations. Unknown costs are not treated as zero.
- `metadata` must describe validation semantics, task, and environment. Metadata, hardware, GPU count, and relevant workload values must match for equivalence. Missing `task_type`, unresolved hardware, or an unknown declared validation-dimension value prevents cross-test pruning. Include those dimension values in `workload` even when they have no reduction binding.
- `validation_dimensions` protects relevant dimensions; `unrelated_dimensions` explicitly identifies eligible reductions. Unclassified dimensions stay unchanged.
- `bindings` maps dimensions to existing runner arguments or environment variables. The runner must actually consume those values.

A reduction policy looks like:

```json
{
  "reductions": {"layers": 2, "sequence": 128, "logging": false},
  "path_dimensions": {"*.validate_long_sequence": ["sequence"]}
}
```

Observed paths matching a pattern protect the listed dimensions, even if metadata labels them unrelated. Argument bindings replace existing `--flag value` or `key=value` tokens. Add `"style": "equals"` to append new `key=value` arguments. Environment bindings use `"kind": "env"`. See [examples/rules.json](../examples/rules.json).

## Select tests and generate a plan

```sh
fastci plan \
  --repo /path/to/source --source feature --target main \
  --suite suite.json --graphs .fastci/graphs --state .fastci/state.json \
  --rules rules.json --out .fastci/plan.json
```

The source is compared against its merge base with the target. Selection uses that version's graph, falling back to a stable graph when unavailable. The plan records selection reasons, equivalence groups, risk components, proposed overrides, and an immutable base-graph snapshot.

`--baseline baseline.json` accepts a JSON array produced by your existing CI rules; it defaults to the full catalog. Non-Python changes retain that baseline. Mixed changes combine it with Python selections. Untraced tests are retained. If neither direct evidence nor expansion evidence is available, selection falls back to the baseline.

| Mode | Behavior |
| --- | --- |
| `full` | Selection, pruning, risk ordering, workload proposals |
| `select` | Selection and pruning |
| `no_prune` | Evidence selection without pruning, risk ordering, or reductions |
| `baseline` | Original CI test set and order |
| `drain` | New plans use baseline CI; previously admitted optimized plans can finish |
| `disabled` | Baseline execution with reductions, sampling, and FastCI state/graph updates disabled |

To control rollback through one field, write `{"mode": "full"}` to a configuration file and pass `--config fastci.json` to `plan`. Execution re-reads the same file. Changing it to `drain` prevents new optimized plans; changing it to `disabled` also reroutes an existing plan to its stored baseline and disables FastCI state writes and sampling. `ingest --plan ...` inherits that configuration; `sample --config fastci.json` and standalone hooks using `FASTCI_CONFIG` honor it as well. This controls future CLI stages; it does not kill already running jobs or stop an external CI service.

## Execute a plan

Your CI submission adapter can consume `order` and `proposals` from the plan. A local executor is also provided:

```sh
fastci run \
  --plan .fastci/plan.json --cwd /path/to/clean/source-checkout \
  --state .fastci/state.json --environment test-image-v1 \
  --run-id run-001 --pools pools.json --out .fastci/run.json
```

Example `pools.json`:

```json
{"gpu": ["0", "1", "2", "3"]}
```

Pool labels must match the catalog's `hardware` values. CPU-only execution needs no pool file. Device IDs must be unique across pools. The executor scans tests in priority order, dispatches those that fit, and sets `CUDA_VISIBLE_DEVICES` to the allocated devices. Containers, topology constraints, distributed launch, and dependency graphs belong to your existing runner.

Before execution, the CLI checks the source revision, clean tracked files, history digest, and frozen base snapshot. It checks source stability around each attempt, including between original/reduced comparisons. Provide fixed environment and input data through your CI runner. `--stop-on-failure` stops pending work; already running tests finish and remain counted.

A reduction receives an original/reduced paired run before acceptance and periodically thereafter. By default, recheck every 10 executions of that test; configure `guard.interval` in the state file. Disagreement restores the original workload and revokes the reduction for that policy/environment fingerprint. Timeouts and execution errors never approve a reduction. If both workloads fail, the test still fails. Both paired attempts contribute to execution-cost reporting.

State files use atomic replacement and require a single writer. A coordinator handling concurrent runs must serialize state updates. Local logs and state are stored under the output locations you supply.

## Collect runtime evidence

`run` automatically injects the packaged rank-0 sampler hook at 50 Hz for local execution. Set `--trace-dir` to choose a fresh trace directory, `--trace-rate` to change the rate, or `--trace-subprocesses` to enable following. Use `--no-trace` to opt out. Mock-resource mode uses externally instrumented fixture traces instead of launching py-spy. Missing sampling tools or trace output produce diagnostics rather than changing test pass/fail status.

Every attempt records captured-file hashes, and `run.json` has a companion `run.traces.json` manifest. Sampling code is included in the wheel; it chains an existing application `sitecustomize` before adding the sampler. Python commands using isolated/no-site startup or non-Python runners require an explicit integration and will otherwise report missing evidence.

For an external runner, attach to an existing rank-0 Python process:

```sh
fastci sample --pid 12345 --rate 50 --out trace.speedscope
```

Alternatively, use the opt-in `trace_hook/sitecustomize.py` from this checkout. In a dedicated CI Python environment, add `trace_hook` to `PYTHONPATH`, install the package, and set:

```sh
export PYTHONPATH="$PWD/trace_hook${PYTHONPATH:+:$PYTHONPATH}"
export FASTCI_TRACE=1
export FASTCI_TRACE_DIR=/path/to/per-test-traces
```

Sampling runs for `LOCAL_RANK=0`, or for a normal single-process program with no rank set. Optional variables are `FASTCI_TRACE_RATE` (default 50) and `FASTCI_TRACE_SUBPROCESSES=1`. The hook preserves an existing application `sitecustomize`. Attach permissions depend on the target environment.

Build a trace manifest mapping test names to Speedscope files:

```json
{"distributed_smoke": ["traces/rank0.speedscope"]}
```

Paths are relative to the manifest. Include only tests executed at the stated revision. Update from the plan's frozen snapshot:

```sh
fastci ingest \
  --snapshot .fastci/plan.base-graph.json --manifest .fastci/run.traces.json \
  --plan .fastci/plan.json --run .fastci/run.json \
  --repo /path/to/source --revision TESTED_COMMIT --remote-root /workspace/source \
  --graphs .fastci/graphs --out .fastci/ingest.json
```

Verified ingestion binds the revision, plan, base snapshot, test outcomes, and captured bytes. Include every captured file for each supplied test; modified, omitted, unexecuted, and incomplete-run evidence is rejected. Other valid tests can still update their fragments. The parser filters repository-external frames and resolves class-qualified names, including duplicate getter/setter definitions. Missing evidence preserves previous fragments; verified runs still receive a versioned snapshot even if every trace fails.

After integration, add `--stable --stable-ref main`. Stable publication requires a passing verified run and an integration tree identical to the tested tree. Merge/squash commits with new IDs receive their own graph version when the trees match. If integration changes the tree, test that revision first. Manual ingestion without `--plan/--run` remains available as explicitly unverified import and cannot promote a stable graph. Only supplied fragments are processed; historical trace directories are not scanned.

An integer-indexed legacy graph can also be converted:

```sh
fastci import-graph --input graph.json --graphs .fastci/graphs \
  --revision BASE_COMMIT --stable
```

See [Data formats](formats.md) for native graph and history schemas.

## Replay scheduling decisions

The replay input is a JSON array with one record per run: a unique `run_id`, unique chronological `sequence`, changed `files`, original-order `tests`, and complete pass/fail `outcomes` for that selected set. Optional `seconds` and `gpus` maps enable serial-prefix cost reporting.

```sh
fastci replay --input examples/history.json --warmup 0 --out .fastci/replay.json
```

The default warmup is 30 runs. Each run is scored before its outcomes update the history. All-pass runs also contribute history, and ties preserve the original order. Replay compares fixed observed test sets; it does not infer selection recall.

For CPU simulations of device pools, pass `--resource-mode mock` to `run` and `replay`. Execution then reports `mock_device_seconds` and sets `measured_gpu_seconds` to null; replay uses `serial_prefix_mock_device_seconds`. This labels simulations explicitly and does not emulate a GPU. Execution output also records reservation start/end times and wall time, allowing concurrent device-allocation checks.

For CSV input, use the public row format described in [Data formats](formats.md):

```sh
fastci prepare-history --csv examples/history.csv \
  --suite examples/replay-suite.json --out .fastci/history.json
fastci replay --input .fastci/history.json --warmup 0 --out .fastci/replay.json
```

Output includes first-failing-test rank and the GPU work of the serial prefix through that failure, when complete cost evidence exists. This is an ordering metric, not measured cluster feedback latency or cancellation savings.
