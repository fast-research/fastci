# Complete mock CI walkthrough

Run the full scenario through the actual command-line interface:

```sh
python -u examples/mock_ci.py --out .fastci/complete-ci
```

Use a fresh output directory. The requirements are Python 3.10+, Git, and POSIX process handling. No GPU, network, private source, or optional dependency is required. The command exits nonzero if a workflow invariant fails. Three intentionally failing CI runs are expected; the overall scenario succeeds when each run has the specified outcome.

## End-to-end lifecycle

The fixture project includes an `Encoder`, linear/activation operations, CPU reductions representing data/tensor-parallel paths, checkpoint serialization, a size-dependent compiler path, input batches, and a cache. Tests execute actual numerical assertions in Python subprocesses. This is a small executable fixture, not an LLM training benchmark.

```text
new feature branch and source commit
  -> fastci plan against main
  -> fastci run using persisted risk history and workload guard
  -> real CPU assertions + observed Python call stacks
  -> fast-forward main only if the authoritative CI outcome passes
  -> verified fastci ingest using the plan/run, captured-file hashes, and integration ref
  -> next MR consumes updated evidence and previous observations
  -> export actual outcomes through JSON and CSV
  -> replay both histories and compare results
```

There is no prefilled risk history or manually constructed bootstrap graph. The first baseline run supplies initial outcomes, observed execution durations, and call evidence. The following rejected MR supplies the failure used by risk scoring. The mock's fast-forward history is supplemented by separate real Git merge/squash regression tests described in the [core audit](paper-audit.md).

## Real execution and mock boundaries

| Part | Behavior |
| --- | --- |
| Analysis, selection, pruning, risk, workload guard | Actual package functions through the CLI |
| Source revisions and integration | Actual Git branches/commits; failing revisions do not enter main |
| Test and paired results | Real CPU assertions and subprocess exit codes |
| Scheduling | Actual concurrent dispatch over four named mock slots in two pools |
| Traces | Actual Python call instrumentation exported as Speedscope |
| Missing evidence | Two named entities deliberately removed from captured stacks |
| Historical costs | Synthetic catalog values used to choose representatives |
| Regressions and malformed trace | Explicit fixture fault injections |
| GPU measurements | None; measured GPU work is null |

Call instrumentation substitutes for py-spy to avoid attach permissions and make the example deterministic. It is not statistical sampling and cannot establish sampling recall or overhead. The `--resource-mode mock` flag labels the work and changes cost reporting; it does not emulate CUDA or rewrite test programs. The fixture commands are CPU-only.

## Test catalog

The initial catalog has 12 jobs, with a thirteenth registered later:

| Jobs | Resources | Purpose |
| --- | --- | --- |
| smoke_slow, smoke_fast | 1 compute slot each | Equivalent context and metadata; minimum-cost representative wins |
| distributed_dp, distributed_tp | 2 compute slots each | Distinct parallel validation scopes |
| long_context | 1 memory slot | Observed validation path protects sequence length |
| checkpoint_roundtrip | 1 memory slot | Protected model-size dimension and state serialization |
| shape_guard | 1 compute slot | Intentionally wrong scope annotation; initial comparison revokes reduction |
| compiled_train | 1 compute slot | Safe reduction becomes unsafe after a later code change |
| mixed_precision | 1 compute slot | Distinct metadata and floating-point inputs |
| evaluation | 1 memory slot | Distinct evaluation path |
| loader_cpu, metrics_cpu | CPU only | Unaffected jobs whose evidence survives partial runs |
| new_cache | CPU only | New job without previous evidence |

Each hardware pool has two slots. At most four worker tasks run concurrently. Actual dispatch intervals must show overlapping jobs without double allocation; two-slot jobs queue until a complete allocation is available.

## Scenario sequence

| Scenario | Expected behavior |
| --- | --- |
| bootstrap | Run all 12 original workloads; capture calls and initialize stable evidence/history |
| failed_tp | Introduce a tensor-reduction bug; reject the failing MR |
| model_a | Select 10 model jobs and retain 9; prioritize the previously failing TP test, remove the slower equivalent smoke test, revoke an unsafe shape reduction |
| model_b | Remove exact-version materialization to exercise stable fallback; reuse accepted reductions and restore revoked ones |
| compile_regression | Original model fails while reduced model passes; the periodic comparison revokes the reduction and rejects the MR |
| compile_repair | The compiler test runs at the restored original workload and passes |
| class_expansion | Recover tests for an unobserved method through observed siblings, without fabricating its context |
| caller_expansion | Recover tests for an unobserved function through its static caller |
| config_and_new_test | Use a three-job baseline, retain a new untraced job, and preserve old checkpoint evidence after a malformed trace |
| unknown_entity | An unobserved, uncalled hook triggers the full 13-job baseline |
| rollback_failfast | Baseline mode with one worker; a failure leaves two pending jobs skipped, without fabricated pass observations |

The guard interval is 2 for this fixture so accepted/reduced/paired/revoked transitions occur quickly. The package default remains 10. The example checks detection when periodic comparison is due; it does not show that every size-dependent regression is detected immediately.

The current scenario executes 97 real subprocess attempts, including original/reduced comparisons. Mock-slot costs and synthetic historical estimates do not measure GPU savings.

## Output artifacts

| Artifact | Contents |
| --- | --- |
| report.md | Scenario table and the 22 verified invariants |
| summary.json | Selection, risk components, outcomes, dispatch timing, and trace updates |
| commands.json | Actual CLI commands, output, and exit codes |
| project/ | Synthetic source repository and feature-branch history |
| runs/SCENARIO/plan.json | Modified entities, groups, order, overrides, and base snapshot |
| runs/SCENARIO/run.json | Exit codes, workloads, logs, and reservation intervals |
| runs/SCENARIO/ingest.json | Replaced and preserved-on-error fragments |
| traces/ | Speedscope files from individual subprocesses |
| state.json, graphs/ | Persisted risk/workload state and versioned evidence |
| observations.json, observations.csv | Only executed terminal pass/fail observations |
| replay.json, csv-replay.json | Equal chronological replays labeled as mock device work |

Integration tests execute this workflow and inspect actual traces, branch contents, guard transitions, missing observations, and pairwise reservation intervals. Runtime output contains local paths and belongs under the ignored .fastci/ directory; it is not a checked-in benchmark dataset.
