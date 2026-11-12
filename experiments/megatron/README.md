# Megatron-LM functional CI and reproduction

FastCI runs regular and checkpoint-resume pretraining recipes on a dedicated single node with eight homogeneous GPUs in a prepared CUDA environment. It reads the recipes and invokes the training and validation entry points from the [pinned Megatron-LM revision](https://github.com/NVIDIA/Megatron-LM/commit/d2b0e35cc4f05bc6b70734d558617050fbbb51ed).

Use the [two-case configuration](native.example.yaml) for a preview or the [47-case configuration](native.benchmark.yaml) for the historical suite. The benchmark configuration resolves its case references against the upstream [recipe definitions](https://github.com/NVIDIA/Megatron-LM/tree/d2b0e35cc4f05bc6b70734d558617050fbbb51ed/tests/test_utils/recipes) using the [recipe parser](https://github.com/NVIDIA/Megatron-LM/blob/d2b0e35cc4f05bc6b70734d558617050fbbb51ed/tests/test_utils/python_scripts/recipe_parser.py). The adapter delegates training to the upstream [functional runner](https://github.com/NVIDIA/Megatron-LM/blob/d2b0e35cc4f05bc6b70734d558617050fbbb51ed/tests/functional_tests/shell_test_utils/run_ci_test.sh) and its validation entry points.

## Recompute the historical table on CPU

From the FastCI checkout, use Python 3.10 or later.

```sh
python experiments/megatron/reproduce.py --out .fastci/megatron-replay
```

This reads the bundled data, scores the archived selection decisions, and checks `paper-table.tsv` against the reference table. `results.json` contains per-commit scores and the full-window and cohort summaries. Recall and modified-code coverage retention are averaged over the 47 commits with affected cases. GPU costs use all 100 commits and the measured unsampled case durations. The four graph-based strategies each pay the one-time cost of both graph collection attempts. The cohort summaries each charge the full graph build to show the cost of replaying that cohort on its own.

The evaluation uses one frozen call graph and line-coverage oracle for 77 commits at or before the oracle and 23 after it. Recall describes tests affected according to that oracle. Net GPU savings are estimates based on the matched GPU baseline. The replay scores saved decisions for the historical FastCI variants, Megatron L0, Codex Planner, and NameRTS. It runs on CPU without contacting a model, invoking NameRTS, or executing the historical GPU suite. See the [data provenance and schema](data/README.md).

To recompute the four FastCI variants from the public upstream source and the bundled 47-case graph, fetch the same first-parent window and pass the resulting Git directory. The command checks every computed test set against the server export and verifies that the full-window scores regenerate the reference table. It scores the archived L0, Codex Planner, and NameRTS decisions.
```sh
python -m pip install -e .
git init --bare .fastci/megatron-upstream.git
git -C .fastci/megatron-upstream.git fetch --depth=101 https://github.com/NVIDIA/Megatron-LM.git a7a6d2225700a8ee701453288894501d868c6bd4
python experiments/megatron/reproduce.py --repo .fastci/megatron-upstream.git --out .fastci/megatron-replay
```

`recomputed-selector.json` contains the recomputed decisions, changed functions, matched and unmatched files, and GPU cost breakdown. Use `--commit SHA` to verify one commit and `--jobs N` to set CPU parallelism. The archived table is still scored for all 100 commits.

The server benchmark uses function AST diffs, direct observations, and same-class or import-scoped caller expansion. The file-matching variants add all traced cases for a structurally changed file or a file with an unresolved modified function. The benchmark CEP groups each observed node by its two-hop forward and backward reachability levels and keeps the first job in the recorded job-ID traversal. It applies CEP to expansion evidence and adds file matches after CEP.

The native adapter follows the paper's pruning rule with three-hop directed context subgraphs, matching validation metadata, and the cheapest representative. Its selection adds file-scoped evidence when function expansion fails. A function with no sampled or file evidence is recorded as a diagnostic while other scoped evidence still contributes to the plan. Missing or incompatible graphs, unresolved source structure without file evidence, and changes without any scoped evidence use the configured baseline. Non-Python changes follow the existing baseline policy.

## Recheck the saved training prefixes on CPU

The bundle includes the first 25 steps of the golden and observed `lm loss` and `num-zeros` metrics for two 50-step training cases. This runs the [pinned upstream pytest](https://github.com/NVIDIA/Megatron-LM/blob/d2b0e35cc4f05bc6b70734d558617050fbbb51ed/tests/functional_tests/python_test_utils/test_pretraining_regular_pipeline.py) against the saved values.

```sh
python -m pip install -r experiments/megatron/requirements-prefix.txt
python experiments/megatron/reproduce_prefix.py --repo .fastci/megatron-upstream.git --out .fastci/megatron-prefix
```

Each case gets a pytest log and the script writes `prefix-verification.json`. The observed prefix comes from saved full-workload measurements. The historical GPU pilot also passed both reduced workloads.

## Preview and run native Megatron CI

Install the adapter in the prepared environment. The CPU preview needs Python, Git, Bash, tar, and the adapter extras. GPU runs additionally need the Megatron dependencies for the pinned recipe, a compatible PyTorch/CUDA, Transformer Engine and NCCL stack, Mike Farah yq v4, `envsubst`, `nvidia-smi`, and the selected launchers such as `ft_launcher`. Sampling requires `py-spy` with permission to attach to the training processes.

```sh
python -m pip install -e '.[megatron,trace]'
mkdir -p .fastci
cp experiments/megatron/native.example.yaml .fastci/megatron.yaml
MEGATRON_REPO="$PWD/.fastci/megatron-upstream.git"
BASE=d2b0e35cc4f05bc6b70734d558617050fbbb51ed

fastci-megatron plan --repo "$MEGATRON_REPO" --config .fastci/megatron.yaml \
  --source "$BASE" --target "$BASE" --mode baseline --out .fastci/bootstrap-plan.json
fastci-megatron run --repo "$MEGATRON_REPO" --plan .fastci/bootstrap-plan.json \
  --out .fastci/megatron-preview --dry-run
```

Set `MEGATRON_REPO` to an upstream checkout containing the reference revision if you already have one. The adapter exports tracked files from an immutable Git revision and runs upstream recipe code, so supply trusted source revisions. The preview writes an isolated tree and rendered Bash scripts.

For GPU execution, edit `megatron.yaml` to use the prepared Python, yq, data and tokenizer/cache locations. Set `environment_id` to identify the software and data inputs, and update it whenever those inputs change. The adapter resolves a common upstream recipe platform from the requested cases. If multiple platforms match, set `platform` to the label defined in the upstream recipe. You can use `hardware` to require a match against the local GPU names. Runtime compatibility checks also use the detected GPU and driver details.

Provision training assets and any hardware-matched golden overrides separately. If you set `golden_dir`, preserve the model and case subdirectories and the golden filename referenced by the selected recipe.

After editing the configuration, regenerate the plan and run these commands on the GPU host from the FastCI checkout.

```sh
fastci-megatron check --config .fastci/megatron.yaml --out .fastci/megatron-runtime.json
fastci-megatron plan --repo "$MEGATRON_REPO" --config .fastci/megatron.yaml \
  --source "$BASE" --target "$BASE" --mode baseline --out .fastci/bootstrap-plan.json
fastci-megatron run --repo "$MEGATRON_REPO" --plan .fastci/bootstrap-plan.json \
  --out .fastci/megatron-baseline
fastci-megatron ingest --run .fastci/megatron-baseline/run.json --out .fastci/megatron-graph.json
```

Use a new output directory per run. Ingestion requires passing full workloads and hashed sample traces for all eight ranks in every configured repeat and training phase. The graph becomes eligible for selection only after every baseline case has fresh evidence.

To try a change, set `SOURCE` to its Git ref and use its target branch or commit as `--target`. Pass the graph at the merge-base tree. `shadow` runs the configured baseline and records the proposed tests. `evaluate` reports failures among cases the planner would have omitted. `enforce` executes the retained tests in risk order when a compatible graph is available. Missing or incompatible graph snapshots and unsupported changes fall back to the configured baseline. Tests with new or changed validation recipes are protected from omission. To carry risk history forward, use `fastci-megatron observe --run RUN_JSON --out HISTORY_JSON`, then pass `--history HISTORY_JSON` to both later plans and history updates.

```sh
fastci-megatron plan --repo "$MEGATRON_REPO" --config .fastci/megatron.yaml \
  --source "$SOURCE" --target "$BASE" --mode shadow --graph .fastci/megatron-graph.json \
  --out .fastci/shadow-plan.json
fastci-megatron run --repo "$MEGATRON_REPO" --plan .fastci/shadow-plan.json \
  --out .fastci/megatron-shadow
fastci-megatron evaluate --run .fastci/megatron-shadow/run.json --out .fastci/shadow-summary.json
```

`workload_reduction: true` opts eligible deterministic regular cases into paired prefix validation during compatible enforce runs. FastCI preserves the original training schedule and uses an earlier exit interval in the prefix copy, then checks the matching golden prefix. The full run remains authoritative and supplies graph evidence. Paired runs consume extra GPU time.
