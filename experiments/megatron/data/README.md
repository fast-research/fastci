# Megatron-LM benchmark data

The mainline window contains 100 consecutive first-parent commits ending at [a7a6d22](https://github.com/NVIDIA/Megatron-LM/commit/a7a6d2225700a8ee701453288894501d868c6bd4). The 47 functional cases were traced and line-profiled on one eight-GPU node at [d2b0e35](https://github.com/NVIDIA/Megatron-LM/commit/d2b0e35cc4f05bc6b70734d558617050fbbb51ed). The unsampled timing run used the same case recipes and runtime at [bfa3f78](https://github.com/NVIDIA/Megatron-LM/commit/bfa3f78058601a8f8f62582f88765e624e359a2a).

- `mainline.json.gz` contains commit and parent SHAs, a digest of the upstream changed paths, saved selections, and affected case IDs. Per-commit `line_hits` maps each case to executable line numbers grouped by zero-based positions in the upstream first-parent changed-path list. Taking the union across selected tests recovers the modified-code coverage numerator. A case counts as affected if its coverage included the first executable statement of a changed function at the reference revision.
- `call-graph.json.gz` contains the frozen sampled call graph, with upstream Python symbols as nodes and job IDs annotating observed calls. Its `execution_times` come from sampled runs. The replay uses the unsampled times for cost comparisons.
- `mainline.json.gz` also contains per-case GPU-seconds from matched sampled and unsampled runs, the sampling rate, the static L0 job list, and the GPU-seconds for the initial and retry graph-build attempts.
- `prefix.json` contains the two saved golden and observed numerical prefixes. The case configs and pytest implementation come from upstream Git at the declared revision.
- `expected-table.tsv` is the historical reference output. The replay regenerates it from the data above.
- `SHA256SUMS` records file hashes for integrity verification. Run `shasum -a 256 -c SHA256SUMS` from this directory.

Benchmark case IDs are stable references to upstream case names. The native benchmark config resolves them from the [pinned recipe definitions](https://github.com/NVIDIA/Megatron-LM/tree/d2b0e35cc4f05bc6b70734d558617050fbbb51ed/tests/test_utils/recipes). The replay reads changed paths directly from the referenced upstream commits.

The bundle projects the measurements to the fields needed for replay. Raw traces, raw logs, training assets, environment inventories, and planner prompts stay outside this dataset.
