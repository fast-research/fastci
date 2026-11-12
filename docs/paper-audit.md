# Core implementation audit

The implementation was rechecked against FastCI's evidence-selection, context-pruning, risk-ordering, workload-optimization, runtime-graph-maintenance, and deployment-safeguard descriptions. The complete mock workflow alone did not establish these properties: it used unique function definitions, curated metadata, simple argument lists, manually instrumented traces, and exclusively fast-forward integration.

This audit added counterexamples and independent runtime tests rather than changing expected mock outcomes to hide inconsistencies.

## Findings and corrections

| Area | Gap exposed outside the original mock | Correction and regression evidence |
| --- | --- | --- |
| Validation metadata | Equal dimension labels could collapse sequence 128 and 8192; unknown hardware could also be grouped | Compare relevant workload values and keep unknown scope/resource cases distinct; `test_paper_audit.py` reproduces the previous erroneous pruning |
| Source identity | Getter/setter definitions shared a name, so the last definition discarded the getter's source span | Preserve every definition span while aggregating the shared entity identity and caller evidence; both property accessors resolve to the class-qualified entity |
| Static expansion | Direct calls could be recovered while assignment aliases and callbacks referring to the same function were omitted | Track conservative function references and fixed-point assignment aliases, including alias chains; independent callback/alias regression exercises all caller paths |
| Workload override | Replacing only the first repeated flag could leave a later effective value unchanged | Replace every separate-value and equals-style occurrence; the repeated/mixed-flag regression fails under the previous implementation |
| Runtime sampling | The instrumented mock worker supplied traces, but ordinary execution never installed the sampler | Inject the wheel-packaged rank-0 hook for local CLI execution, preserve application startup, expose rate/following controls, and record missing evidence without changing test outcomes |
| Evidence provenance | Ingestion accepted traces from unrelated revisions/tests or a changed snapshot; a manifest could omit the original half of a paired capture | Bind revision, run, plan digest, frozen snapshot, completed outcomes, complete per-test captures, and file hashes; reject or preserve invalid fragments with diagnostics |
| Graph publication | Any caller could mark a graph stable before integration; fast-forward-only mocks concealed merge/squash identity changes | Require a passing verified run and equality of tested/integrated trees; publish an exact graph under the integration commit; reject changed trees and stale promotion attempts |
| Historical cost | Trace duration approximations and fixed catalog estimates could dominate newly observed execution costs | Store authoritative attempt duration from verified execution as `timing_source=execution`; use it for future representatives; account for both paired attempts and full reservation intervals separately |
| Missing/corrupt evidence | Corrupt snapshots and malformed profile objects could abort the workflow rather than preserve useful evidence | Fall back from corrupt/wrong-version graph snapshots with diagnostics; validate profile structure; preserve failed fragments and store a verified version snapshot even with no new usable trace |
| Rollback | Drain/disable were only planner labels, leaving execution, sampling, and graph/state writes to caller convention | Implement one configuration `mode` field across CLI boundaries: drain stops new optimized plans, disable reroutes pending plans and bypasses optimizer state/sampling/updates |
| Same-version comparisons | A test could mutate tracked source between original and reduced execution | Verify source revision and cleanliness around each attempt; a changing baseline cannot approve a reduction |

The mock catalog was also corrected to supply values for its declared validation dimensions. The mock now uses runner-owned trace directories, captured-file hashes, verified ingestion, observed cost updates, and integration-before-promotion. Its numerical test assertions remain real CPU executions; its resources remain explicitly mock slots.

## Checks that already matched the method

- Exact test-specific directed context subgraphs, K=3 by default, independently grouped per modified entity and then unioned.
- Direct evidence with missing-evidence class/caller expansion; conservative retention when the original context is unknown.
- The risk formula, positive-file-rate confidence, W=5, gamma=0.8, lambda=5, pass attenuation at two/four passes, stable ties, and score-before-observe chronology.
- Explicitly unrelated dimensions only, path-based protection, periodic original/reduced comparison, authoritative original results on disagreement, and persistent revocation.

The exact directed-neighborhood interpretation, final no-evidence fallback, deterministic file-rate ties, validation interval, and resource packing remain documented implementation choices where the method is not fully prescriptive.

## Independent verification

```sh
python -m unittest discover -s tests -p 'test_paper_audit.py' -v
python -m unittest discover -s tests -p 'test_runtime_evidence.py' -v
python -m unittest discover -s tests -v
python -u examples/mock_ci.py --out .fastci/audited-ci
```

The runtime-evidence suite uses real temporary Git histories, separate subprocesses, actual startup hooks, source modifications, and file hashes. It covers fast-forward, non-fast-forward, and squash integration; rejects a later untested tree; checks default hook packaging and application startup; and exercises configuration rollback without reading corrupt optimizer state.

The same suite is also run against an installed wheel with ordinary bytecode generation enabled. Git fixtures ignore runtime caches rather than depending on a no-bytecode environment; source-tree equality checks remain strict.

The sampler transport test uses an explicitly synthetic executable implementing the py-spy command-line protocol. It verifies PID/rate/following transport, rank selection, hook behavior, and missing-tool handling. It is not an actual statistical-sampling experiment. Real py-spy attach permissions and overhead, GPU execution, and distributed hardware behavior remain target-environment validation requirements.

## Evidence boundaries

These checks establish the exercised implementation contracts, not a proof of complete test selection or preserved fault detection. Sampling can miss code; Python reflection limits static callers; metadata must represent test semantics; and periodic workload comparisons can miss a regression between checks. Runtime records and configuration are trusted CI-controller inputs, not cryptographic remote-execution attestations. State updates require a single writer and fixed, caller-controlled test inputs/environment. No paper performance number is inferred from this audit or the CPU mock.
