# FastCI

FastCI is a reference implementation for GPU-intensive CI. It selects tests from sampled call graphs, prunes equivalent execution contexts, prioritizes likely failures, and validates workload reductions.

## Quick start

From a checkout with Python 3.10+, Git, and a POSIX environment, install FastCI and run the CPU demo.

```sh
python -m pip install -e .
python examples/demo.py --out .fastci/demo
```

For the full CPU mock CI, run `python examples/mock_ci.py --out .fastci/complete-ci`. Choose a fresh output directory for each run. [Mock CI walkthrough](docs/mock-ci.md)

## Megatron-LM

Recompute the benchmark table for 100 public commits and 47 functional tests on CPU.

```sh
python experiments/megatron/reproduce.py --out .fastci/megatron-replay
```

The Megatron guide also covers recomputing selection from upstream Git and running native functional CI. [Megatron reproduction guide](experiments/megatron/README.md)

## Documentation

- [CLI guide](docs/usage.md) for planning, execution, tracing, and scheduling replay
- [Design](docs/design.md), [implementation audit](docs/paper-audit.md), and [data formats](docs/formats.md)
- [Contributing](CONTRIBUTING.md) and [MIT License](LICENSE)
