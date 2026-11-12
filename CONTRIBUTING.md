# Contributing

## Development setup

Use Python 3.10+ and Git in a POSIX environment:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python examples/demo.py --out .fastci/dev-demo
```

Use a fresh demo output directory on subsequent runs. Core tests use only the standard library and self-contained temporary repositories; they do not require a GPU or sampler.

The test suite also executes the [complete mock CI](docs/mock-ci.md), including real subprocesses, observed calls, multiple revisions, malformed traces, and failed/repaired workloads. Run it separately with `python -u examples/mock_ci.py --out .fastci/complete-ci` to retain the inspectable report and artifacts.

## Changes

- Explain the behavior being changed and its effect on selection, ordering, or execution.
- Add focused tests for meaningful behavior changes, especially fallback, evidence replacement, history chronology, and workload validation.
- Update [Design](docs/design.md) when algorithm definitions or method assumptions change, and [Data formats](docs/formats.md) when input/output contracts change.
- Use synthetic fixtures with generic test names. Keep local state, credentials, source snapshots, job catalogs, and generated traces out of contributions.
- Keep the core standard-library-only; sampling is an optional dependency.

The GitHub Actions workflow runs the unit suite and CPU demonstration on supported Python versions. Test results from the CPU fixtures do not establish GPU performance or regression-detection coverage.

## Reports

For a bug report, provide the Python version, command, expected behavior, and a minimal synthetic input or reproducer. Include algorithm parameters when reporting differences in selected tests or risk order.

## Release archives

Build distributions, then normalize source archive metadata before sharing:

```sh
python -m pip install build
python -m build
python scripts/sanitize_sdist.py dist/*.tar.gz
python scripts/sanitize_sdist.py --check dist/*.tar.gz
```

The sanitizer removes account names, numeric ownership IDs, extended timestamps, and gzip filename metadata. It preserves file contents. Inspect the contents and package metadata as well as archive headers when reviewing a release.

Contributions are distributed under the project's [MIT License](LICENSE).
