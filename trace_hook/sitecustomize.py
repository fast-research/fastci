"""Opt-in PYTHONPATH hook. Use only in CI's dedicated test Python environment."""

try:
    from fastci.sampler import install_hook

    install_hook(__file__)
except Exception as exc:
    import sys

    print(f"[FastCI] sampling setup failed: {exc}", file=sys.stderr)
