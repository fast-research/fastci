"""Chain application startup before enabling optional FastCI sampling."""

try:
    from fastci.sampler import install_hook

    install_hook(__file__)
except Exception as _error:
    import sys

    print(f"[FastCI] sampling setup failed: {_error}", file=sys.stderr)
