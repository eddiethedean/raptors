# Build and release Raptors 0.1

The 0.1 package is a small, NumPy-independent preview backed by the separate
safe `raptors-storage` crate. It supports GIL-enabled CPython 3.12–3.14 on
Linux x86-64, macOS x86-64/arm64, and Windows x86-64. See the machine-readable
[contract](../compat/raptors-0.1.json) for the exact boundary.

## Local development

From the repository root, prepare the locked oracle and test environment:

```bash
uv sync --project raptors-python --extra dev --locked --python 3.14 --no-install-project
uv run --project raptors-python --extra dev --no-sync maturin develop --manifest-path raptors-python/Cargo.toml --release
uv run --project raptors-python --extra dev --no-sync pytest raptors-python/tests/preview -q
```

`uv.lock` pins NumPy 2.5.3 and its wheel hashes for testing. NumPy is not a
runtime dependency. The root pytest configuration imports `raptors` directly;
a missing or unloadable extension fails the test run.

## Build and check a wheel

```bash
uv run --project raptors-python --extra dev --no-sync maturin build --release --locked --manifest-path raptors-python/Cargo.toml --interpreter python3.14 --out /tmp/raptors-wheels
uv run --project raptors-python --extra dev --no-sync python scripts/check_wheel_contract.py /tmp/raptors-wheels/raptors-0.1.0-cp314-*.whl --python 3.14
uv run --project raptors-python --extra dev --no-sync python -m twine check /tmp/raptors-wheels/raptors-0.1.0-cp314-*.whl
uv run --project raptors-python --extra dev --no-sync python scripts/check_clean_install.py /tmp/raptors-wheels/raptors-0.1.0-cp314-*.whl
```

The contract check verifies the Python tag, `Requires-Python`, absence of
runtime dependencies, native extension, and packaged MIT license. The clean
install check creates an isolated environment with no NumPy and exercises
construction, views, and mutation. Raptors 0.1 publishes wheels only; the
source distribution remains disabled until the sibling storage crate can be
built from a clean source archive.

## Release

Run the `Release to PyPI` workflow manually on the candidate branch to execute
the full build-only gate before tagging. Manual runs cannot publish. A valid
`vX.Y.Z` tag runs the same checks and then uploads the already-tested wheels
through PyPI Trusted Publishing. The publisher must name repository
`eddiethedean/raptors`, workflow `release.yml`, and environment `pypi`.

The published compatibility subset and support policy are versioned in
`compat/raptors-0.1.json`; release readiness also requires the safety and
cross-platform evidence listed in [RELEASE_0_1.md](../docs/RELEASE_0_1.md).
