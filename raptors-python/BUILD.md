# Build and release Raptors 0.3

The 0.3.0 release candidate adds numeric ufuncs to the published array and dtype
foundation. Its exact tested boundary is recorded in the machine-readable
[0.3 contract](../compat/raptors-0.3.json). The published 0.2 and 0.1 packages
remain available under their [0.2](../compat/raptors-0.2.json) and
[0.1](../compat/raptors-0.1.json) contracts.

The published `0.1.0` release contains twelve version-specific wheels for its
original four platform targets. The published `0.2.0` release contains eight
stable-ABI wheels, one for each of these targets:

- manylinux x86-64 and ARM64
- macOS x86-64 and ARM64
- musllinux x86-64 and ARM64
- Windows x86-64 and ARM64

Each new wheel is tagged `cp312-abi3` and is tested on GIL-enabled CPython
3.12, 3.13, and 3.14. It does not support free-threaded CPython.

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
uv run --project raptors-python --extra dev --no-sync maturin build --release --locked --manifest-path raptors-python/Cargo.toml --interpreter python3.12 --out /tmp/raptors-wheels
wheel=$(find /tmp/raptors-wheels -maxdepth 1 -name 'raptors-*.whl' -print -quit)
uv run --project raptors-python --extra dev --no-sync python scripts/check_wheel_contract.py "$wheel" --python 3.12 --platform-family macosx --platform-fragment arm64
uv run --project raptors-python --extra dev --no-sync python -m twine check "$wheel"
uv run --project raptors-python --extra dev --no-sync python scripts/check_clean_install.py "$wheel" --python python3.12
```

The contract check verifies the `cp312-abi3` tag, platform family and architecture,
`Requires-Python`, absence of runtime dependencies, native extension, and
packaged MIT license. The clean install check creates an isolated environment
with no NumPy and exercises construction, views, and mutation. The release
workflow publishes wheels only. Source distributions remain excluded until a
clean source archive can build the workspace and sibling storage crate.

## Release

Run the `Release to PyPI` workflow manually on the release candidate commit to
execute the full build-only gate before tagging. It builds and tests all eight target
wheels on CPython 3.12, 3.13, and 3.14. Manual runs cannot publish. A valid
`vX.Y.Z` tag runs the same checks and then uploads the eight already-tested
wheels through PyPI Trusted Publishing. The publisher must name repository
`eddiethedean/raptors`, workflow `release.yml`, and environment `pypi`.

The compatibility subset and support policy are versioned in
`compat/raptors-0.3.json`; release readiness also requires the safety and
cross-platform evidence listed in [RELEASE_0_3.md](../docs/RELEASE_0_3.md).
