# Raptors

Raptors is building a Rust-backed Python package for NumPy's public functionality. Applications can use `import raptors as np`; replacing NumPy's C ABI underneath precompiled extensions is outside the goal. Full compatibility is the destination, with each 0.x release advertising only its verified subset.

## Current status

Version 0.1 is implemented locally as a narrow native preview. It supports explicit `bool`, `int64`, `uint64`, `float32`, and `float64` arrays; metadata; basic integer and slice views; scalar and exact-shape, same-dtype assignment; and independent copies. It does not implement arithmetic or general NumPy compatibility and makes no speed claim.

The 0.1 release gate remains pending. Local CPython 3.12–3.14 differential/property suites, Rust tests, Miri, CPython 3.14 wheel metadata checks, a clean install without NumPy, and an informational benchmark have run on macOS ARM64. The hosted Linux/macOS/Windows wheel matrix and Linux AddressSanitizer gate have not run on the candidate changes. See the [verification record](docs/NUMPY_TEST_VERIFICATION.md) and [0.1 execution plan](docs/RELEASE_0_1.md).

## Repository map

| Location | Purpose |
| --- | --- |
| [raptors-storage](raptors-storage/) | New checked, initialized storage and signed-stride views for the preview; no unsafe Rust |
| [raptors-python](raptors-python/) | PyO3 preview, package metadata, and differential/property tests |
| [raptors-core](raptors-core/) | Legacy engine retained for audit and future reference; not used by the preview |
| [compat](compat/) | NumPy 2.5.3 API inventory and executable 0.1 subset contract |
| [numpy-reference](numpy-reference/) | Pinned NumPy 2.5.3 source checkout used for reference and provenance |
| [docs](docs/README.md) | Rebuild plan, release roadmap, implementation evidence, and development guides |

NumPy is an optional development/test dependency and is not required at runtime. The preview extension owns its Rust buffers and can be installed and imported without NumPy.

## Build and test the preview

From the repository root with CPython 3.12, 3.13, or 3.14:

```bash
uv sync --project raptors-python --extra dev --locked --python 3.14 --no-install-project
uv run --project raptors-python --extra dev --no-sync maturin develop --manifest-path raptors-python/Cargo.toml --release
uv run --project raptors-python --extra dev --no-sync python -m pytest raptors-python/tests/preview -q
cargo test --locked -p raptors-storage
```

See [Python build](raptors-python/BUILD.md), [Python testing](raptors-python/TESTING.md), and the [0.x roadmap](docs/CONVERSION_ROADMAP.md) for the supported boundary and required checks.

## Release workflow

`.github/workflows/release.yml` validates the tag and package versions, runs Rust safety checks, builds and tests wheels across the declared platform/Python matrix, then publishes those wheels to PyPI through the configured trusted publisher. It triggers on exact `vX.Y.Z` tags. A manual run performs the validation/build path without publishing.

Do not tag 0.1 until the pending gates in the [release plan](docs/RELEASE_0_1.md) have passed. Source distributions are not published until a clean source build is verified.

## Documentation

Start with the [documentation index](docs/README.md), [rebuild plan](docs/REBUILD_PLAN.md), and [release 0.1 evidence](docs/RELEASE_0_1.md). The project is MIT licensed; see [LICENSE](LICENSE).
