# Raptors

Raptors is building a Rust-backed Python package for NumPy's public functionality. Applications can use `import raptors as np`; replacing NumPy's C ABI underneath precompiled extensions is outside the goal. Full compatibility is the destination, with each 0.x release advertising only its verified subset.

## Current status

Version 0.1.0 is published as a narrow native preview. It supports explicit `bool`, `int64`, `uint64`, `float32`, and `float64` arrays; metadata; basic integer and slice views; scalar and exact-shape, same-dtype assignment; and independent copies. It does not implement arithmetic or general NumPy compatibility and makes no speed claim.

The tagged `v0.1.0` workflow passed Rust, Miri, AddressSanitizer, and twelve version-specific wheel builds and published them to PyPI. This does not establish general NumPy compatibility, a general memory-safety guarantee, or a performance advantage. See the [verification record](docs/NUMPY_TEST_VERIFICATION.md) and [0.1 release record](docs/RELEASE_0_1.md).

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

`.github/workflows/release.yml` validates the tag and package versions, runs Rust safety checks, builds and tests eight target wheels, then publishes those wheels to PyPI through the configured trusted publisher. Each `cp312-abi3` wheel is tested on CPython 3.12, 3.13, and 3.14. It triggers on exact `vX.Y.Z` tags. A manual run performs the validation/build path without publishing.

The published 0.1.0 wheel set predates this eight-target strategy. Run the workflow manually and confirm it passes before creating a later release tag. Source distributions remain disabled until a clean source build is verified.

## Documentation

Start with the [documentation index](docs/README.md), [rebuild plan](docs/REBUILD_PLAN.md), and [release 0.1 evidence](docs/RELEASE_0_1.md). The project is MIT licensed; see [LICENSE](LICENSE).
