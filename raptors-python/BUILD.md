# Building and releasing Raptors Python

**Status: build guidance and tagged publishing automation.** Building and testing a wheel does not establish full NumPy compatibility or satisfy every roadmap release gate. Follow the [rebuild plan](../docs/REBUILD_PLAN.md).

## Development build

Create the isolated environment in [DEVELOPMENT.md](DEVELOPMENT.md). From the repository root:

```bash
maturin develop --manifest-path raptors-python/Cargo.toml
python -c "import raptors; print(raptors.__file__)"
```

Repeat the build after Rust edits. Use an optimized build for measurements:

```bash
maturin develop --release --manifest-path raptors-python/Cargo.toml
```

The current package links the local core crate and depends on NumPy at runtime. The rebuild's native execution and optional-adapter dependency policy is not yet implemented.

## Build a wheel

From the repository root with the development environment active:

```bash
maturin build --release --manifest-path raptors-python/Cargo.toml --out /tmp/raptors-wheels
```

Install the exact newly produced wheel into a second clean environment. Avoid a wildcard that may select stale builds. Verify the imported path and version, then run the required tests from the checkout using that environment's interpreter.

For source distributions, verify that the sibling core crate and required manifests/source are included and that a clean source build succeeds. The existing packaging declarations require validation.

## Version and platform policy

Release 0.1 selects exact Python, NumPy, Rust, and tool versions. Do not infer support from historical Python 3.7+ classifiers or the existing CI matrix.

The tagged workflow currently builds the historical Python 3.7–3.12 grid. Confirm or change that matrix when release 0.1 declares its supported Python versions; passing these jobs alone does not establish the support policy.

Version declarations currently exist in the root `Cargo.toml`, both crate manifests, and `pyproject.toml`; the crate declarations are not automatically inherited from the workspace. Reconcile them during release preparation.

The intended release matrix includes Linux, macOS, and Windows with explicitly declared architectures. Validate wheel tags, imports, runtime dependencies, numerical backends, and installed-package behavior on each supported combination.

## Existing automation

Push an exact `vX.Y.Z` tag to trigger the [PyPI release workflow](../.github/workflows/release.yml). It checks that the tag matches every package version declaration, runs Rust formatting/tests/Clippy, builds and tests wheels across the configured operating-system and Python matrix, checks wheel metadata, and publishes only after those jobs pass.

Configure the PyPI Trusted Publisher for repository `eddiethedean/raptors`, workflow file `release.yml`, and GitHub environment `pypi`. The [manual TestPyPI workflow](../.github/workflows/publish-python.yml) is separate and does not publish to PyPI.

These automated checks do not establish full NumPy compatibility, memory-safety evidence, or a performance advantage. Do not tag a public release until the applicable roadmap gate also passes. Remaining release evidence includes:

1. Pinned reference and compatibility manifest for the release.
2. Required semantic tests, with build/import failures fatal.
3. Safety review and applicable Miri, sanitizer, and fuzz evidence.
4. Reproducible benchmarks for every public performance claim.
5. Clean wheel installs and application tests on the declared platform matrix.
6. Consistent package metadata, actual license files, upstream notices, and documentation.

Publishing should consume the validated artifacts. The documentation update itself does not authorize or trigger a release.

## Troubleshooting

Check the selected `PYO3_PYTHON`, interpreter version, extension path, and captured build output first. The checked-in Cargo configuration is machine-specific. Native Rust tests for the bindings can also need Python embedding/linker configuration.

The core build script currently writes its header into the source tree; directing Cargo output elsewhere does not prevent that. Generated-output cleanup is a v0.1 task.

See [testing](TESTING.md) and [verification](../docs/NUMPY_TEST_VERIFICATION.md) before interpreting a successful build as functional evidence.
