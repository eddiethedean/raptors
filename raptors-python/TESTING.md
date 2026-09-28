# Testing Raptors Python

The current tests exercise the legacy prototype. The [rebuild plan](../docs/REBUILD_PLAN.md) requires a new NumPy differential harness and explicit compatibility evidence; those systems are planned.

## Build and verify the artifact

Follow [DEVELOPMENT.md](DEVELOPMENT.md), then run from the repository root:

```bash
maturin develop --manifest-path raptors-python/Cargo.toml
python -c "import raptors; print(raptors.__file__); print(raptors.__version__)" &&
python -m pytest raptors-python/tests/ -v
```

The explicit import is essential: existing `conftest.py` files can skip collection if the extension cannot load. Correcting that skip is a v0.1 gate. Confirm the path points to the newly built artifact.

To run a selected file:

```bash
python -c "import raptors" &&
python -m pytest raptors-python/tests/test_array.py -v
```

Capture failures and crashes as baseline evidence. These commands are not claimed to pass today.

## Rust checks

From the repository root:

```bash
cargo test -p raptors-core --tests
cargo test -p raptors-core --doc
cargo check -p raptors-python
cargo test -p raptors-python --tests
```

The last command exercises binding integration targets and may need Python embedding/linker setup for the platform. A linker failure is a failed check, not a successful or silently skipped suite.

`cargo test --lib` does not run integration files in `tests/`. The initial core library-only run collected zero tests. `cargo check` compiles without demonstrating Python runtime behavior.

## Existing layout

| Location | Purpose |
| --- | --- |
| [test_array.py](tests/test_array.py) | Array construction, properties, operators, and methods |
| [test_dtype.py](tests/test_dtype.py) | Dtype bindings |
| [test_ufunc.py](tests/test_ufunc.py) | Registered functions and reductions |
| [test_numpy_interop.py](tests/test_numpy_interop.py) | Conversion behavior |
| [numpy_port](tests/numpy_port/) | Additional compatibility-oriented Python tests |
| [Rust integration tests](tests/) | Binding tests written in Rust |
| [Core integration tests](../raptors-core/tests/) | Rust engine behavior |

Filenames and old test totals do not prove upstream provenance or complete coverage.

## Planned differential gate

Execute the same cases against the pinned NumPy version and Raptors. Compare values, dtype, scalar/array return, shape, applicable strides/flags, alias mutation, exceptions, and warnings.

Prioritize narrow-dtype construction, large integer precision, negative strides, overlapping assignment, zero-sized dimensions, owner deletion, scalar promotion, and `out=` behavior. Add generated operation sequences and retain minimized failures.

Use exact structural/integer comparisons and operation-specific numerical criteria. Keep native and delegated execution separate. Required native cases must pass with fallback disabled.

## Reporting and skips

Record revision, environment, artifact path, commands, exit codes, collected/passed/failed/skipped cases, and crashes. Every skip or expected failure needs a specific reason and removal condition. Required supported cases cannot stay skipped at their release gate.

Legacy `run_tests.sh` and Make targets are convenience wrappers, not the acceptance authority. The runner also pipes a Rust command through `tee` without enabling pipeline failure propagation; v0.1 must correct that before relying on its success message.

## Safety and CI

Miri for isolated Rust storage/layout code, property tests, fuzzing, and supported sanitizer builds complement Python behavioral tests. They do not replace review of unsafe preconditions, aliases, and foreign lifetimes.

The current workflow is [ci.yml](../.github/workflows/ci.yml). Release 0.1 will audit its build dependencies, coverage, and failure handling. The release gates are not yet enforced by it.

See [test porting](../docs/TEST_PORTING.md) and the dated [verification record](../docs/NUMPY_TEST_VERIFICATION.md).
