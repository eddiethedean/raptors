# Python binding tests

These tests belong to the legacy prototype. They are input to the rebuild audit, not proof of complete NumPy compatibility.

Use the parent [testing guide](../TESTING.md) for commands and the [test porting guide](../../docs/TEST_PORTING.md) for reference provenance and differential testing requirements.

## Directory map

| Files | Scope |
| --- | --- |
| `test_array.py` | Array properties, construction, operators, and methods |
| `test_dtype.py` | Dtype bindings |
| `test_ufunc.py` | Functions and reductions |
| `test_numpy_interop.py` | NumPy conversion |
| `numpy_port/*.py` | Additional compatibility-oriented cases |
| `*_test.rs` | Rust integration tests for bindings |

The `numpy_port` label does not establish a faithful upstream port. Audit the source revision, original assertion, and adaptation for each reused case.

## Run against a fresh build

From the repository root, after activating the environment described in [development setup](../DEVELOPMENT.md):

```bash
maturin develop --manifest-path raptors-python/Cargo.toml
python -c "import raptors; print(raptors.__file__)" &&
python -m pytest raptors-python/tests/ -v
```

An import failure must stop validation. Current collection code can skip when the package is missing; fixing that is a v0.1 deliverable.

For the Rust integration files, use `cargo test -p raptors-python --tests`; library-only tests do not include them. Platform-specific Python linking may be required, and failures must be reported.

## Adding or revising cases

Match the pinned NumPy behavior and test observable results, dtypes, errors, warnings, and alias effects. Include adversarial input and lifetime sequences. Keep unsupported cases visible with a specific milestone and removal condition.

Do not change expected values to fit the implementation or count generated placeholders as coverage. The [verification record](../../docs/NUMPY_TEST_VERIFICATION.md) is the place for actual run evidence.
