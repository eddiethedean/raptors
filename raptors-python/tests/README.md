# Python binding tests

The required 0.2 release suite lives in [`preview/`](preview/) and is configured as the default pytest path. It runs against the NumPy 2.5.3 oracle and covers the declared numeric boundary, generated operation sequences, and seeded harness faults. Missing package imports fail the run.

Run the supported suite from the repository root after following the [development setup](../DEVELOPMENT.md):

```bash
uv run --project raptors-python --extra dev --no-sync python -m pytest raptors-python/tests/preview -q
```

The top-level legacy Python and Rust tests are retained for audit. Their directory names and historical `numpy_port` labels do not establish faithful upstream ports or current NumPy conformance. The nested `numpy_port` conftest still has a legacy skip path, but that suite is not part of the 0.2 release gate.

| Files | Scope |
| --- | --- |
| `preview/test_differential.py` | Pinned-reference examples for creation, conversion, layout, indexing, assignment, and copy |
| `preview/test_properties.py` | Deterministic Hypothesis cases for slices, nested views, overlap, construction, and owner deletion |
| `preview/test_harness.py` | Seeded wrong-backend checks for dtype, byte order, flags, shape, axis order, aliasing, and owner lifetime |
| `test_array.py`, `test_dtype.py`, `test_ufunc.py`, `test_numpy_interop.py` | Legacy prototype cases; not 0.2 coverage |
| `numpy_port/*.py`, `*_test.rs` | Historical compatibility-oriented tests; audit source and assertion provenance before reuse |

The [0.2 release record](../../docs/RELEASE_0_2.md) documents current preview results; the [verification record](../../docs/NUMPY_TEST_VERIFICATION.md) preserves 0.1 and legacy observations. Do not change expected values to fit Raptors or count generated placeholders as coverage.
