# Release 0.2: numeric array and dtype foundation

**Release status:** `v0.2.0` is published to [PyPI](https://pypi.org/project/raptors/0.2.0/). The annotated tag points to commit `3b143968c7c44b4ba77e26e4fbd6fbdfc7976b9d`. The [tagged release workflow](https://github.com/eddiethedean/raptors/actions/runs/36511142066) passed its validation, safety, API inventory, and eight-target wheel gates, then published eight wheels. PyPI lists all eight `cp312-abi3` wheels and no source distribution.

NumPy 2.5.3 is the pinned behavior reference at source commit `dd88c0c19b54ad9ed3533224221285bf0873249a`. The 0.2 implementation is the fixed-width numeric slice of the [full dtype plan](DTYPE_ARCHITECTURE.md); that plan maps all eleven classic dtype groups and NumPy 2.x `StringDType` to their later release phases.

## Release boundary

| Area | 0.2 behavior | Limits |
| --- | --- | --- |
| Dtypes | `bool`; signed and unsigned 8/16/32/64-bit integers; `float16`, `float32`, `float64`; `complex64`, `complex128`; platform aliases and the `longdouble`/`clongdouble` forms exposed by NumPy on that target. | Canonical names, scalar wrappers, size, alignment, byte order, aliases, and promotion follow the pinned NumPy build. Apple ARM64 and Windows expose the extended names with the same precision as `float64`/`complex128`. |
| Construction | `array(data, dtype=None, copy=True, order='K')`, `zeros(shape, dtype=None, order='C')`, and `empty(shape, dtype=None, order='C')`; numeric inference; rectangular nested lists/tuples; zero-dimensional and empty arrays; C/F allocation. | Inputs are Python bool, int, float, complex, Raptors arrays, supported Raptors scalar wrappers, and rectangular list/tuple nesting. Object, string, structured, datetime, foreign-buffer, and arbitrary conversion-protocol inputs are unsupported. `copy=False` can reuse only a compatible Raptors array. |
| Conversion | Numeric casts and assignment casts, warning behavior for tested overflow/complex-to-real cases, and `promote_types` for every pair in the declared numeric dtype set. | No scalar arithmetic or ufunc promotion is implemented. Python `Decimal` and arbitrary user-defined scalar protocols are unsupported. |
| Layout | Checked shape and signed byte strides; dtype byte order and alignment; array flags; C/F order; transpose/axis permutations; reshape view/copy behavior; independent copies. | Read-only and foreign-backed arrays are outside the contract. No typed Rust reference is formed from stored bytes. |
| Indexing and mutation | Integer, slice, tuple, ellipsis, new axis, boolean scalar/mask, integer-array/fancy and mixed indexing; scalar, sequence, broadcast, mask and fancy assignment; duplicate-index writes and overlap snapshots. | Result values, dtype metadata, shape, strides, flags, warning/error classes, aliasing, and copy/view cases covered by the contract tests are compared with NumPy 2.5.3. |

The machine-readable API surface and unsupported boundary are in [`compat/raptors-0.2.json`](../compat/raptors-0.2.json). The implementation does not claim every method on NumPy's scalar or `ndarray` classes.

## Implementation and verification

The storage crate uses a checked byte-addressed owner, shape, signed byte strides, offset, and shared views. Reads copy bytes into local arrays before decoding, with explicit native and non-native byte order handling. Extended formats use x87 80-bit values on x86-64, IEEE binary128 on Linux ARM64, and 64-bit floating storage where NumPy does. Scalar wrappers are Raptors types rather than NumPy scalar classes.

Local verification on macOS 26.5.2 arm64, Rust 1.96.0, NumPy 2.5.3, and CPython 3.12.13, 3.13.11, and 3.14.3:

- The built `cp312-abi3` macOS arm64 wheel passed `tests/preview`: **774 passed, 0 skipped** on each supported CPython version.
- The differential suite covers every declared numeric dtype pair for casts and promotion, supported dtype aliases, scalar wrappers, reshape copy/view cases, masks, fancy indices, assignment, byte order, and array flags. Deterministic generated operation sequences combine construction, views, indexing, writes, casts, copies, and owner deletion.
- `rustfmt --edition 2021 --check raptors-storage/src/lib.rs raptors-python/src/preview.rs raptors-python/src/preview_lib.rs raptors-python/build.rs`, `cargo check --locked -p raptors-python`, and `cargo clippy --locked -p raptors-storage -p raptors-python --all-targets -- -D warnings` passed.
- `cargo test --locked -p raptors-storage` and `cargo +nightly miri test --locked -p raptors-storage` each passed all 9 tests.
- The local wheel passed `scripts/check_wheel_contract.py`, `twine check`, and `scripts/check_clean_install.py`; the clean install imported and exercised the wheel without NumPy.
- An informational baseline for eight foundation operations is recorded in [`raptors-0.2.0-baseline.json`](benchmarks/raptors-0.2.0-baseline.json). It reports timing, traced Python allocation peak bytes, and coarse process RSS deltas. It makes no performance claim.

The hosted [main CI run](https://github.com/eddiethedean/raptors/actions/runs/36509962693) and [tagged release workflow](https://github.com/eddiethedean/raptors/actions/runs/36511142066) passed on the release commit. The tagged workflow passed hosted Miri and AddressSanitizer, API inventory validation, Rust checks, and the eight-target `cp312-abi3` wheel build/test matrix on CPython 3.12–3.14. Its PyPI publish job succeeded. The published files are recorded in the [0.2 compatibility contract](../compat/raptors-0.2.json).

## Explicitly deferred work

The other built-in dtype groups remain on the roadmap: datetime/timedelta (`M`/`m`), object (`O`), byte strings (`S`), Unicode strings (`U`), fixed-width void/structured data (`V`), and NumPy 2.x `StringDType` (`T`). Arithmetic operators, ufuncs, reductions, scientific routines, NumPy/foreign-memory interoperation, the NumPy C ABI, free-threaded CPython, and full scalar-class protocol compatibility are also outside 0.2. These limits are deliberate and remain visible in the 0.2 manifest.
