# Raptors Python 0.1 preview

Raptors is rebuilding toward NumPy's public Python functionality through
`import raptors as np`. Version 0.1 is a narrow, native Rust preview. It is not
a general NumPy replacement and makes no performance claim.

The preview supports explicit `bool`, `int64`, `uint64`, `float32`, and
`float64` arrays; shape, dtype, size, and signed byte-stride metadata; basic
integer and slice views; scalar assignment; exact-shape, same-dtype array
assignment; and independent copies. The supported calls are listed in the
[0.1 compatibility manifest](../compat/raptors-0.1.json). Arithmetic,
reductions, dtype inference, casts, broadcasting, reshaping, NumPy interop, and
the rest of NumPy's API remain outside this preview.

Scalar indexing returns typed Raptors scalar wrappers. They support dtype
inspection and basic conversion/comparison, but are not NumPy scalar classes
and do not implement full scalar arithmetic or protocol behavior.

```python
import raptors

a = raptors.array([[1, 2], [3, 4]], dtype=raptors.int64)
reverse_column = a[::-1, 1]
reverse_column[0] = 9
```

The extension uses the separate [`raptors-storage`](../raptors-storage)
crate. It owns initialized typed Rust vectors, shares views through checked
reference-counted storage, and does not import NumPy at runtime. NumPy 2.5.3 is
used only by the test suite as the pinned behavior reference.

## Build and test

Use CPython 3.12–3.14. From the repository root, build with maturin against
`raptors-python/Cargo.toml`, then run the preview suite in
`raptors-python/tests/preview`. The repository's `uv.lock` pins the oracle and
test tools. See [BUILD.md](BUILD.md), [TESTING.md](TESTING.md), and the
[0.1 release gate](../docs/RELEASE_0_1.md).

Only a clean wheel install without NumPy is currently part of the release
package. Source distributions are not published until a clean source build is
verified.
