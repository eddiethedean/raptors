# Evaluating migration from NumPy

Raptors' long-term target is Python applications that can change `import numpy as np` to `import raptors as np` while retaining the public functionality they use. Version 0.1 is a narrow preview and is not suitable for general migration. Read the [compatibility manifest](../compat/raptors-0.1.json) before trying it.

The rebuild supersedes the former guide to rewriting Python code as Rust. The Rust API is an implementation interface; ordinary NumPy-style application code should not need to be rewritten as pointer operations.

## Current preview boundary

The 0.1 package supports explicit `bool`, `int64`, `uint64`, `float32`, and `float64` construction; metadata; basic integer and slice views; scalar assignment; exact-shape, same-dtype assignment from another Raptors array; and copies. Dtype inference, arithmetic, reductions, broadcasting, reshape/transpose, advanced indexing, full NumPy scalar behavior, NumPy interoperation, and most public APIs are not implemented.

Changing the import is a future adoption path, not a compatibility guarantee today:

```python
# Original application
import numpy as np

# Future replacement, only when the application's used APIs are verified
# import raptors as np
```

## Evaluating a future migration

1. Record the application's NumPy version, public APIs, dtypes, layouts, mutation patterns, and external dependencies.
2. Compare those requirements with the pinned API inventory and release-specific manifest.
3. Run the application's tests against the exact NumPy reference and matching Raptors release.
4. Compare return types, values, errors, warnings, aliases, lifetimes, and serialized data.
5. Measure full workflows, including conversions to external libraries and allocations.
6. Adopt only when required behavior has evidence and the release's limits are acceptable.

## External libraries

A package that checks for `numpy.ndarray` or uses NumPy's binary interface may require real NumPy arrays. Future explicit adapters must specify their copy, dtype, ownership, and lifetime behavior. Import compatibility in application code does not replace NumPy underneath SciPy, pandas, or compiled extensions.

The retained legacy source contains methods named `from_numpy` and `to_numpy`; they are not exposed by the 0.1 preview and do not establish zero-copy support or protocol conformance.

## Application cases to include later

| Behavior | Migration check |
| --- | --- |
| Large integers and mixed scalars | Exact values, promotion, and cast errors |
| Slices, transposes, and aliases | Shared mutation, strides, and owner lifetimes |
| In-place operations and `out=` | Overlap results and writeability |
| Reductions | Axis tuples, negative axes, `keepdims`, dtype, empty inputs |
| Random generation | Seed/state and promised stream reproducibility |
| Saved arrays and objects | Format, dtype metadata, and serialization behavior |
| Foreign buffers | Ownership, access synchronization, and copy policy |

Match defined behavior. Arbitrary raw pointers and out-of-bounds stride tricks cannot carry a blanket memory-safety guarantee. See [API guide](API_GUIDE.md), [testing guide](../raptors-python/TESTING.md), and [performance requirements](PERFORMANCE.md).
