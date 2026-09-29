# Evaluating migration from NumPy

Raptors' long-term target is Python applications that can change `import numpy as np` to `import raptors as np` while retaining the public functionality they use. Version 0.2.0 is a limited numeric foundation and is not suitable for general migration. Read the [0.2 compatibility contract](../compat/raptors-0.2.json) before evaluating it; the [0.1 contract](../compat/raptors-0.1.json) documents the earlier release.

The rebuild supersedes the former guide to rewriting Python code as Rust. The Rust API is an implementation interface; ordinary NumPy-style application code should not need to be rewritten as pointer operations.

## Published 0.2 preview boundary

The 0.2 package supports numeric dtype inference and casts; boolean, signed and unsigned integer, floating-point, and complex dtypes; promotion; metadata; reshape and transpose; integer, slice, boolean, and fancy indexing; broadcast assignment; and copies. Arithmetic, ufuncs, reductions, non-numeric dtype families, NumPy interoperation, full NumPy scalar behavior, and most public APIs are not implemented. Use the [API guide](API_GUIDE.md) for details.

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

The retained legacy source contains methods named `from_numpy` and `to_numpy`; they are not exposed by the 0.2 package and do not establish zero-copy support or protocol conformance.

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
