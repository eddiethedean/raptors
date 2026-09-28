# Evaluating migration from NumPy

Raptors' rebuild targets Python applications that can change `import numpy as np` to `import raptors as np` while retaining the same public functionality and defined behavior. The existing prototype is not ready for a general migration.

The [rebuild plan](REBUILD_PLAN.md) supersedes the former guide to rewriting Python code as Rust. The Rust API is an implementation interface; users should not need to rewrite ordinary array expressions into pointer operations.

## Intended adoption path

1. Record the application's NumPy version, public APIs, dtypes, layouts, mutation patterns, and external dependencies.
2. Compare those requirements with the compatibility manifest once v0.1 creates it.
3. Run representative tests against the pinned NumPy reference and rebuilt Raptors.
4. Compare return types, values, errors, warnings, shared-memory effects, and serialized data.
5. Measure complete application workloads, including conversions to external libraries.
6. Adopt a release only when the required behavior has evidence and its limitations are acceptable.

An import-change example describes the goal, not a current compatibility guarantee:

```python
# Original application
import numpy as np

# Target replacement import
# import raptors as np
```

## External libraries

A package that checks for `numpy.ndarray` or uses NumPy's binary interface may require real NumPy arrays. Use explicit conversion adapters and include their copy, dtype, and lifetime behavior in tests. Import compatibility in application code does not automatically replace NumPy inside SciPy, pandas, or compiled extensions.

The current prototype exposes `from_numpy` and `to_numpy`; their names do not establish zero-copy support or full protocol conformance. Those contracts must be verified during the rebuild.

## Cases to include

| Application behavior | Migration check |
| --- | --- |
| Large integers and mixed scalars | Exact values, promotion, and cast errors |
| Slices, transposes, and aliases | Shared mutation, strides, and owner lifetimes |
| In-place operations and `out=` | Overlap results and writeability |
| Reductions | Axis tuples, negative axes, `keepdims`, dtype, empty inputs |
| Random generation | Seed/state and promised stream reproducibility |
| Saved arrays and objects | Format, dtype metadata, and serialization behavior |
| Foreign buffers | Ownership, access synchronization, copy policy |

Match valid defined behavior. Arbitrary raw pointers and out-of-bounds stride tricks cannot inherit a blanket memory-safety guarantee.

## Reporting a mismatch

Provide a minimal case, exact reference and Raptors revisions, platform, expected and observed behavior, and whether conversion or delegated execution occurred. Add the minimized case to the differential corpus. Do not work around a bug by weakening its assertion.

See the [API orientation](API_GUIDE.md), [testing guide](../raptors-python/TESTING.md), and [performance requirements](PERFORMANCE.md).
