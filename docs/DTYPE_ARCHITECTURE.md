# NumPy dtype implementation plan

This plan covers every NumPy built-in dtype family before implementation grows beyond the numeric foundation. The behavioral oracle stays pinned to NumPy 2.5.3 and the source revision recorded in [`RELEASE_0_1.md`](RELEASE_0_1.md).

## Terms and scope

NumPy uses eleven `dtype.kind` categories: `b i u f c m M O S U V`. A `kind` is a family code, not a complete dtype identifier. Individual dtypes also expose `dtype.char`, which distinguishes concrete built-in scalar types (for example bool, signed byte, unsigned byte, float16, float32, and float64 have different `char` values). C-name aliases can refer to the same fixed-width dtype, and `longdouble`, `int_`, `long`, and pointer-sized integers depend on the platform. Probe and record the pinned reference on every supported wheel target instead of inferring physical layout from a spelling. See NumPy's [dtype reference](https://numpy.org/doc/stable/reference/arrays.dtypes.html), [dtype class list](https://numpy.org/doc/stable/reference/routines.dtypes.html), and [scalar type reference](https://numpy.org/doc/stable/reference/arrays.scalars.html).

NumPy 2.x also has variable-width `StringDType` (`kind` and `char` `T`). It is a separate new-style dtype, outside the eleven legacy `kind` codes. Include it in the complete compatibility target. NumPy documents that its UTF-8 payload is stored outside the ordinary element buffer, so it cannot be modeled as a fixed-width `U` array. See [working with strings and bytes](https://numpy.org/doc/stable/user/basics.strings.html) and [NEP 55](https://numpy.org/neps/nep-0055-string_dtype.html).

## Family inventory and release order

| `kind` | Family and concrete forms | Contract details | Planned implementation |
| --- | --- | --- | --- |
| `b` | Boolean (`bool_`) | One-byte values; NumPy bool is distinct from integer inheritance and has its own casting and reduction rules. | 0.2 numeric foundation |
| `i` | Signed integers (`int8/16/32/64`, C aliases, `intp`) | Exact width and range; C aliases and pointer-sized aliases resolve per platform; byte order applies above one byte. | 0.2 numeric foundation |
| `u` | Unsigned integers (`uint8/16/32/64`, C aliases, `uintp`) | Exact width and range; mixed signed/unsigned promotion and Python integer inference need reference cases. | 0.2 numeric foundation |
| `f` | `float16/32/64`, `longdouble`/`float128` aliases | Rounding, overflow warnings, NaN, infinities, signed zero, subnormals, and platform-specific long-double precision/size/alignment. | 0.2 numeric foundation |
| `c` | `complex64/128`, `clongdouble`/`complex256` aliases | Two components with the corresponding real precision; component casts, real/complex promotion, NaN/infinity, and platform-specific extended precision. | 0.2 numeric foundation |
| `m` | `timedelta64[unit]` | Unit is descriptor metadata; preserve integer ticks, unit conversion/overflow, `NaT`, generic-unit behavior, and Python scalar conversion. | 0.7 specialized dtypes |
| `M` | `datetime64[unit]` | Unit is descriptor metadata; preserve ticks, `NaT`, calendar `Y`/`M` versus linear units, parsing, conversion, and timezone assumptions. | 0.7 specialized dtypes |
| `O` | Python objects | Each element owns a Python reference. Correct drop, shallow-copy, view, assignment, callback, exception, reentrancy, and GIL behavior are required. | 0.7 specialized dtypes; only after object-lifetime design |
| `S` | Fixed-width bytes strings (`bytes_`) | Width is part of the dtype; define encoding-independent byte handling, zero padding, truncation, null bytes, scalar results, and casts. | 0.7 specialized dtypes |
| `U` | Fixed-width Unicode strings (`str_`) | Width counts Unicode code points; backing storage uses fixed UCS-4 units; test non-ASCII, supplementary characters, padding, truncation, and byte order. | 0.7 specialized dtypes |
| `V` | Fixed-width raw bytes (`void`) | Item size is part of the dtype. It is the base for opaque records and structured dtypes, with field offsets, overlaps, padding, nested fields, and subarray fields. | 0.7 specialized dtypes; raw fixed-size values before structured fields |
| `T` | Variable-width `StringDType` (NumPy 2.x extension) | UTF-8 payload and per-element metadata use an external payload store; include `na_object`, coercion, copying, slicing, mutation, and serialization semantics. | 0.7 specialized dtypes; after payload ownership and null-sentinel design |

The release assignments describe when each family becomes supported. They do not limit the architectural dtype descriptor to numeric-only fields: the descriptor design below must be able to represent the later families without changing array shape/stride ownership invariants.

This document describes the target architecture. The 0.2 branch now uses one byte-addressed owner for fixed-width numeric values, with local-copy decoding for unaligned access and explicit native/non-native endian conversion. On Windows and Apple ARM64, `longdouble` and `clongdouble` keep their distinct `g` and `G` dtype codes and scalar wrappers while using the same precision as `float64` and `complex128`. The recursive descriptor and specialized-family owners below remain future architecture; the current `DType` is still a numeric enum.

## Shared descriptor model

Use an immutable, recursively representable dtype descriptor instead of growing a single enum that conflates scalar width, byte order, and flexible metadata. At minimum the descriptor records:

- A family/scalar code and canonical NumPy name, plus alias resolution that records which public spellings resolve to the same descriptor on the current platform.
- `itemsize`, `alignment`, `kind`, `char`, byte order, native-order status, and the exact array-interface format where one exists.
- Flexible-width parameters such as bytes, Unicode, and void item size; datetime/timedelta units; and any StringDType parameters that affect equality, conversion, or missing values.
- Optional recursive subarray descriptor plus shape, or an ordered list of structured fields. Each field records its name, optional title, byte offset, nested descriptor, and optional subarray shape. The parent records total size and aligned/packed layout.
- Whether the descriptor contains Python references and therefore requires a Python-aware owner and object lifetime path.

Descriptor equality and hashing must follow the pinned NumPy behavior, including byte order, units, field layout/titles, subarray shape, and StringDType parameters. Constructors must reject invalid sizes, units, field overlaps where disallowed, offsets outside the parent item, impossible alignments, and recursive definitions that cannot be represented safely.

## Storage and access model

Separate the array layout from the storage owner and from dtype interpretation:

1. An array/view carries shape, signed byte strides, byte offset, writeability, and a shared owner. Layout validation checks every reachable byte range before access.
2. Fixed-width numeric, time, fixed string, and void values use a byte-addressed owner. Accessors decode or encode using the descriptor's byte order and alignment. They must not form a typed Rust reference from unchecked or misaligned bytes.
3. Object arrays use a Python-reference owner with explicit strong-reference operations under the supported interpreter/GIL contract. Never treat an object pointer as plain copyable numeric bytes.
4. StringDType uses a fixed-size metadata buffer plus an owned UTF-8 payload arena and per-element missing-value handling. Views retain both owners; mutations update payload ownership without invalidating other views.
5. Structured and subarray dtypes compose the preceding accessors recursively. A field view shares the parent owner and adds the field offset; a subarray field appends its dimensions while retaining checked strides.

All assignment first determines destination coordinates and casting behavior. Snapshot overlapping source values where NumPy's result requires it. Python object callbacks and destructors may execute user code; lock ordering and reentrancy rules must be explicit before releasing the GIL or entering parallel loops.

## Dependency sequence

1. **0.2: fixed-width numeric foundation.** Implement bool, every signed/unsigned width, float and complex widths, platform aliases and extended precision; dtype inference; numeric casts and promotion; byte order/alignment; checked storage/layout; reshape/transpose; integer/slice/boolean/fancy indexing; and assignment. Keep nonnumeric families unsupported in the 0.2 manifest.
2. **0.3–0.5: numeric execution.** Build ufuncs and array algorithms on the same descriptors, promotion/casting plan, safe accessors, and view semantics. Add numeric routines, linear algebra, FFT, and random behavior in the release order already defined by the [roadmap](CONVERSION_ROADMAP.md).
3. **0.6: serialized and foreign representations.** Define per-dtype NPY/NPZ/text behavior, foreign-buffer lifetime and bounds contracts, and explicit NumPy adapters. Object serialization must be opt-in and document pickle execution/security; memory maps and foreign buffers cannot assume every dtype is an inline POD value.
4. **0.7: specialized built-in families.** Add `M`, `m`, `O`, `S`, `U`, and `V`, then StringDType `T`, followed by nested structured and subarray layouts. Order work by dependency: raw fixed-size values and recursive descriptors first; object lifetime and variable payload ownership only after their owners are designed.
5. **0.8–0.9: public dtype surface and closure.** Implement the pinned public dtype classes, aliases, constructors, conversion helpers, protocols, scalar identities, and inventory remainder. Conformance closes only when each in-scope entry has reviewed cases and all required gates pass.

## Evidence by family

Every supported family needs reference cases for construction/inference, descriptor metadata and equality, scalar extraction, casts, view/copy, indexing, assignment, empty arrays, zero-sized axes, non-native byte order where applicable, and exceptions/warnings. Add family-specific cases:

- Numeric: every width and alias on every wheel platform; range edges, large integers, signed/unsigned mixtures, float16 rounding, extended precision, complex components, byte swaps, alignment, and promotion with Python scalars.
- `M`/`m`: every supported unit, unit promotion/conversion, `NaT`, calendar and linear units, overflow, epoch boundaries, and conversions to/from Python date/time objects.
- `S`/`U`/`V`: zero and maximum widths, embedded NULs, truncation/padding, Unicode edge code points, opaque byte round-trips, and exact field/subarray offsets.
- Structured/subarray: packed and aligned layouts, titles, nested fields, offset overlap, field-view mutation, zero-sized subarrays, and dtype equality.
- `O`: reference counts over owner/view/copy deletion, assignment replacement, exceptions from conversion, mutable object aliasing, and reentrant destructors.
- `T`: empty/long UTF-8 values, embedded NULs, sentinel variants, copy/view ownership, assignment rollback, casts, and payload lifetime after parent deletion.

The [compatibility inventory](../compat/README.md) is a backlog map, not evidence. For each release, assign exact inventory entries and case files only after reviewing the pinned NumPy behavior. Do not mark a family supported from a dtype name, an enum variant, or a successful construction example alone.
