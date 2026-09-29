# Performance validation

**Status: no reproducible NumPy speed advantage has been established.** The existing [Criterion benchmarks](../raptors-core/benches/) measure selected Rust operations. They are useful local probes, but do not demonstrate end-to-end Python performance against NumPy.

The README shows the measurements for the newest published Raptors release. Its generated section links to the exact compatibility contract and raw benchmark JSON. See [release stats maintenance](RELEASE_STATS.md) for the source and required post-release update.

The [rebuild plan](REBUILD_PLAN.md) makes performance a separate acceptance gate after the foundation is correct.

## Baseline before optimization

Release 0.1 has a Python benchmark harness and a local baseline for creation, slicing, assignment, and copying. Release 0.2 records a foundation baseline for numeric casts and layout operations. Neither has a speed threshold. Release 0.3 records the first numeric ufunc path. Release 0.4 profiles and optimizes it against the broader workload matrix below.

The [0.1 JSON report](benchmarks/raptors-0.1-baseline.json) contains eight observations from CPython 3.14.3, NumPy 2.5.3, and macOS 26.5.2 ARM64, using 250,000 int64-compatible values and five measured repetitions per operation. The recorded median times were:

| Operation | NumPy | Raptors preview |
| --- | ---: | ---: |
| Create from the same Python list | 3.75 ms | 7.68 ms |
| Slice a view | 0.67 µs | 0.96 µs |
| Overlapping assignment | 34.2 µs | 7.56 ms |
| Independent copy | 112 µs | 4.25 ms |

This single-host measurement shows lower median latency for NumPy on all four operations, although the slice difference is under one microsecond. It is an informational baseline, not a performance claim or a release-quality comparison. Python `tracemalloc` omits native buffer allocations; the process high-water RSS delta is coarse and allocator-dependent.

The 0.2 release baseline is in [`raptors-0.2.0-baseline.json`](benchmarks/raptors-0.2.0-baseline.json). It records 16 observations for create, slice, cast, reshape, transpose, fancy indexing, assignment, and copy on the same host, input count, and repetition count. Each operation is checked against NumPy on a small input before timing. The report records median call latency, Python `tracemalloc` peak bytes, and a process high-water RSS delta. This is a foundation-level snapshot on one input size and one platform; it does not establish a performance advantage or representative native allocation costs.

Reproduce the 0.2 release baseline after building the extension:

```bash
uv run --project raptors-python --extra dev --no-sync python scripts/bench_0_2.py --count 250000 --repeats 5 --output /tmp/raptors-0.2.0-reproduced.json
```

The 0.1 measurement can be reproduced after installing the locked development environment and building the extension. Write to a temporary path so the committed historical baseline remains unchanged:

```bash
uv run --project raptors-python --extra dev --no-sync python scripts/bench_0_1.py --count 250000 --repeats 5 --output /tmp/raptors-0.1-reproduced.json
```

The command uses a warmup and launches separate processes for each backend/operation. The script and report record interpreter, NumPy/Raptors versions, platform, CPU architecture, and thread environment.

Measure equivalent work through both public Python APIs, with matching inputs, dtypes, layouts, output reuse, and correctness criteria. Include allocations and conversions when an application would incur them.

Record exact versions, compiler/build options, CPU, operating system, numerical backend, thread counts, and fallback state. Use optimized builds, warmups, repeated samples, and variability estimates. Run controlled single-thread comparisons and realistic defaults separately.

## Workload matrix

| Dimension | Cases |
| --- | --- |
| Array size | Tiny arrays dominated by call overhead; medium and large arrays |
| Layout | C/F order, slices, transposes, negative strides, broadcast dimensions |
| Dtype | Integer, float, complex, mixed-type and scalar promotion |
| Output | New allocation, reusable `out=`, in-place and overlapping outputs |
| Operations | Construction, arithmetic, reductions, conversion, and later specialized kernels |
| Applications | Preregistered multi-operation Python workflows |

Report latency, throughput where meaningful, allocation count, native peak memory, and conversion/copy costs. Python-only allocation tracking is insufficient for native buffers.

## Optimization order

1. Remove redundant input copies, metadata allocations, and per-element binding overhead.
2. Improve the validated iteration plan and contiguous fast paths.
3. Add dtype-specific kernels and runtime-selected SIMD where supported.
4. Add parallel thresholds based on measurements and the storage access model.
5. Evaluate specialized numerical backends under the same semantic tests.

NumPy already dispatches optimized SIMD kernels. Rust is an implementation choice, not speed evidence; see [NumPy's SIMD documentation](https://numpy.org/doc/stable/reference/simd/index.html).

Keep scalar reference kernels and run optimized paths against them and NumPy. Preserve eager behavior, numerical requirements, errors, and mutation visibility. Implicit fusion or reassociation cannot change the promised result.

## Proposed 0.4 release gate

Before tuning, finalize the workloads and thresholds proposed in the rebuild plan:

- At least three preregistered application workloads improve median end-to-end time by 20% or more, with evidence beyond measurement noise.
- No regression over 10% on the declared common-operation suite remains unexplained and unaccepted.
- Correctness is maintained, with memory and allocation results reported alongside timing.
- The report includes regressions and delegated execution, not only favorable results.

These thresholds are proposals, not achieved results or a promise that every operation will be faster.

## Existing benchmark commands

From the repository root:

```bash
cargo bench -p raptors-core --bench array_creation
cargo bench -p raptors-core --bench operations
cargo bench -p raptors-core --bench indexing
```

Use [Python build instructions](../raptors-python/BUILD.md) for an optimized extension. Do not infer acceleration from the presence of Rayon or SIMD-related modules.

Async service latency, GPU execution, and distributed throughput are deferred topics, outside the initial NumPy compatibility performance gate.
