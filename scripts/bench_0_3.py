#!/usr/bin/env python3
"""Record a correctness-checked Python baseline for the 0.3 ufunc path."""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

try:
    import resource
except ImportError:  # pragma: no cover - resource is unavailable on Windows
    resource = None

ROOT = Path(__file__).resolve().parents[1]

WORKLOADS = (
    "tiny_add",
    "medium_add",
    "large_add",
    "broadcast_add",
    "fortran_add",
    "strided_add",
    "mixed_dtype_add",
    "list_conversion_add",
    "reused_out_add",
    "large_reduce",
)


def peak_rss_bytes():
    if resource is None:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def build(backend, values, dtype):
    if backend == "numpy":
        import numpy as np

        return np.array(values, dtype=dtype)
    import raptors

    return raptors.array(values, dtype=getattr(raptors, dtype))


def take_sample(value, coordinates):
    if getattr(value, "shape", ()):
        selected = value[coordinates]
    else:
        selected = value
    if hasattr(selected, "item"):
        selected = selected.item()
    return selected


def assert_matches_samples(actual, expected, np):
    expected_array = np.asarray(expected)
    assert actual.dtype.name == expected_array.dtype.name
    assert tuple(getattr(actual, "shape", ())) == expected_array.shape
    if expected_array.ndim == 0:
        coordinates = [()]
    elif expected_array.size:
        coordinates = [
            tuple(0 for _ in expected_array.shape),
            tuple(int(size // 2) for size in expected_array.shape),
            tuple(int(size - 1) for size in expected_array.shape),
        ]
    else:
        coordinates = []
    for coordinate in coordinates:
        actual_value = take_sample(actual, coordinate)
        expected_value = take_sample(expected_array, coordinate)
        if expected_array.dtype.kind in "fc":
            np.testing.assert_allclose(actual_value, expected_value, rtol=1e-7, atol=1e-12)
        else:
            assert actual_value == expected_value, (coordinate, actual_value, expected_value)


def prepare(backend, workload, count):
    import numpy as np
    import raptors

    module = np if backend == "numpy" else raptors

    tiny = min(count, 16)
    medium = min(count, 4096)
    if workload == "tiny_add":
        left = build(backend, list(range(tiny)), "int64")
        right = build(backend, [3] * tiny, "int64")
        expected = np.arange(tiny, dtype=np.int64) + 3
        execute = lambda: left + right
    elif workload == "medium_add":
        left = build(backend, [i / 7 for i in range(medium)], "float64")
        right = build(backend, [2.0] * medium, "float64")
        expected = np.arange(medium, dtype=np.float64) / 7 + 2.0
        execute = lambda: left + right
    elif workload == "large_add":
        left_values = [i % 997 for i in range(count)]
        right_values = [i % 127 for i in range(count)]
        left = build(backend, left_values, "float64")
        right = build(backend, right_values, "float64")
        expected = np.asarray(left_values, dtype=np.float64) + np.asarray(right_values, dtype=np.float64)
        execute = lambda: left + right
    elif workload == "broadcast_add":
        rows = min(count, 2048)
        left_values = [[float(row)] for row in range(rows)]
        right_values = [[float(column) for column in range(8)]]
        left = build(backend, left_values, "float32")
        right = build(backend, right_values, "float32")
        expected = np.asarray(left_values, dtype=np.float32) + np.asarray(right_values, dtype=np.float32)
        execute = lambda: left + right
    elif workload == "fortran_add":
        side = max(1, int(medium**0.5))
        left_values = [[float(row * side + column) for column in range(side)] for row in range(side)]
        right_values = [[2.0] * side for _ in range(side)]
        left = build(backend, left_values, "float64").T
        right = build(backend, right_values, "float64").T
        expected = np.asarray(left_values, dtype=np.float64).T + np.asarray(right_values, dtype=np.float64).T
        execute = lambda: left + right
    elif workload == "strided_add":
        left_values = [i % 997 for i in range(count * 2)]
        right_values = [i % 127 for i in range(count * 2)]
        left_base = build(backend, left_values, "int64")
        right_base = build(backend, right_values, "int64")
        left, right = left_base[::2], right_base[1::2]
        expected = np.asarray(left_values[::2], dtype=np.int64) + np.asarray(right_values[1::2], dtype=np.int64)
        execute = lambda: left + right
    elif workload == "mixed_dtype_add":
        left_values = [i % 2000 for i in range(medium)]
        right_values = [i / 13 for i in range(medium)]
        left = build(backend, left_values, "int16")
        right = build(backend, right_values, "float32")
        expected = np.asarray(left_values, dtype=np.int16) + np.asarray(right_values, dtype=np.float32)
        execute = lambda: left + right
    elif workload == "list_conversion_add":
        left_values = [i / 11 for i in range(medium)]
        right_values = [i / 19 for i in range(medium)]
        expected = np.asarray(left_values, dtype=np.float64) + np.asarray(right_values, dtype=np.float64)
        execute = lambda: module.add(
            build(backend, left_values, "float64"),
            build(backend, right_values, "float64"),
        )
    elif workload == "reused_out_add":
        left_values = [i / 11 for i in range(count)]
        right_values = [i / 19 for i in range(count)]
        left = build(backend, left_values, "float64")
        right = build(backend, right_values, "float64")
        output = (
            np.empty(count, dtype=np.float64)
            if backend == "numpy"
            else __import__("raptors").empty(count, dtype=__import__("raptors").float64)
        )
        expected = np.asarray(left_values, dtype=np.float64) + np.asarray(right_values, dtype=np.float64)
        execute = lambda: module.add(left, right, out=output)
    elif workload == "large_reduce":
        values = [i % 127 for i in range(count)]
        source = build(backend, values, "int64")
        expected = np.add.reduce(np.asarray(values, dtype=np.int64))
        execute = lambda: module.add.reduce(source)
    else:
        raise ValueError(workload)
    return execute, expected


def run_worker(backend, workload, count, repeats):
    import numpy as np
    import raptors

    execute, expected = prepare(backend, workload, count)
    expected_array = np.asarray(expected)
    assert_matches_samples(execute(), expected, np)
    gc.collect()
    before_rss = peak_rss_bytes()
    warmup = execute()
    del warmup
    gc.collect()
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    before_snapshot = tracemalloc.take_snapshot()
    timings = []
    peak_live_allocations = 0
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = execute()
        timings.append(time.perf_counter_ns() - started)
        snapshot = tracemalloc.take_snapshot()
        live_delta = sum(
            max(0, stat.count_diff)
            for stat in snapshot.compare_to(before_snapshot, "traceback")
        )
        peak_live_allocations = max(peak_live_allocations, live_delta)
        del result
    _, peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    after_rss = peak_rss_bytes()
    return {
        "backend": backend,
        "workload": workload,
        "configured_max_count": count,
        "output_shape": list(expected_array.shape),
        "output_elements": int(expected_array.size),
        "repeats": repeats,
        "median_latency_ns": int(statistics.median(timings)),
        "python_tracemalloc_peak_delta_bytes": max(0, peak_traced - before_current),
        "python_live_allocations_peak_delta": peak_live_allocations,
        "process_peak_rss_delta_bytes": (
            None if before_rss is None or after_rss is None else max(0, after_rss - before_rss)
        ),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "raptors": raptors.__version__,
        "thread_settings": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=250_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default="docs/benchmarks/raptors-0.3-baseline.json")
    parser.add_argument("--worker", nargs=2, metavar=("BACKEND", "WORKLOAD"))
    args = parser.parse_args()
    if args.count <= 0 or args.repeats <= 0:
        parser.error("count and repeats must be positive")
    if args.worker:
        print(json.dumps(run_worker(*args.worker, args.count, args.repeats), sort_keys=True))
        return

    rows = []
    for backend in ("numpy", "raptors"):
        for workload in WORKLOADS:
            command = [
                sys.executable,
                __file__,
                "--worker",
                backend,
                workload,
                "--count",
                str(args.count),
                "--repeats",
                str(args.repeats),
            ]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            rows.append(json.loads(result.stdout))
    report = {
        "schema_version": 1,
        "candidate_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "purpose": "informational 0.3 ufunc baseline; not a speed claim",
        "measurement": {
            "input_sizes": {"tiny": min(args.count, 16), "medium": min(args.count, 4096), "large": args.count},
            "workloads": list(WORKLOADS),
            "timing": "median wall-clock nanoseconds over repeated equivalent Python API calls in a fresh subprocess",
            "python_memory": "tracemalloc peak bytes and peak net live blocks above baseline; excludes Rust/native allocations",
            "native_memory": "process high-water RSS delta after imports and input preparation; coarse and platform dependent",
            "correctness": "dtype, shape, and representative output values are checked against NumPy before timing",
            "limitations": [
                "single local host and runtime",
                "sampled values verify large results; the differential suite checks full representative arrays",
                "allocator and page-size effects influence memory deltas",
                "the measurements are not a release performance claim",
            ],
        },
        "results": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} benchmark observations to {output}")


if __name__ == "__main__":
    main()
