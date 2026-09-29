#!/usr/bin/env python3
"""Record an informational Python-level baseline for the 0.2 array foundation."""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

OPERATIONS = ("create", "slice", "cast", "reshape", "transpose", "fancy_index", "assignment", "copy")


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def assert_matches_samples(actual, expected):
    assert actual.dtype.name == expected.dtype.name
    assert tuple(actual.shape) == expected.shape
    coordinates = [tuple(0 for _ in expected.shape)]
    if expected.size:
        coordinates.append(tuple(int(size - 1) for size in expected.shape))
    for coordinate in coordinates:
        actual_value = actual[coordinate]
        expected_value = expected[coordinate]
        if hasattr(actual_value, "item"):
            actual_value = actual_value.item()
        if hasattr(expected_value, "item"):
            expected_value = expected_value.item()
        assert actual_value == expected_value, (coordinate, actual_value, expected_value)


def verify_operation(np, raptors, backend, operation):
    values = list(range(12))
    reference = np.array(values, dtype=np.int64)
    if backend == "numpy":
        create = lambda data: np.array(data, dtype=np.int64)
        index = np.arange(0, len(values), 2, dtype=np.int64)
        target_float = np.float32
    else:
        create = lambda data: raptors.array(data, dtype=raptors.int64)
        index = raptors.array(list(range(0, len(values), 2)), dtype=raptors.int64)
        target_float = raptors.float32

    if operation == "create":
        actual = create(values)
        expected = reference.copy()
    else:
        left = create(values)
        if operation == "slice":
            actual, expected = left[::2], reference[::2]
        elif operation == "cast":
            actual, expected = left.astype(target_float), reference.astype(np.float32)
        elif operation == "reshape":
            actual, expected = left.reshape((len(values), 1)), reference.reshape((len(values), 1))
        elif operation == "transpose":
            actual, expected = left.reshape((len(values), 1)).transpose(), reference.reshape((len(values), 1)).transpose()
        elif operation == "fancy_index":
            actual, expected = left[index], reference[np.arange(0, len(values), 2)]
        elif operation == "assignment":
            left[1::2] = 17
            reference[1::2] = 17
            actual, expected = left, reference
        elif operation == "copy":
            actual, expected = left.copy(), reference.copy()
        else:
            raise ValueError(operation)
    assert_matches_samples(actual, expected)


def run_worker(backend, operation, count, repeats):
    import numpy as np
    import raptors

    values = list(range(count))
    if backend == "numpy":
        create = lambda data: np.array(data, dtype=np.int64)
        target_float = np.float32
    else:
        create = lambda data: raptors.array(data, dtype=raptors.int64)
        target_float = raptors.float32

    if operation != "create":
        left = create(values)
    if operation == "fancy_index":
        index_values = list(range(0, count, 2))
        index = (
            np.arange(0, count, 2, dtype=np.int64)
            if backend == "numpy"
            else raptors.array(index_values, dtype=raptors.int64)
        )

    def execute():
        if operation == "create":
            return create(values)
        if operation == "slice":
            return left[::2]
        if operation == "cast":
            return left.astype(target_float)
        if operation == "reshape":
            return left.reshape((count, 1))
        if operation == "transpose":
            return left.reshape((count, 1)).transpose()
        if operation == "fancy_index":
            return left[index]
        if operation == "assignment":
            left[1::2] = 17
            return None
        if operation == "copy":
            return left.copy()
        raise ValueError(operation)

    verify_operation(np, raptors, backend, operation)
    gc.collect()
    before_rss = peak_rss_bytes()
    warmup = execute()
    del warmup
    gc.collect()
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    timings = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = execute()
        timings.append(time.perf_counter_ns() - started)
        del result
    _, peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    after_rss = peak_rss_bytes()
    return {
        "backend": backend,
        "operation": operation,
        "count": count,
        "repeats": repeats,
        "median_latency_ns": int(statistics.median(timings)),
        "python_tracemalloc_peak_delta_bytes": max(0, peak_traced - before_current),
        "process_peak_rss_delta_bytes": max(0, after_rss - before_rss),
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
    parser.add_argument("--output", default="docs/benchmarks/raptors-0.2.0-baseline.json")
    parser.add_argument("--worker", nargs=2, metavar=("BACKEND", "OPERATION"))
    args = parser.parse_args()
    if args.count <= 0 or args.repeats <= 0:
        parser.error("count and repeats must be positive")
    if args.worker:
        print(json.dumps(run_worker(*args.worker, args.count, args.repeats), sort_keys=True))
        return

    rows = []
    for backend in ("numpy", "raptors"):
        for operation in OPERATIONS:
            command = [
                sys.executable,
                __file__,
                "--worker",
                backend,
                operation,
                "--count",
                str(args.count),
                "--repeats",
                str(args.repeats),
            ]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            rows.append(json.loads(result.stdout))
    report = {
        "schema_version": 1,
        "purpose": "informational 0.2 foundation baseline; does not support a speed claim",
        "measurement": {
            "input": "one Python list(range(count)) of int64-compatible integers",
            "operations": list(OPERATIONS),
            "timing": "median wall-clock nanoseconds over repeated equivalent Python calls in a fresh subprocess",
            "python_memory": "tracemalloc peak delta; excludes Rust/native allocations",
            "native_memory": "process high-water RSS delta after imports and inputs are prepared; coarse and platform dependent",
            "limitations": [
                "single local host",
                "one input size and int64 source dtype",
                "allocator and page-size effects",
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
