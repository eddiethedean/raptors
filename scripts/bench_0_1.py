#!/usr/bin/env python3
"""Record a comparable, informational Python-level 0.1 benchmark baseline."""
from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

OPERATIONS = ("create", "slice", "assignment", "copy")


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def run_worker(backend, operation, count, repeats):
    import numpy as np
    import raptors

    values = list(range(count))
    if backend == "numpy":
        create = lambda data: np.array(data, dtype=np.int64)
    else:
        create = lambda data: raptors.array(data, dtype=raptors.int64)

    if operation != "create":
        left = create(values)
        right = create(values)
    gc.collect()
    before_rss = peak_rss_bytes()

    def execute():
        if operation == "create":
            return create(values)
        if operation == "slice":
            return left[::2]
        if operation == "assignment":
            left[1:] = right[:-1]
            return None
        if operation == "copy":
            return left.copy()
        raise ValueError(operation)

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
            name: __import__("os").environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", default="docs/benchmarks/raptors-0.1-baseline.json")
    parser.add_argument("--worker", nargs=2, metavar=("BACKEND", "OPERATION"))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(run_worker(*args.worker, args.count, args.repeats), sort_keys=True))
        return

    rows = []
    for backend in ("numpy", "raptors"):
        for operation in OPERATIONS:
            command = [sys.executable, __file__, "--worker", backend, operation,
                       "--count", str(args.count), "--repeats", str(args.repeats)]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            rows.append(json.loads(result.stdout))
    report = {
        "schema_version": 1,
        "purpose": "informational preview baseline; does not support a speed claim",
        "measurement": {
            "input": "one Python list(range(count)) of int64-compatible integers",
            "operations": list(OPERATIONS),
            "timing": "median wall-clock nanoseconds over repeated equivalent Python calls in a fresh subprocess",
            "python_memory": "tracemalloc peak delta; excludes Rust/native allocations",
            "native_memory": "process high-water RSS delta after imports and inputs are prepared; coarse and platform dependent",
            "limitations": ["single local host", "allocator and page-size effects", "timing is not a release performance claim"],
        },
        "results": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} benchmark observations to {output}")


if __name__ == "__main__":
    main()
