#!/usr/bin/env python3
"""Run the preregistered 0.4 Python workflow benchmark in isolated workers."""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on Windows
    resource = None

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/benchmarks/raptors-0.4-workloads.json"
WORKFLOWS = (
    "column_normalization",
    "grouped_count_and_histogram",
    "assemble_transpose_and_order",
)
BACKENDS = ("numpy", "raptors")


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


def scalar_value(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def raptors_to_numpy(value, np):
    if isinstance(value, (tuple, list)):
        converted = [raptors_to_numpy(item, np) for item in value]
        return tuple(converted) if isinstance(value, tuple) else converted
    shape = getattr(value, "shape", None)
    dtype = getattr(getattr(value, "dtype", None), "name", None)
    if shape is None or dtype is None:
        return np.asarray(scalar_value(value))
    shape = tuple(shape)
    if not shape:
        return np.asarray(scalar_value(value), dtype=dtype)
    flat = []
    for coordinates in np.ndindex(shape):
        key = coordinates[0] if len(coordinates) == 1 else coordinates
        flat.append(scalar_value(value[key]))
    return np.asarray(flat, dtype=dtype).reshape(shape)


def compare_results(actual, expected, np):
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple) and len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            compare_results(actual_item, expected_item, np)
        return
    actual_array = raptors_to_numpy(actual, np)
    expected_array = np.asarray(expected)
    assert actual_array.dtype == expected_array.dtype, (actual_array.dtype, expected_array.dtype)
    assert actual_array.shape == expected_array.shape, (actual_array.shape, expected_array.shape)
    if expected_array.dtype.kind in "fc":
        np.testing.assert_allclose(actual_array, expected_array, rtol=2e-6, atol=1e-7, equal_nan=True)
    else:
        np.testing.assert_array_equal(actual_array, expected_array)


def prepare(backend, workflow):
    import numpy as np
    import raptors

    module = np if backend == "numpy" else raptors
    if workflow == "column_normalization":
        values = [
            [((row * 17 + column * 13) % 251 - 125) / 31 for column in range(8)]
            for row in range(256)
        ]
        source = build(backend, values, "float32")
        execute = lambda: (source - module.mean(source, axis=0)) / module.std(source, axis=0)
    elif workflow == "grouped_count_and_histogram":
        values = [(index * 37 + 11) % 1024 for index in range(4096)]
        source = build(backend, values, "int64")
        groups = module.remainder(source, 32)
        execute = lambda: (
            module.bincount(groups, minlength=32),
            module.histogram(source, bins=32, range=(0, 1024)),
        )
    elif workflow == "assemble_transpose_and_order":
        values = [[(row * 97 + column * 29) % 1009 for column in range(32)] for row in range(64)]
        source = build(backend, values, "int32")

        def execute():
            reshaped = module.reshape(source, (32, 64))
            transposed = module.transpose(reshaped)
            assembled = module.concatenate((transposed, transposed), axis=0)
            return module.sort(assembled, axis=1)

    else:
        raise ValueError(workflow)
    return execute


def run_worker(backend, workflow, warmup, samples):
    import numpy as np
    import raptors

    execute = prepare(backend, workflow)
    result = execute()
    if backend == "raptors":
        reference_execute = prepare("numpy", workflow)
        compare_results(result, reference_execute(), np)
    else:
        raptors_execute = prepare("raptors", workflow)
        compare_results(raptors_execute(), result, np)
    del result

    for _ in range(warmup):
        del_result = execute()
        del del_result
    gc.collect()
    rss_before = peak_rss_bytes()
    tracemalloc.start()
    _, traced_before = tracemalloc.get_traced_memory()
    timings = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        result = execute()
        timings.append(time.perf_counter_ns() - started)
        del result
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = peak_rss_bytes()
    return {
        "backend": backend,
        "workflow": workflow,
        "samples_ns": timings,
        "median_ns": int(statistics.median(timings)),
        "python_tracemalloc_peak_delta_bytes": max(0, traced_peak - traced_before),
        "process_peak_rss_delta_bytes": (
            None if rss_before is None or rss_after is None else max(0, rss_after - rss_before)
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "raptors": raptors.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "thread_settings": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def bootstrap_ratio_interval(numpy_samples, raptors_samples, iterations=4000):
    rng = random.Random(404)
    ratios = []
    for _ in range(iterations):
        numpy_draw = [rng.choice(numpy_samples) for _ in numpy_samples]
        raptors_draw = [rng.choice(raptors_samples) for _ in raptors_samples]
        ratios.append(statistics.median(raptors_draw) / statistics.median(numpy_draw))
    ratios.sort()
    return [ratios[int(0.025 * (iterations - 1))], ratios[int(0.975 * (iterations - 1))]]


def run_suite(warmup, samples, output):
    manifest = json.loads(MANIFEST.read_text())
    environment = os.environ.copy()
    environment.update(manifest["measurement"]["thread_environment"])
    results = []
    for workflow in WORKFLOWS:
        backend_results = {}
        for backend in BACKENDS:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                backend,
                workflow,
                "--warmup",
                str(warmup),
                "--samples",
                str(samples),
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            backend_results[backend] = json.loads(completed.stdout)
        numpy_result = backend_results["numpy"]
        raptors_result = backend_results["raptors"]
        results.append(
            {
                "workflow": workflow,
                "numpy": numpy_result,
                "raptors": raptors_result,
                "median_ratio": raptors_result["median_ns"] / numpy_result["median_ns"],
                "median_ratio_ci95": bootstrap_ratio_interval(
                    numpy_result["samples_ns"], raptors_result["samples_ns"]
                ),
            }
        )
    report = {
        "manifest": "docs/benchmarks/raptors-0.4-workloads.json",
        "status": "measurement_only_release_gate_pending",
        "warmup_calls": warmup,
        "timed_samples": samples,
        "results": results,
        "native_allocation_count": "unavailable; process peak RSS is a coarse native-memory proxy",
    }
    Path(output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote 0.4 workflow measurements to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--output", default="docs/benchmarks/raptors-0.4-baseline.json")
    parser.add_argument("--worker", nargs=2, metavar=("BACKEND", "WORKFLOW"))
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 3:
        parser.error("warmup must be non-negative and samples must be at least 3")
    if args.worker:
        backend, workflow = args.worker
        if backend not in BACKENDS or workflow not in WORKFLOWS:
            parser.error("invalid worker backend or workflow")
        print(json.dumps(run_worker(backend, workflow, args.warmup, args.samples)))
    else:
        run_suite(args.warmup, args.samples, args.output)


if __name__ == "__main__":
    main()
