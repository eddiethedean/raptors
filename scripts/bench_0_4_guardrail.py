#!/usr/bin/env python3
"""Measure 0.4 routine workloads against the pre-optimization Raptors baseline."""
from __future__ import annotations

import argparse
import gc
import hashlib
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
except ImportError:  # pragma: no cover - resource is unavailable on Windows
    resource = None

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/benchmarks/raptors-0.4-common-workloads-v1.json"


def peak_rss_bytes():
    if resource is None:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def scalar_value(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def array(values, dtype, raptors):
    return raptors.array(values, dtype=getattr(raptors, dtype))


def prepare(workload, size):
    import numpy as np
    import raptors

    indices = range(size)
    if workload == "arange_creation":
        def execute():
            return raptors.arange(0, size, dtype=raptors.int32)

        expected = np.arange(0, size, dtype=np.int32)
    elif workload == "reshape_transpose":
        side = int(size**0.5)
        values = list(indices)
        source = array(values, "int32", raptors)
        expected_source = np.asarray(values, dtype=np.int32)
        def execute():
            return raptors.transpose(raptors.reshape(source, (side, side)))

        expected = np.transpose(np.reshape(expected_source, (side, side)))
    elif workload == "concatenate":
        left_values = [(index * 13) % 501 for index in indices]
        right_values = [(index * 17 + 7) % 503 for index in indices]
        left = array(left_values, "int32", raptors)
        right = array(right_values, "int32", raptors)
        expected = np.concatenate(
            (np.asarray(left_values, dtype=np.int32), np.asarray(right_values, dtype=np.int32))
        )
        def execute():
            return raptors.concatenate((left, right), axis=0)

    elif workload == "mean":
        values = [((index * 17 + 7) % 251 - 125) / 31 for index in indices]
        source = array(values, "float32", raptors)
        expected = np.mean(np.asarray(values, dtype=np.float32))
        def execute():
            return raptors.mean(source)

    elif workload == "take":
        values = list(indices)
        index_values = list(range(0, size, 2))
        source = array(values, "int32", raptors)
        take_indices = array(index_values, "int64", raptors)
        expected = np.take(np.asarray(values, dtype=np.int32), np.asarray(index_values, dtype=np.int64))
        def execute():
            return raptors.take(source, take_indices)

    elif workload == "where":
        condition_values = [index % 2 == 0 for index in indices]
        left_values = [index / 13 for index in indices]
        right_values = [-index / 11 for index in indices]
        condition = array(condition_values, "bool", raptors)
        left = array(left_values, "float32", raptors)
        right = array(right_values, "float32", raptors)
        expected = np.where(
            np.asarray(condition_values, dtype=np.bool_),
            np.asarray(left_values, dtype=np.float32),
            np.asarray(right_values, dtype=np.float32),
        )
        def execute():
            return raptors.where(condition, left, right)

    elif workload == "sort":
        values = [((index * 37 + 11) % 1024) - 512 for index in indices]
        source = array(values, "int32", raptors)
        expected = np.sort(np.asarray(values, dtype=np.int32))
        def execute():
            return raptors.sort(source)

    elif workload == "searchsorted":
        source_values = list(indices)
        query_values = [(index * 37) % size for index in range(0, size, 11)]
        source = array(source_values, "int32", raptors)
        queries = array(query_values, "int32", raptors)
        expected = np.searchsorted(
            np.asarray(source_values, dtype=np.int32),
            np.asarray(query_values, dtype=np.int32),
        )
        def execute():
            return raptors.searchsorted(source, queries)

    elif workload == "bincount":
        values = [index % 512 for index in indices]
        source = array(values, "int64", raptors)
        expected = np.bincount(np.asarray(values, dtype=np.int64), minlength=512)
        def execute():
            return raptors.bincount(source, minlength=512)

    elif workload == "histogram":
        values = [index % 1024 for index in indices]
        source = array(values, "int64", raptors)
        expected = np.histogram(np.asarray(values, dtype=np.int64), bins=64, range=(0, 1024))
        def execute():
            return raptors.histogram(source, bins=64, range=(0, 1024))

    elif workload == "unique":
        distinct = max(2, size // 8)
        values = [index % distinct for index in indices]
        source = array(values, "int32", raptors)
        expected = np.unique(np.asarray(values, dtype=np.int32))
        def execute():
            return raptors.unique(source)

    elif workload == "isclose":
        left_values = [index / 17 for index in indices]
        right_values = [value + 1e-8 for value in left_values]
        left = array(left_values, "float64", raptors)
        right = array(right_values, "float64", raptors)
        expected = np.isclose(
            np.asarray(left_values, dtype=np.float64),
            np.asarray(right_values, dtype=np.float64),
        )
        def execute():
            return raptors.isclose(left, right)

    else:
        raise ValueError(f"unknown workload: {workload}")
    return execute, expected


def assert_matches(actual, expected, np):
    if isinstance(expected, tuple):
        assert isinstance(actual, (tuple, list)) and len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            assert_matches(actual_item, expected_item, np)
        return

    expected_array = np.asarray(expected)
    actual_shape = getattr(actual, "shape", None)
    actual_dtype = getattr(getattr(actual, "dtype", None), "name", None)
    if actual_shape is None:
        actual_shape = ()
        actual_scalar = scalar_value(actual)
        actual_array = np.asarray(actual_scalar)
        if actual_dtype is not None and actual_dtype != expected_array.dtype.name:
            raise AssertionError((actual_dtype, expected_array.dtype.name))
        if expected_array.dtype.kind in "fc":
            np.testing.assert_allclose(actual_array, expected_array, rtol=2e-6, atol=1e-7)
        else:
            np.testing.assert_array_equal(actual_array, expected_array)
        return

    assert tuple(actual_shape) == expected_array.shape, (actual_shape, expected_array.shape)
    assert actual_dtype == expected_array.dtype.name, (actual_dtype, expected_array.dtype.name)
    if not expected_array.size:
        return
    if expected_array.size <= 4096:
        coordinates = list(np.ndindex(expected_array.shape))
    else:
        coordinates = list(dict.fromkeys((
            tuple(0 for _ in expected_array.shape),
            tuple(int(size // 2) for size in expected_array.shape),
            tuple(int(size - 1) for size in expected_array.shape),
        )))
    for coordinate in coordinates:
        key = coordinate[0] if len(coordinate) == 1 else coordinate
        candidate = np.asarray(scalar_value(actual[key]), dtype=expected_array.dtype)
        reference = np.asarray(expected_array[coordinate], dtype=expected_array.dtype)
        if expected_array.dtype.kind in "fc":
            np.testing.assert_allclose(candidate, reference, rtol=2e-6, atol=1e-7)
        else:
            np.testing.assert_array_equal(candidate, reference)


def run_worker(workload, size, warmup, samples):
    import numpy as np
    import raptors

    execute, expected = prepare(workload, size)
    assert_matches(execute(), expected, np)
    for _ in range(warmup):
        result = execute()
        del result
    gc.collect()

    timings = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        result = execute()
        timings.append(time.perf_counter_ns() - started)
        del result

    gc.collect()
    rss_before = peak_rss_bytes()
    tracemalloc.start()
    _, traced_before = tracemalloc.get_traced_memory()
    result = execute()
    _, traced_peak = tracemalloc.get_traced_memory()
    del result
    tracemalloc.stop()
    rss_after = peak_rss_bytes()
    expected_array = np.asarray(expected[0] if isinstance(expected, tuple) else expected)
    return {
        "backend": "raptors",
        "workload": workload,
        "input_size": size,
        "output_shape": list(expected_array.shape),
        "output_dtype": expected_array.dtype.name,
        "warmup_calls": warmup,
        "samples_ns": timings,
        "median_ns": int(statistics.median(timings)),
        "python_tracemalloc_peak_delta_bytes": max(0, traced_peak - traced_before),
        "python_memory_profile_calls": 1,
        "timing_tracemalloc_enabled": False,
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


def git_head():
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def run_suite(candidate_commit, label, warmup, samples, output):
    manifest = json.loads(MANIFEST.read_text())
    environment = os.environ.copy()
    environment.update(manifest["measurement"]["thread_environment"])
    results = []
    for workload in manifest["workloads"]:
        for size_label, size in manifest["measurement"]["input_sizes"].items():
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                workload["name"],
                str(size),
                "--warmup",
                str(warmup),
                "--samples",
                str(samples),
            ]
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            result = json.loads(completed.stdout)
            result["size_label"] = size_label
            result["family"] = workload["family"]
            results.append(result)

    report = {
        "schema_version": 1,
        "label": label,
        "candidate_commit": candidate_commit,
        "benchmark_harness_commit": git_head(),
        "manifest": "docs/benchmarks/raptors-0.4-common-workloads-v1.json",
        "baseline_candidate_commit": manifest["revision"]["baseline_commit"],
        "workload_count": len(manifest["workloads"]),
        "input_sizes": manifest["measurement"]["input_sizes"],
        "results": results,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {len(results)} common-operation cells to {output_path}")


def bootstrap_ratio_interval(baseline_samples, current_samples, iterations=4000, seed=404):
    rng = random.Random(seed)
    ratios = []
    for _ in range(iterations):
        baseline_draw = [rng.choice(baseline_samples) for _ in baseline_samples]
        current_draw = [rng.choice(current_samples) for _ in current_samples]
        ratios.append(statistics.median(current_draw) / statistics.median(baseline_draw))
    ratios.sort()
    return [ratios[int(0.025 * (iterations - 1))], ratios[int(0.975 * (iterations - 1))]]


def compare_reports(baseline_path, current_path, output_path, review_path=None):
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    baseline = json.loads(Path(baseline_path).read_text())
    current = json.loads(Path(current_path).read_text())
    if baseline["candidate_commit"] != manifest["revision"]["baseline_commit"]:
        raise SystemExit("baseline report candidate does not match the preregistered commit")
    if current["baseline_candidate_commit"] != manifest["revision"]["baseline_commit"]:
        raise SystemExit("current report does not identify the preregistered baseline")
    expected_manifest = "docs/benchmarks/raptors-0.4-common-workloads-v1.json"
    if baseline.get("manifest") != expected_manifest or current.get("manifest") != expected_manifest:
        raise SystemExit("measurement reports do not identify the preregistered manifest")
    def key(row):
        return row["workload"], row["input_size"]

    baseline_cells = {key(row): row for row in baseline["results"]}
    current_cells = {key(row): row for row in current["results"]}
    if baseline_cells.keys() != current_cells.keys():
        raise SystemExit("baseline and current reports do not contain identical cells")

    threshold = manifest["regression_threshold"]["maximum_current_to_baseline_median_ratio"]
    review = json.loads(Path(review_path).read_text()) if review_path else None
    accepted_exceptions = {}
    if review is not None:
        if review["manifest"] != expected_manifest:
            raise SystemExit("review file does not identify the preregistered manifest")
        if review["manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest():
            raise SystemExit("review file does not match the frozen preregistered manifest")
        if review["baseline_candidate_commit"] != baseline["candidate_commit"]:
            raise SystemExit("review file does not match the measured baseline commit")
        if review["current_candidate_commit"] != current["candidate_commit"]:
            raise SystemExit("review file does not match the measured candidate commit")
        if review["baseline_report_sha256"] != hashlib.sha256(
            Path(baseline_path).read_bytes()
        ).hexdigest():
            raise SystemExit("review file does not match the measured baseline report")
        if review["current_report_sha256"] != hashlib.sha256(
            Path(current_path).read_bytes()
        ).hexdigest():
            raise SystemExit("review file does not match the measured candidate report")
        for item in review.get("accepted_exceptions", []):
            if item.get("decision") != "accepted":
                continue
            key = (item["workload"], item["input_size"])
            if key in accepted_exceptions:
                raise SystemExit(f"duplicate accepted exception for {key}")
            accepted_exceptions[key] = item["reason"]
    cells = []
    for workload, size in sorted(baseline_cells):
        before = baseline_cells[(workload, size)]
        after = current_cells[(workload, size)]
        ratio = after["median_ns"] / before["median_ns"]
        cell = {
            "workload": workload,
            "input_size": size,
            "size_label": after["size_label"],
            "baseline_median_ns": before["median_ns"],
            "current_median_ns": after["median_ns"],
            "current_to_baseline_median_ratio": ratio,
            "ratio_ci95": bootstrap_ratio_interval(
                before["samples_ns"],
                after["samples_ns"],
                seed=404 + len(cells),
            ),
            "exceeds_10_percent": ratio > threshold,
        }
        if (workload, size) in accepted_exceptions:
            cell["reviewed_accepted_exception"] = accepted_exceptions[(workload, size)]
        cells.append(cell)
    regressions = [cell for cell in cells if cell["exceeds_10_percent"]]
    unaccepted_regressions = [
        cell for cell in regressions if "reviewed_accepted_exception" not in cell
    ]
    regression_keys = {(cell["workload"], cell["input_size"]) for cell in regressions}
    misplaced_exceptions = accepted_exceptions.keys() - regression_keys
    if misplaced_exceptions:
        raise SystemExit(
            f"review file accepts cells that do not exceed the threshold: {sorted(misplaced_exceptions)}"
        )
    summary = {
        "schema_version": 1,
        "status": (
            "passed"
            if not regressions
            else "passed_with_accepted_exceptions"
            if not unaccepted_regressions
            else "regressions_require_review"
        ),
        "manifest": "docs/benchmarks/raptors-0.4-common-workloads-v1.json",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "review_file": str(Path(review_path)) if review_path else None,
        "baseline_report": str(Path(baseline_path)),
        "baseline_report_sha256": hashlib.sha256(Path(baseline_path).read_bytes()).hexdigest(),
        "baseline_candidate_commit": baseline["candidate_commit"],
        "current_report": str(Path(current_path)),
        "current_report_sha256": hashlib.sha256(Path(current_path).read_bytes()).hexdigest(),
        "current_candidate_commit": current["candidate_commit"],
        "threshold_current_to_baseline": threshold,
        "cell_count": len(cells),
        "regression_count": len(regressions),
        "regressions_over_10_percent": regressions,
        "unaccepted_regression_count": len(unaccepted_regressions),
        "unaccepted_regressions": unaccepted_regressions,
        "cells": cells,
    }
    Path(output_path).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Compared {len(cells)} cells; {len(regressions)} exceed the 10% guardrail")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-commit")
    parser.add_argument("--label", choices=("baseline", "current"), default="current")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--output", default="docs/benchmarks/raptors-0.4-common-current-v1.json")
    parser.add_argument("--worker", nargs=2, metavar=("WORKLOAD", "SIZE"))
    parser.add_argument("--baseline-report")
    parser.add_argument("--current-report")
    parser.add_argument("--review-file")
    parser.add_argument("--summary-output")
    args = parser.parse_args()

    if args.worker:
        result = run_worker(args.worker[0], int(args.worker[1]), args.warmup, args.samples)
        print(json.dumps(result))
        return
    if args.baseline_report or args.current_report:
        if not (args.baseline_report and args.current_report and args.summary_output):
            parser.error("comparison requires --baseline-report, --current-report, and --summary-output")
        compare_reports(
            args.baseline_report,
            args.current_report,
            args.summary_output,
            args.review_file,
        )
        return
    if args.review_file:
        parser.error("--review-file is only valid when comparing baseline and current reports")
    if args.warmup < 0 or args.samples < 2:
        parser.error("warmup must be nonnegative and samples must be at least two")
    candidate_commit = args.candidate_commit or git_head()
    run_suite(candidate_commit, args.label, args.warmup, args.samples, args.output)


if __name__ == "__main__":
    main()
