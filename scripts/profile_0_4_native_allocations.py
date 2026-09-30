#!/usr/bin/env python3
"""Profile Rust allocator traffic for Phase 0.4 routines and workflows."""
from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import raptors

from bench_0_4 import ROOT, WORKFLOWS, compare_results, peak_rss_bytes, prepare


DEFAULT_OUTPUT = ROOT / "docs/benchmarks/raptors-0.4-native-allocations.json"


ROUTINE_CASES = (
    ("creation", "linspace", "allocation_only"),
    ("shape_and_rearrangement", "transpose", "contiguous_input"),
    ("shape_and_rearrangement", "transpose", "strided_input"),
    ("joining_and_splitting", "concatenate", "contiguous_input"),
    ("joining_and_splitting", "concatenate", "strided_input"),
    ("reductions_and_statistics", "sum", "contiguous_input"),
    ("reductions_and_statistics", "sum", "strided_input"),
    ("selection_and_mutation", "take", "contiguous_input"),
    ("selection_and_mutation", "take", "strided_input"),
    ("ordering_search_sets_and_bins", "sort", "contiguous_input"),
    ("ordering_search_sets_and_bins", "sort", "strided_input"),
    ("ordering_search_sets_and_bins", "histogram", "contiguous_input"),
    ("ordering_search_sets_and_bins", "histogram", "strided_input"),
    ("numeric_convenience", "isclose", "contiguous_input"),
    ("numeric_convenience", "isclose", "strided_input"),
)
ROUTINE_FAMILIES = tuple(dict.fromkeys(family for family, _, _ in ROUTINE_CASES))


def _numeric_array(backend: str, values, dtype: str):
    if backend == "numpy":
        return np.asarray(values, dtype=dtype)
    return raptors.array(values, dtype=getattr(raptors, dtype))


def prepare_routine(backend: str, routine: str, layout: str):
    module = np if backend == "numpy" else raptors

    if routine == "linspace":
        return lambda: module.linspace(0.0, 1.0, 1024, dtype=module.float32)

    if routine == "transpose":
        values = [[(row * 17 + column * 13) % 251 for column in range(32)] for row in range(64)]
        source = _numeric_array(backend, values, "int32")
        if layout == "strided_input":
            source = module.transpose(source)
        return lambda: module.transpose(source)

    if routine == "concatenate":
        values = [[(row * 17 + column * 13) % 251 for column in range(16)] for row in range(128)]
        source = _numeric_array(backend, values, "int32")
        if layout == "strided_input":
            source = module.transpose(source)
        return lambda: module.concatenate((source, source), axis=0)

    if routine == "sum":
        values = [[(row * 17 + column * 13) % 251 - 125 for column in range(32)] for row in range(128)]
        source = _numeric_array(backend, values, "float32")
        if layout == "strided_input":
            source = module.transpose(source)
        return lambda: module.sum(source, axis=0)

    if routine == "take":
        values = [[(row * 17 + column * 13) % 251 for column in range(32)] for row in range(64)]
        source = _numeric_array(backend, values, "int32")
        if layout == "strided_input":
            source = module.transpose(source)
        indices = list(range(0, 32, 2))
        return lambda: module.take(source, indices, axis=1)

    if routine == "sort":
        values = [[(row * 97 + column * 29) % 1009 for column in range(64)] for row in range(64)]
        source = _numeric_array(backend, values, "int32")
        if layout == "strided_input":
            source = module.transpose(source)
        return lambda: module.sort(source, axis=1)

    if routine == "histogram":
        if layout == "strided_input":
            values = [[(index * 37 + 11) % 1024, 0] for index in range(4096)]
            source = _numeric_array(backend, values, "int64")
            source = module.transpose(source)[0]
        else:
            values = [(index * 37 + 11) % 1024 for index in range(4096)]
            source = _numeric_array(backend, values, "int64")
        return lambda: module.histogram(source, bins=32, range=(0, 1024))

    if routine == "isclose":
        left_values = [[((row * 17 + column * 13) % 251 - 125) / 31 for column in range(32)] for row in range(64)]
        right_values = [[value + 1e-6 for value in row] for row in left_values]
        left = _numeric_array(backend, left_values, "float32")
        right = _numeric_array(backend, right_values, "float32")
        if layout == "strided_input":
            left = module.transpose(left)
            right = module.transpose(right)
        return lambda: module.isclose(left, right)

    raise ValueError(f"no native allocation case for routine {routine!r}")


def profile_routine(family: str, routine: str, layout: str, warmup: int) -> dict[str, object]:
    execute = prepare_routine("raptors", routine, layout)
    reference = prepare_routine("numpy", routine, layout)
    expected = reference()
    actual = execute()
    compare_results(actual, expected, np)
    del actual, expected

    for _ in range(warmup):
        result = execute()
        del result
    gc.collect()

    rss_before = peak_rss_bytes()
    raptors._bench.start()
    result = execute()
    (
        allocation_calls,
        zeroed_allocation_calls,
        reallocation_calls,
        deallocation_calls,
        requested_bytes,
        peak_live_bytes,
        live_bytes_at_stop,
    ) = raptors._bench.stop()
    rss_after = peak_rss_bytes()

    return {
        "family": family,
        "routine": routine,
        "input_layout": layout,
        "native_rust_allocator": {
            "allocation_calls": allocation_calls,
            "zeroed_allocation_calls": zeroed_allocation_calls,
            "reallocation_calls": reallocation_calls,
            "deallocation_calls": deallocation_calls,
            "requested_bytes": requested_bytes,
            "peak_live_bytes_above_prepared_inputs": peak_live_bytes,
            "live_bytes_at_stop_above_prepared_inputs": live_bytes_at_stop,
        },
        "process_peak_rss_delta_bytes": (
            None if rss_before is None or rss_after is None else max(0, rss_after - rss_before)
        ),
        "result_retained_until_after_measurement": True,
        "result_type": type(result).__name__,
    }


def profile_workflow(workflow: str, warmup: int) -> dict[str, object]:
    execute = prepare("raptors", workflow)
    reference = prepare("numpy", workflow)
    expected = reference()
    actual = execute()
    compare_results(actual, expected, np)
    del actual, expected

    for _ in range(warmup):
        result = execute()
        del result
    gc.collect()

    rss_before = peak_rss_bytes()
    raptors._bench.start()
    result = execute()
    (
        allocation_calls,
        zeroed_allocation_calls,
        reallocation_calls,
        deallocation_calls,
        requested_bytes,
        peak_live_bytes,
        live_bytes_at_stop,
    ) = raptors._bench.stop()
    rss_after = peak_rss_bytes()

    return {
        "workflow": workflow,
        "native_rust_allocator": {
            "allocation_calls": allocation_calls,
            "zeroed_allocation_calls": zeroed_allocation_calls,
            "reallocation_calls": reallocation_calls,
            "deallocation_calls": deallocation_calls,
            "requested_bytes": requested_bytes,
            "peak_live_bytes_above_prepared_inputs": peak_live_bytes,
            "live_bytes_at_stop_above_prepared_inputs": live_bytes_at_stop,
        },
        "process_peak_rss_delta_bytes": (
            None if rss_before is None or rss_after is None else max(0, rss_after - rss_before)
        ),
        "result_retained_until_after_measurement": True,
        "result_type": type(result).__name__,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("warmup must be non-negative")
    if not hasattr(raptors, "_bench"):
        parser.error(
            "native profiling hooks are missing; rebuild the extension with --features alloc-profile"
        )

    workflow_reports = [profile_workflow(workflow, args.warmup) for workflow in WORKFLOWS]
    routine_reports = [
        profile_routine(family, routine, layout, args.warmup)
        for family, routine, layout in ROUTINE_CASES
    ]
    report = {
        "schema_version": 2,
        "purpose": "diagnostic process-wide Rust allocation and peak-live-byte profile across representative Phase 0.4 routines and frozen workflows; not a latency run",
        "manifest": "docs/benchmarks/raptors-0.4-workloads.json",
        "candidate_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "measurement": {
            "warmup_calls": args.warmup,
            "profiled_calls_per_case": 1,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "raptors": raptors.__version__,
            "numpy": np.__version__,
            "method": "feature-gated Rust GlobalAlloc counters around one correctness-checked eager public call per representative routine case and frozen workflow after imports, inputs, and warmups; cases are profiled serially",
            "allocator_scope": "all Rust allocations in this process during the measured interval; not attributed to individual crates or routines",
            "limitations": [
                "only Rust global-allocator traffic is counted; Python allocator calls are excluded",
                "peak live bytes are measured relative to prepared inputs; workload closures retain their source arrays throughout the profile",
                "unrelated concurrent Rust allocation activity, if any, is included in the process-wide counters",
                "process high-water RSS is included only as a coarse allocator-dependent cross-check",
                "profiling atomics are enabled only in this separate run and are not included in latency measurements",
            ],
        },
        "routine_results": routine_reports,
        "workflow_results": workflow_reports,
        "coverage": {
            "routine_case_count": len(ROUTINE_CASES),
            "routine_families": list(ROUTINE_FAMILIES),
            "workflow_case_count": len(WORKFLOWS),
            "all_routines_in_each_family_profiled": False,
            "note": "representative routines cover every Phase 0.4 family in C-contiguous and strided forms where applicable; ordering/search/sets/bins includes both sort and histogram, while creation has no input layout",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote native allocation profile to {args.output}")


if __name__ == "__main__":
    main()
