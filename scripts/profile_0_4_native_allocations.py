#!/usr/bin/env python3
"""Profile Rust allocator traffic for the frozen Phase 0.4 workflows."""
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

    reports = [profile_workflow(workflow, args.warmup) for workflow in WORKFLOWS]
    report = {
        "schema_version": 1,
        "purpose": "diagnostic process-wide Rust allocation and peak-live-byte profile; not a latency run",
        "manifest": "docs/benchmarks/raptors-0.4-workloads.json",
        "candidate_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "measurement": {
            "warmup_calls": args.warmup,
            "profiled_calls_per_workflow": 1,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "raptors": raptors.__version__,
            "numpy": np.__version__,
            "method": "feature-gated Rust GlobalAlloc counters around one correctness-checked eager workflow call after imports, inputs, and warmups; workflows are profiled serially",
            "allocator_scope": "all Rust allocations in this process during the measured interval; not attributed to individual crates or routines",
            "limitations": [
                "only Rust global-allocator traffic is counted; Python allocator calls are excluded",
                "peak live bytes are measured relative to prepared inputs; workload closures retain their source arrays throughout the profile",
                "unrelated concurrent Rust allocation activity, if any, is included in the process-wide counters",
                "process high-water RSS is included only as a coarse allocator-dependent cross-check",
                "profiling atomics are enabled only in this separate run and are not included in latency measurements",
            ],
        },
        "results": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote native allocation profile to {args.output}")


if __name__ == "__main__":
    main()
