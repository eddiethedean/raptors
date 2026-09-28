# Raptors project messaging

This document replaces the previous async-service README concept. Use the [root README](../README.md) for current project information and the [rebuild plan](REBUILD_PLAN.md) for scope.

## Project description

Raptors is rebuilding a Rust-backed Python package with the goal of providing NumPy's public functionality through `import raptors as np`, with a sound memory model and measured performance improvements.

The published 0.1 preview is implemented on a separate checked storage path. It is a small supported subset, not a full NumPy replacement. Its hosted release gates passed, no general memory-safety guarantee is made, and the current benchmark shows no speed advantage.

## Intended audience

Python users working in scripts, notebooks, scientific computing, data processing, and services may benefit once their required APIs are validated. Early evaluation should focus on workloads covered by the compatibility evidence.

The project is not restricted to FastAPI or async applications. Full public NumPy functionality is the destination, with supported subsets disclosed for early releases.

## Claims and supporting evidence

| Claim | Evidence needed before using it |
| --- | --- |
| Compatible with a particular NumPy surface | Pinned reference and passing semantic cases |
| Memory-safe array implementation | Written invariants, reviewed unsafe boundaries, and safety validation |
| Faster for a workload | Reproducible end-to-end benchmark with correctness and memory results |
| Ready for a platform | Clean wheel installation and the required platform tests |

State goals as goals until the evidence exists. Avoid completion percentages based on test totals, blanket speed comparisons, or claims of automatic compatibility with compiled NumPy extensions.

## Deferred topics

Async job APIs, GPU support, JIT compilation, and distributed execution are outside the initial rebuild milestones. The retained [async design note](raptors_vs_numpy_async_design.md) describes their status without promising unavailable methods.

The project and Python package declare MIT; see the root and package `LICENSE` files. Preserve upstream attribution for any code or tests reused later. The current preview tests are authored against NumPy and do not copy upstream test code.
