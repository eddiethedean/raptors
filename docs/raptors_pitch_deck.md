# Raptors: NumPy functionality, rebuilt in Rust

Updated: 2026-09-28. Source for the accompanying [PDF brief](raptors_pitch_deck.pdf).

## Goal

Provide NumPy's public functionality through `import raptors as np`, with a sound memory model and measured performance improvements. Scripts, notebooks, scientific computing, data processing, and services are all within the intended audience.

## Current status

The repository now contains a narrow 0.1 preview on a separate checked storage path. Local CPython 3.12–3.14 differential/property suites pass on macOS ARM64, along with Rust and Miri checks. The CPython 3.14 wheel and NumPy-free installation checks pass. Hosted cross-platform and AddressSanitizer gates remain pending. This preview does not establish full compatibility, a general memory-safety guarantee, or faster execution.

## Why rebuild

The previous approach declared broad completion without proving behavior. The new effort starts with a pinned NumPy reference, direct behavioral comparisons, and reviewed storage and mutation invariants.

## Execution plan

1. Pin NumPy 2.5.3 and record the legacy baseline and public API inventory.
2. Build a differential harness with seeded faulty-backend checks.
3. Implement checked storage, shared views, and a small Python preview.
4. Pass hosted safety and release-wheel gates before publishing 0.1.
5. Extend dtype, array, and numeric behavior through the 0.x phases.
6. Demonstrate reproducible workload gains only after broader correctness is established.

## Evidence before claims

Compatibility requires matching values, types, mutation, errors, and warnings. Safety requires sound invariants and reviewed unsafe boundaries. Performance requires equivalent Python workloads with timing, memory, and regression results.

## Scope

Changing the import is acceptable. Replacing NumPy's binary interface underneath compiled extensions is not a release requirement; explicit adapters will support consumers requiring NumPy arrays.

Async job APIs, GPU support, JIT compilation, and distributed execution are deferred. Early releases must identify their supported subset; full public functionality remains the destination.

## Next milestone

Run the release workflow's build-only validation on the candidate changes, resolve any cross-platform or sanitizer failures, and publish 0.1 only after its full gate passes.

See the [rebuild plan](REBUILD_PLAN.md) for detailed milestones and acceptance gates.
