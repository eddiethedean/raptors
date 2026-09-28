# Async execution: deferred design topic

**Status: deferred.** The [rebuild plan](REBUILD_PLAN.md) supersedes the former async-first architecture and positioning. Raptors' primary goal is NumPy's public Python functionality through `import raptors as np`.

The current implementation does not establish the previously advertised `dot_async`, `matmul_async`, bounded job queues, cancellation contracts, or predictable service latency. Those claims and examples have been removed.

## Current execution contract

The rebuilt public API should preserve NumPy's eager, synchronous behavior. Rust execution and safe GIL release may enable concurrency, but neither makes a CPU-heavy call awaitable or guarantees event-loop responsiveness.

Storage safety, view mutation, dtype behavior, and correctness come before a scheduler. Parallel kernels require a proven access model for shared arrays.

## Conditions for revisiting async APIs

Consider a separate proposal only after the numeric foundation and performance gates pass and an application workload demonstrates a need. The proposal would have to define:

- Ownership and access to inputs while work is queued or running.
- Bounded queues, backpressure, and resource limits.
- Interaction between worker threads and numerical-backend threads.
- Cancellation of queued work versus completed or running computation.
- Result and exception delivery to Python.
- Shutdown, Python callbacks, and object lifetimes.
- Observable mutation order and behavior of aliased arrays.

Any `*_async` names would be extensions beyond the NumPy compatibility target and need their own tests and benchmarks.

## Evaluation requirements

Measure service latency and throughput under realistic concurrency, including queueing, allocation, copies, and oversubscription. Compare equivalent deployment configurations and disclose limitations.

No service-specific performance advantage is currently established. See [architecture](ARCHITECTURE.md), [performance](PERFORMANCE.md), and [project messaging](README_IDEA.md).
