# Raptors communication strategy

**Direction:** a Rust-backed Python package targeting NumPy's public functionality through a different import. The [rebuild plan](REBUILD_PLAN.md) is authoritative. This replaces the previous async-service positioning.

## Present message

The published 0.2.0 release has a narrow, checked Rust storage path and pinned NumPy comparison tests. The legacy engine remains separate for audit. Its hosted release gates passed; the release does not establish full compatibility, general memory safety, or a performance advantage.

Do not describe the current package as a completed NumPy replacement or production-ready numerical engine. Those claims need release evidence.

## Audience and adoption

The intended audience includes scientific Python users, notebook users, data engineers, and application developers. Begin evaluations with concrete workflows covered by the compatibility manifest; expand as conformance grows.

Application import replacement is the adoption goal. Existing packages requiring actual NumPy arrays may need adapters. Binary replacement underneath precompiled NumPy extensions is not a release requirement.

## Evidence-based milestones

| Communication milestone | Required artifact |
| --- | --- |
| Rebuild announced | Scope, architecture proposal, and explicit prototype status |
| First numeric preview | Passing differential cases, safety review, and known limitations |
| Performance preview | Reproducible workload report including regressions and memory |
| Broader compatibility release | Updated public API manifest and application validation |
| Stable release | Clean wheel installs, supported platform matrix, and complete release gates |

Use examples that the released package actually supports. Keep planned APIs clearly labeled. Publish native and delegated execution coverage separately.

## Performance messaging

Use specific statements such as a measured improvement for a named workload, version, platform, and thread configuration. Report the benchmark method and include conversion costs.

Do not claim Rust automatically beats NumPy or that Raptors is universally superior. NumPy's optimized implementation is the baseline we must measure against. See [performance validation](PERFORMANCE.md).

## Safety messaging

Describe the storage and mutation model, reviewed unsafe boundaries, and the checks performed. Separate Raptors-owned memory from foreign pointers and native backends. Passing tests alone does not prove universal memory safety.

## Success measures

Track verified API behavior, reproducible application results, actionable user regressions, native coverage, and benchmark quality. Repository stars and generated test counts do not measure compatibility.

Async scheduling, GPU execution, JIT compilation, and distributed computing remain deferred. Do not advertise them as existing differentiators.

The [pitch brief](raptors_pitch_deck.md) and its [PDF](raptors_pitch_deck.pdf) carry this same positioning.
