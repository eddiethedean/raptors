# Contributing to Raptors

Raptors is rebuilding toward NumPy's public Python functionality through `import raptors as np`. Start with the [rebuild plan](REBUILD_PLAN.md), [architecture](ARCHITECTURE.md), and [0.x release roadmap](CONVERSION_ROADMAP.md).

The current engine is a legacy prototype. New work should advance the current acceptance gate; broad feature additions before the storage and comparison infrastructure are proven recreate the original failure mode.

## Set up

Clone the actual repository and follow the [Python development guide](../raptors-python/DEVELOPMENT.md):

```bash
git clone https://github.com/eddiethedean/raptors.git
cd raptors
cargo build -p raptors-core
```

Use an isolated Python environment outside the checkout. Release 0.1 will lock toolchains, reference versions, and the supported platform matrix. Current metadata and CI matrices are historical configurations, not verified support promises.

## Change workflow

1. State the exact NumPy behavior and reference version.
2. Capture oracle cases and meaningful adversarial regressions.
3. Identify storage, aliasing, dtype, or Python-lifetime invariants affected.
4. Implement the smallest coherent change.
5. Review semantics and unsafe assumptions separately.
6. Run relevant correctness, safety, and performance checks.
7. Update the compatibility evidence and documentation.

Do not weaken expected results, add blanket skips, suppress arbitrary errors, or change supported behavior to make a test green. Generated tests need behavioral assertions and reviewed provenance.

## Relevant checks

From the repository root:

```bash
cargo test -p raptors-core --tests
cargo test -p raptors-core --doc
cargo fmt --all -- --check
cargo clippy -p raptors-core -- -D warnings
```

Run the relevant integration target while developing, then the required suite for the change. Legacy failures and warnings must be captured in the v0.1 baseline; these commands are not claimed to pass today. `cargo test --lib` does not run integration tests under `tests/`.

For binding changes, rebuild the extension and run the [Python tests](../raptors-python/TESTING.md) against that artifact. Native linking tests may require platform-specific Python configuration.

## Safety review

Every unsafe block must explain its preconditions and the checks that establish them. Review ownership, layout bounds, initialization, alignment, aliases, access synchronization, and foreign lifetimes. `Arc` alone does not justify `Send` or `Sync`.

Miri, fuzzing, and supported sanitizers supplement review. The new gates are planned infrastructure; record what actually ran and any limitations.

## Review and release evidence

A change description should explain the trigger, resulting behavior, reference cases, verification, and remaining limitations. Performance changes need comparable measurements. Public completion claims must correspond to the compatibility manifest once v0.1 creates it.

Release gates include clean wheel installs, declared-platform tests, real application cases, reconciled license metadata, and evidence-backed documentation. Existing publishing automation is not proof of release readiness.

Preserve NumPy attribution and licensing for reused code or tests. Keep generated build output and machine-specific environments out of future changes.

## Documentation

Use [docs/README.md](README.md) to find the relevant guide. Distinguish current behavior, proposed design, and observed results. Update the canonical rebuild plan when the accepted direction changes, then align related guides. Source-level API comments should follow the same rule.
