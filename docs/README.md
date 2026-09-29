# Raptors documentation

The [rebuild plan](REBUILD_PLAN.md) is the authoritative statement of direction. Raptors aims to provide NumPy's public functionality through a different Python import. Version 0.2.0 is published as a numeric foundation; its exact supported surface is in the [release record](RELEASE_0_2.md) and [compatibility contract](../compat/raptors-0.2.json). The legacy engine is retained separately for audit.

## Plan and engineering

| Document | Purpose |
| --- | --- |
| [Rebuild plan](REBUILD_PLAN.md) | Scope, architecture proposal, phases, and acceptance gates |
| [0.x release roadmap](CONVERSION_ROADMAP.md) | Versioned deliverables, exit gates, and current release status |
| [Release 0.1 execution plan](RELEASE_0_1.md) | Published preview API, validation evidence, and release gate |
| [Release 0.2 execution plan](RELEASE_0_2.md) | Numeric dtype and array foundation scope, work order, risks, and release gate |
| [NumPy dtype architecture](DTYPE_ARCHITECTURE.md) | Full built-in dtype family inventory, shared descriptor/storage design, and release dependencies |
| [Architecture](ARCHITECTURE.md) | Implemented 0.2 storage path, retained legacy code, and proposed later layers |
| [API guide](API_GUIDE.md) | Existing entry points and target compatibility behavior |
| [Migration guide](CONVERSION_GUIDE.md) | Evaluating Python applications with an import change |
| [Contributing](CONTRIBUTING.md) | Evidence required for implementation changes |
| [Test porting](TEST_PORTING.md) | Differential testing and upstream test provenance |
| [Verification record](NUMPY_TEST_VERIFICATION.md) | What has actually been checked |
| [Performance](PERFORMANCE.md) | Benchmark design and performance acceptance criteria |
| [Release stats](RELEASE_STATS.md) | README stats source, refresh process, and release update gate |
| [Compatibility inventory](../compat/README.md) | NumPy API inventory scope and regeneration method |

## Python development

- [Package overview](../raptors-python/README.md)
- [Development setup](../raptors-python/DEVELOPMENT.md)
- [Build and release process](../raptors-python/BUILD.md)
- [Testing guide](../raptors-python/TESTING.md)
- [Test directory map](../raptors-python/tests/README.md)

## Positioning and deferred work

- [Project messaging](README_IDEA.md)
- [Communication strategy](raptors_marketing_strategy.md)
- [Async design status](raptors_vs_numpy_async_design.md)
- [Pitch brief source](raptors_pitch_deck.md) and [PDF](raptors_pitch_deck.pdf)

The older filenames are retained so existing references continue to resolve. Their previous async-first positioning and completion claims are superseded.

## Documentation rules

Describe current behavior separately from proposed behavior. Compatibility statements must name their reference version and supporting tests; performance claims must link to reproducible measurements. Describe unknown behavior as unverified. Vendor documentation and license notices inside development environments are not Raptors project documentation.
