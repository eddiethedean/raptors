# Compatibility inventory and 0.1 manifest

[`raptors-0.1.json`](raptors-0.1.json) is the executable description of the supported preview. [`numpy-api-2.5.3.json`](numpy-api-2.5.3.json) records the NumPy 2.5.3 public names and members discovered by the pinned inventory generator.

## Inventory scope

The generated inventory has 13,481 entries and no module import failures. Generation includes NumPy's top-level exports, importable non-private package modules other than test support modules, their non-private public members, selected methods on classes defined or exported by the owning module, and the five public ufunc methods `reduce`, `accumulate`, `reduceat`, `outer`, and `at`.

This is a reproducible backlog map, not a machine-readable transcription of every sentence in NumPy's documentation and not evidence that Raptors implements those entries. Private modules, test-only modules, arbitrary pointer behavior, and the NumPy C ABI are outside the target. Aliases remain visible under their public module path; methods on unrelated re-exported classes are not duplicated under each alias.

Each entry records:

- `status`: `partial` for the three 0.1 calls, otherwise `not_implemented`;
- `target_release`: an initial release-family assignment made by the generator;
- `required_case_plan`: a minimum behavior checklist grouped by API kind;
- `case_plan_status`: `authored` for the 0.1 preview calls and `planned_not_authored` for later work;
- `known_limits`: the specific preview boundary or a reminder to review per-entry semantics before implementation.

The later-release assignments and generic case plans are intentionally preliminary. Review and specialize them against versioned NumPy documentation before implementing each API family. A plan entry is not a test case, a conformance claim, or evidence for a release gate. `required_cases` points only to the current 0.1 tests.

## Regeneration

Use the exact NumPy wheel in the lock file in the canonical generation environment: macOS ARM64, CPython 3.14.3. NumPy exposes a small number of inventory members differently across operating systems, so the checked-in inventory and its byte-for-byte release check use this platform and interpreter patch:

```bash
uv sync --project raptors-python --extra dev --locked --python 3.14.3 --no-install-project
uv run --project raptors-python --extra dev --locked --no-sync python scripts/generate_numpy_api_inventory.py
```

The generator rejects any NumPy version other than 2.5.3 and normalizes address-bearing default representations. CI and release validation run it on the same macOS ARM64/CPython 3.14.3 environment and compare the output byte-for-byte. `scripts/check_api_inventory.py` also checks the reference, entry count, unique sorted names, release assignments, case plans, and preview entries. Review entry-count and target changes in the same change as an intentional reference-version update; do not overwrite the pin silently.
