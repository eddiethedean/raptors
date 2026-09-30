# Compatibility inventory and release manifests

[`raptors-0.2.json`](raptors-0.2.json) is the executable description of the published numeric foundation. [`raptors-0.1.json`](raptors-0.1.json) records the earlier, narrower preview. README release stats use only manifests whose status is `published`. [`numpy-api-2.5.3.json`](numpy-api-2.5.3.json) records the NumPy 2.5.3 public names and members discovered by the pinned inventory generator; it is a planning inventory, not behavioral conformance evidence.

## Inventory scope

The generated inventory has 13,481 entries and no module import failures. Generation includes NumPy's top-level exports, importable non-private package modules other than test support modules, their non-private public members, selected methods on classes defined or exported by the owning module, and the five public ufunc methods `reduce`, `accumulate`, `reduceat`, `outer`, and `at`.

This is a reproducible backlog map, not a machine-readable transcription of every sentence in NumPy's documentation and not evidence that Raptors implements those entries. Private modules, test-only modules, arbitrary pointer behavior, and the NumPy C ABI are outside the target. Aliases remain visible under their public module path; methods on unrelated re-exported classes are not duplicated under each alias.

Each entry records:

- `status`: `partial` for the three 0.1 calls, otherwise `not_implemented`;
- `target_release`: an initial release-family assignment made by the generator;
- `required_case_plan`: a minimum behavior checklist grouped by API kind;
- `case_plan_status`: `authored` for the 0.1 preview calls and `planned_not_authored` for later work;
- `known_limits`: the specific preview boundary or a reminder to review per-entry semantics before implementation.

The release assignments have been narrowed so the 0.2 bucket contains numeric dtype descriptors and the array foundation, rather than every `ndarray`, scalar, and ufunc member. Later assignments and generic case plans remain preliminary. Review and specialize them against versioned NumPy documentation before implementing each API family. A plan entry is not a test case, a conformance claim, or evidence for a release gate. `required_cases` is populated only for the three 0.1 preview calls; it does not imply coverage of the full inventory.

The [0.3 execution plan](../docs/RELEASE_0_3.md) identifies the candidate top-level numeric ufunc set. The generator assigns generalized matrix ufuncs to 0.5 and reserves `numpy.matlib` for later public-submodule review. These are planning assignments; the 0.3 contract must freeze reviewed behavior and test provenance before release.

[`raptors-0.3.json`](raptors-0.3.json) is the current generated implementation-contract draft. It records the 101 public names, canonical aliases, numeric loop signatures, and known gaps. Regenerate it together with the Rust loop tables using [`generate_ufunc_metadata.py`](../scripts/generate_ufunc_metadata.py). Its presence does not imply differential coverage or release qualification.

The [0.4 execution plan](../docs/RELEASE_0_4.md) and [`raptors-0.4.json`](raptors-0.4.json) narrow the former 834-entry candidate bucket to 153 reviewed numeric inventory entries: 127 top-level routines and 26 `ndarray` methods. The generator assigns out scientific routines, I/O/serialization/interoperation, specialized dtype behavior, and generic protocols to their dependency-owning 0.5–0.8 releases. The checker verifies the contract against the inventory and its frozen signature-source digest. Local focused/full Python tests, Rust checks, and a preregistered pre-optimization measurement are now recorded in the contract; release, safety, cross-platform, and performance gates remain pending. Absence from the 0.4 scope is not permission to lose an entry from the full backlog, and a numeric implementation does not complete later specialized-dtype or protocol variants.

## Regeneration

Use the exact NumPy wheel in the lock file in the canonical generation environment: macOS ARM64, CPython 3.14.3. NumPy exposes a small number of inventory members differently across operating systems, so the checked-in inventory and its byte-for-byte release check use this platform and interpreter patch:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/raptors-numpy-api-env uv sync --project raptors-python --extra dev --locked --python 3.14.3 --no-install-project
UV_PROJECT_ENVIRONMENT=/tmp/raptors-numpy-api-env uv run --project raptors-python --extra dev --locked --no-sync python scripts/generate_numpy_api_inventory.py
```

The generator rejects any NumPy version other than 2.5.3 and normalizes address-bearing default representations. CI and release validation run it on the same macOS ARM64/CPython 3.14.3 environment and compare the output byte-for-byte. `scripts/check_api_inventory.py` also checks the reference, entry count, unique sorted names, release assignments, case plans, and preview entries. Review entry-count and target changes in the same change as an intentional reference-version update; do not overwrite the pin silently.
