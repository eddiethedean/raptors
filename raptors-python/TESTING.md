# Testing Raptors 0.1

The required suite is `tests/preview`; pytest is configured to collect only
that directory by default. Tests import NumPy 2.5.3 as the pinned oracle and
compare it with the new Rust-backed preview. Import failures are fatal, and
required preview tests have no skip or expected-failure markers.

## Python differential and generated cases

```bash
uv sync --project raptors-python --extra dev --locked --python 3.14 --no-install-project
uv run --project raptors-python --extra dev --no-sync maturin develop --manifest-path raptors-python/Cargo.toml --release
uv run --project raptors-python --extra dev --no-sync pytest raptors-python/tests/preview -q
```

The deterministic Hypothesis suite generates integer sequences, nested views,
negative/stepped/empty slices, overlapping assignments, rectangular inputs,
and owner deletion. `test_harness.py` seeds wrong dtype, shape/broadcast,
alias, and owner-lifetime backends to verify that the comparator fails when
those results are wrong. Test provenance and scope live in
[`compat/raptors-0.1.json`](../compat/raptors-0.1.json); the complete public
NumPy inventory is [`compat/numpy-api-2.5.3.json`](../compat/numpy-api-2.5.3.json).

## Rust and safety checks

```bash
rustfmt --edition 2021 --check raptors-storage/src/lib.rs raptors-python/src/preview.rs raptors-python/src/preview_lib.rs raptors-python/build.rs
cargo test --locked -p raptors-storage
cargo clippy --locked -p raptors-storage -p raptors-python --all-targets -- -D warnings
cargo +nightly miri test --locked -p raptors-storage
```

The preview storage crate contains no `unsafe` blocks. CI runs Miri and an
AddressSanitizer build of its isolated layout and mutation tests. The Python
owner-lifetime cases run on every release wheel job.

## Wheel and dependency checks

```bash
maturin build --release --locked --manifest-path raptors-python/Cargo.toml --interpreter python3.14 --out /tmp/raptors-wheels
python scripts/check_wheel_contract.py /tmp/raptors-wheels/raptors-0.1.0-cp314-*.whl --python 3.14
python scripts/check_clean_install.py /tmp/raptors-wheels/raptors-0.1.0-cp314-*.whl
```

The first script checks wheel tags, metadata, runtime requirements, extension,
and license inclusion. The second installs the wheel in a fresh environment
with no NumPy and runs a mutation smoke test. The release workflow executes
these checks across all advertised OS and Python combinations before PyPI
publishing.

## Legacy tests and baseline

The top-level legacy Python tests and the older NumPy-port suite describe the
retired prototype. They are retained for audit but are not included in the 0.1
preview run: their expectations cover operations deferred to later phases.
The captured pre-rebuild observations and failures are recorded in
[`NUMPY_TEST_VERIFICATION.md`](../docs/NUMPY_TEST_VERIFICATION.md).
