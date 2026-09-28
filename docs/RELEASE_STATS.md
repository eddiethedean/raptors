# Published release stats

The README parity and performance section describes the newest Raptors version that has actually been published. Each set of figures is tied to an exact package version, NumPy oracle version, compatibility contract, and raw benchmark report.

## Source of truth

- A published compatibility contract lives at `compat/raptors-<minor>.json` and has `status: "published"` plus an exact `package_version`, such as `0.1.0`.
- The contract records differential test counts and the tested Python versions. These counts describe the declared Raptors subset only.
- If benchmarks exist, `evidence.performance_baseline.path` points to an immutable versioned JSON report. The report's Raptors and NumPy versions must match the contract exactly, and each benchmark operation must have a result for both packages.
- `scripts/update_readme_release_stats.py` selects the highest published package version and renders the README section. Its `--check` mode fails when the README is stale or the source evidence is mismatched.

The API inventory is not a parity denominator. Its names and member records have preliminary semantic classification, so do not report a percentage of inventory entries as NumPy compatibility.

## Updating after a release

1. Before publishing, run the differential suite and the release checks for the exact candidate package and pinned NumPy oracle. Keep the test count, Python versions, and unsupported boundary specific to that release.
2. When a benchmark is appropriate, build the optimized candidate and use matching public Python calls, inputs, dtypes, semantics, and correctness checks for both implementations. Record the machine, Python, NumPy, Raptors, input size, repetition count, and raw results. Use a new path such as `docs/benchmarks/raptors-0.2.0-vs-numpy-2.5.3.json`; never overwrite an older release's evidence. Update the renderer alongside any workload-matrix change so it preserves each input size and benchmark case in the README.
3. Publish the release. Then mark its compatibility contract `published`, set its exact `package_version`, and point `evidence.performance_baseline.path` at the new report. If no benchmark was run, record `status: "not_measured"` and omit `path`; the README will say that no comparison is available instead of carrying forward stale measurements.
4. Regenerate and verify the README:

   ```bash
   python scripts/update_readme_release_stats.py
   python scripts/update_readme_release_stats.py --check
   ```

5. Commit the updated release contract, report, and README together. CI checks that the displayed figures match the newest published contract.

Until the post-publication update lands, the README remains accurately pinned to the previous published release. Do not label an unpublished tag or build as the latest release.

## Interpreting the current measurements

The 0.1.0 report is an informational, single-host baseline across four small Python-level operations. It is not a representative application workload suite and does not establish a general speed ratio. Python `tracemalloc` does not include native Rust allocations; the report's process high-water RSS deltas are coarse and allocator-dependent, so they do not support a memory advantage claim. Expand and control the workload matrix before making broader performance claims; see [performance validation](PERFORMANCE.md).
