# Release 0.4: core numeric array routines and measured optimization

**Status: core numeric implementation and local validation are in place; release gates remain open.** The [0.4 compatibility contract](../compat/raptors-0.4.json) narrows the scope to 153 reviewed inventory entries, with native routines and differential tests built on the published [0.3 release](RELEASE_0_3.md). Keep NumPy 2.5.3, source commit `dd88c0c19b54ad9ed3533224221285bf0873249a`, and the locked oracle environment until a reference update is reviewed. This is not a 0.4 release or release-candidate claim.

## Release outcome and boundary

Release 0.4 turns the numeric array foundation and ufunc engine into a coherent set of everyday array routines, then removes the dominant overhead from those paths. The 0.4 contract names the accepted API families, numeric dtype boundary, aliases, known limits, inventory snapshot, and gates; the generated inventory remains the pinned signature oracle and is not itself behavioral evidence. The implementation includes typed storage snapshots, allocation-reduced contiguous float32 statistics and ufunc paths, and specialized int32 concatenation/sorting plus integer histogram/counting kernels. These optimizations remain subject to the recorded workload and cross-host gates below.

The earlier broad default assigned 834 candidates to 0.4. The audit now assigns **153 inventory entries** to the reviewed numeric routine scope: 127 top-level names and 26 `ndarray` methods, with scientific, I/O, specialized-dtype, and generic-surface entries deliberately kept in 0.5–0.8. The executable contract and checker reject accidental drift back to that broad default. Do not hand-edit the generated JSON or count aliases and re-exports as separate implemented kernels.

| Area | Required 0.4 behavior | Boundary |
| --- | --- | --- |
| Numeric creation and coercion | Numeric `arange`; `ones`, `full`, and the numeric `*_like` constructors; `eye`/`identity`; numeric `linspace`, `logspace`, and `geomspace`; and array/order coercion helpers needed by the other 0.4 routines. | Inputs remain the 0.2 numeric Python/Raptors sources. Foreign buffers, DLPack, memory maps, file-backed construction, and third-party conversion/dispatch protocols remain 0.6 or 0.8. |
| Shape and rearrangement | Functional and matching `ndarray` forms of reshape/ravel/flatten, squeeze/expand-dims, transpose/swapaxes/moveaxis/permute-dims, repeat/tile/roll/flip, broadcast shape/materialization, diagonal/triangular helpers, and coordinate/index transforms selected by the contract. | Preserve documented view-versus-copy, order, strides, writeability, and subclass-independent Raptors behavior. Iterator classes, subclass protocols, and arbitrary stride construction are not implied. |
| Joining and splitting | `concatenate`/`concat`, stack helpers, `block`, and array split helpers, including empty inputs, negative axes, dtype/casting controls where accepted, and `out` where documented. | Object/string/structured arrays and third-party dispatch remain outside the release. Aliases share one implementation and one semantic case table. |
| Reductions and descriptive statistics | Top-level and `ndarray` forms of `sum`, `prod`, min/max, any/all, mean, variance, standard deviation, peak-to-peak, cumulative routines, arg reductions, count/nonzero routines, average, median, percentile/quantile, and their numeric NaN-skipping forms when selected by the contract. The supported numeric variance forms include a broadcastable `mean=`; NaN extrema include `where` and `initial`; median/quantile families honor `overwrite_input` for writable Raptors inputs. | Ufunc methods remain covered by 0.3. Complex ordering restrictions, accumulator dtype, `where`, `initial`, `keepdims`, correction/`ddof`, supplied means, overwrite mutation, weights, and warning behavior are reviewed per routine; apparent similarity is not evidence of shared semantics. |
| Selection, indexing, and mutation helpers | Numeric `where`, nonzero/argwhere/flatnonzero, take/put and along-axis forms, choose/compress/extract/select, copy/copyto/putmask/place, fill-diagonal, delete/insert/append, and index conversion helpers selected by the contract. | All indices and output layouts are preflighted before unsafe access. Match NumPy's observable alias, overlap, repeated-index, and partial-mutation behavior rather than imposing blanket atomicity. |
| Ordering, searching, sets, and bins | Sort/argsort, partition/argpartition, lexsort, searchsorted/digitize, bincount, unique and its named-result variants, numeric set operations, membership tests, and one- and multi-dimensional histograms. | Sort kind/stability, NaN/complex ordering, structured fields, optional returns, empty bins, weights, density normalization, and error ordering are separate contract decisions. Structured/object ordering remains later unless explicitly promoted with its dtype dependency. |
| Numeric convenience routines | `array_equal`, `array_equiv`, `isclose`, `allclose`, and other small numeric helpers that are dependency-complete and deliberately accepted during the inventory audit. | Dot/matrix products, einsum, convolution/correlation, polynomial routines, FFT, linear algebra, and random algorithms remain 0.5. I/O/interoperation is 0.6; specialized dtypes are 0.7; printing, configuration, generic iterators, subclass/dispatch behavior, and remaining modules are 0.8. |

Top-level functions and their `ndarray` counterparts must agree where NumPy routes them through the same contract, while retaining any documented signature or return-form differences. Existing 0.2 and 0.3 behavior remains a regression gate. There is no NumPy runtime fallback and no silent use of a later-release protocol.

### Deferred-family ownership

The inventory audit must assign every rejected candidate to a dependency-owning release, not to an unspecified backlog:

| Destination | Families excluded from 0.4 |
| --- | --- |
| 0.5 | Dot/matrix products and generalized matrix ufuncs; einsum; correlation/convolution; polynomial APIs; FFT and linear algebra; interpolation, differencing/gradient, integration, window/special numeric transforms; numeric random generators and state. |
| 0.6 | File/text/NPY/NPZ APIs; memory maps; array byte/list/file serialization; foreign buffers and ownership; DLPack, array interfaces/structs, `ctypes`, `device`/`to_device`, and explicit NumPy adapters. |
| 0.7 | Datetime/timedelta and business-day behavior; string/bytes, void/structured/record, object, and masked arrays; plus specialized-dtype forms of operations whose numeric forms landed earlier. |
| 0.8 | Printing/representation and process option state; generic iterator/helper objects; callback-driven helpers such as `apply_along_axis`, `fromfunction`, `piecewise`, `vectorize`, and `frompyfunc`; ownership-sensitive `ndarray.resize`; `like=` dispatch; matrix/matlib; remaining scalar/dtype class APIs; dispatch, namespace and subclass protocols; typing/testing/f2py; configuration/introspection; constants/aliases; and public-module/export completion. |

These assignments are mirrored in the [roadmap's exclusion-ownership table](CONVERSION_ROADMAP.md#ownership-of-the-04-exclusions). If an API spans families, schedule its basic numeric form at the earliest dependency-complete release and its specialized dtype or protocol variants with their later dependency, with both boundaries stated in the relevant contracts.

## Dependency and implementation sequence

Implement the release as separately reviewable vertical slices. Each slice includes its contract entries, reference probes, native implementation, differential tests, and benchmark rows before the next semantic family begins.

1. **Freeze the 0.4 inventory — complete.** Audit the 834 provisional entries, narrow the generator rules, regenerate `compat/numpy-api-2.5.3.json`, and create `compat/raptors-0.4.json`. The contract records canonical routines, aliases, applicable `ndarray` counterparts, numeric dtype boundary, signature-source digest, and explicit deferrals; `scripts/check_api_inventory.py` rejects bucket drift. Per-entry semantic and signature cases remain part of implementation closure.
2. **Build shared axis, output, and iteration plans.** Normalize integer/tuple/negative axes, `keepdims`, output shapes, dtype/accumulator selection, casting, order, masks, and overlap before execution. Reuse the checked signed-stride and ufunc iteration machinery instead of adding independent per-function walkers. Separate semantic preflight from writes so error precedence is testable.
3. **Complete constructors and shape operations.** Add numeric creators, coercion/order helpers, rearrangement, joining, and splitting. Establish correct view/copy/alias behavior before reductions or sorting depend on it. Include zero-dimensional and zero-size arrays from the start.
4. **Complete reductions and statistics.** First route simple associative reductions through proven 0.3 kernels without changing their top-level semantics; then add mean/variance/statistics, arg reductions, cumulative forms, quantiles, weights, and NaN policies. Use deliberate accumulator algorithms and dtype rules; do not inherit them accidentally from a scalar kernel.
5. **Complete selection and ordered operations.** Add indexing/mutation helpers, sorting and partitioning, searching, unique/set routines, and histogram/binning. Preserve stability, repeated-index order, alias snapshots, and mutation timing. Keep algorithm workspaces checked for size overflow and report their memory cost.
6. **Profile complete public calls and optimize measured bottlenecks.** Remove Python-object work inside element loops, repeated scalar encode/decode, avoidable input/output copies, and repeated shape/index allocations. Then add contiguous and collapsed-dimension kernels, output-reuse paths, and validated dtype specializations. Add SIMD or parallel execution only after scalar equivalence, alias safety, threshold, and thread behavior are measured on every applicable target.
7. **Close the release gate.** Run the frozen compatibility, safety, packaging, and performance suites on the candidate commit and wheel matrix. Publish the contract, raw benchmark data, environment, limits, wins, and regressions. Do not tag 0.4 merely because the API tests pass if the separately declared performance gate has not passed.

## Required semantic cases

Every accepted public name needs cases for its full signature and defaults, representative and boundary values, output type/dtype/shape, mutation and sharing, warnings/errors, and each supported call form. Record the exact NumPy call and result plus upstream source path/test ID when a case is adapted. Exact integer/boolean comparison is required; floating comparison needs a routine-specific tolerance and explicit signed-zero, NaN, infinity, and complex treatment.

| Case family | Minimum adversarial coverage |
| --- | --- |
| Axes and shapes | Scalars; zero-dimensional and zero-size axes; negative, tuple, repeated, and out-of-range axes where accepted; `axis=None`; `keepdims`; C/F/transposed/reversed/zero-stride layouts; invalid shapes; and shape/product arithmetic near platform limits. |
| Dtype and outputs | Every accepted numeric dtype family; platform integer aliases and extended precision; accumulator and result dtype; endian-swapped inputs; Python/Raptors scalars; `dtype`, `casting`, `order`, `out`, `where`, `initial`, and `keepdims` where present; invalid/read-only/wrong-shape outputs. |
| Reduction/statistics | Empty and singleton slices; all-masked and all-NaN slices; integer overflow; cancellation; infinities and signed zero; complex inputs; `ddof`/`correction`; weighted averages; quantile endpoints/methods; warning category/count and mutation order on failure. |
| Shape/join/split | Empty input sequences, mixed ranks, zero-length axes, negative axes, uneven splits, nested blocks, view-versus-copy outcomes, order/stride preservation, parent deletion, and overlapping output/input where an output is accepted. |
| Ordering and search | Stable ties, descending and reverse-strided views, NaNs and complex values, signed zero, empty inputs, invalid `kind`/`order`, kth boundaries, sorter validation, left/right insertion, and in-place mutation timing. |
| Set and histogram | Duplicate values, NaNs, signed zero, complex values, optional index/inverse/count outputs, empty inputs, multidimensional axes, unsorted bins, outliers, non-finite ranges, integer/float weights, density normalization, and bin-edge dtype/rounding. |
| Indexing and mutation | Negative/repeated/broadcast indices, along-axis shape rules, masks, non-contiguous destinations, overlapping aliases, cast failures, out-of-bounds indices, zero-size axes, parent destruction, and reference-compatible partial mutation. |

Add deterministic generated operation sequences that combine 0.2 views and assignment, 0.3 ufuncs, and 0.4 routines. Shrink failures into permanent regression cases. Seed faults in axis normalization, accumulator dtype, stable ordering, output preservation, alias snapshots, bin boundaries, warning state, and empty-slice handling to show that the harness fails closed. Missing extensions, unexpected NumPy imports, blanket skips, and expected failures cannot satisfy required cases.

## Safety and resource evidence

- All shape products, axis lengths, offsets, temporary-buffer sizes, index conversions, and histogram/bin counts are checked before allocation or access. Safe Rust cannot cause out-of-bounds, uninitialized, or misaligned reads/writes, including for negative/zero strides and empty arrays.
- Algorithms that mutate in place document when they snapshot inputs and when NumPy exposes partial mutation. Overlapping aliases, repeated indices, panics/errors, and parent destruction are tested through the Python extension.
- Parallel or SIMD paths retain a scalar reference path. Race-free disjointness, floating-error aggregation, cancellation/error behavior, and thresholds have independent review. No shared mutable alias is processed concurrently without a proved partition.
- Rust tests, Miri-compatible storage/execution tests, supported sanitizers, deterministic property cases, the full prior Python suite, clean NumPy-free installation, and all release wheels pass on the candidate revision.
- Each routine family records temporary bytes and native peak memory for representative contiguous and strided inputs. An optimization that improves time by an unbounded or undisclosed memory trade is not accepted.

## Performance contract

The 0.3 baseline shows Raptors slower than NumPy on all ten measured workloads, often by two or three orders of magnitude. Release 0.4 therefore treats optimization as an independent, falsifiable gate; adding APIs does not count as a performance improvement.

Before tuning, check in the benchmark script and workload manifest with fixed input generation, sizes, dtypes, layouts, output reuse, thread settings, warmup, sample count, correctness checks, and measurement method. The suite has two layers:

- **Common-operation guardrail:** creation/coercion, contiguous and strided ufuncs, broadcast and mixed-dtype calls, reusable `out`, reductions, reshape/transpose, concatenate, sort/search, and a representative statistics routine at tiny, medium, and large sizes.
- **Application workflows:** at least three multi-operation workloads drawn from different semantic families, such as normalization (`mean`/`std` plus ufuncs), grouped counting or binning (`where`/`bincount` or histogram), and array assembly/reordering (reshape/transpose/concatenate/sort). Exact workflows are frozen in the checked-in manifest before implementation is tuned to them.

Run equivalent eager public Python calls on prepared equivalent inputs. Time conversion or allocation only where the workflow actually includes it, and report it separately otherwise. Use isolated worker processes, optimized builds, controlled single-thread settings plus practical defaults, sufficient warmup, and enough samples to report median, dispersion, and a 95% confidence interval for the Raptors/NumPy median ratio. Report Python allocations, native allocation counts where measurable, and native peak memory; `tracemalloc` alone is insufficient.

The gate passes only when all of the following are true on the designated reference host, with a confirmation run on at least one materially different supported architecture:

- At least three preregistered application workflows have a Raptors/NumPy median-time ratio of at most `0.80`, and the upper bound of the reported 95% confidence interval is below `1.00`.
- No common-operation cell regresses by more than 10% relative to the checked-in Raptors pre-optimization baseline without a reviewed explanation and acceptance recorded in the report. This guardrail measures Raptors against itself; it does not disguise a remaining loss to NumPy.
- Every timed result passes the same correctness oracle as the compatibility suite, and optimization paths introduce no required skip, warning, mutation, or dtype divergence.
- The report includes every preregistered cell, failed threshold, regression, delegated backend, thread setting, allocation result, and memory tradeoff—not only favorable measurements.

If the NumPy-relative application threshold is not achieved, publish the measurements and keep 0.4 pending or explicitly revise the roadmap before tagging. Do not redefine the workload, input preparation, or sample exclusion rule after observing results without recording a new preregistration and baseline.

## Release gate and evidence bundle

- `compat/raptors-0.4.json` freezes the exact reviewed names, aliases, signatures, dtype/axis/output boundary, per-entry cases, known limits, and status. Every accepted inventory entry is reconciled with it; every provisional 0.4 entry outside the release is reassigned deliberately.
- All promised differential cases pass against NumPy 2.5.3 on CPython 3.12–3.14 and every supported wheel target, with no required skip or fallback. The complete 0.1–0.3 regression suite remains green.
- Rust, Miri, sanitizer, alias/lifetime, clean-install, metadata, and eight-target stable-ABI wheel gates pass. Candidate wheels import and exercise representative 0.4 calls with NumPy absent.
- The frozen performance suite satisfies the performance contract above. Raw samples, summary calculations, machine/compiler/runtime details, thread settings, native-memory method, and the pre-optimization baseline are committed and reproducible.
- The release record identifies the exact commit, manifest, local commands/results, hosted runs, artifacts, remaining unsupported behavior, and every accepted exception. README compatibility and performance text is generated only after publication.

## Remaining implementation and release blockers

The pinned API inventory check passes for all 13,481 NumPy 2.5.3 entries. The latest full Python regression run passes (9,715 passed, 8 skipped, 1 expected `FutureWarning` from the broadcast writeability comparison); the focused routines/ufunc suite also passes (8,941 passed, 7 skipped, 1 warning). The serialized Rust workspace result and the latest release-profile wheel build are recorded in the contract. Two extended-precision test skips remain host-specific because this macOS ARM64 `longdouble` has no extra mantissa bits. One earlier default-parallel workspace rerun transiently SIGTRAP'ed in `einsum_test`; the isolated target and three subsequent parallel reruns passed, so retain the anomaly in the release matrix.

The immutable pre-optimization baseline remains [raptors-0.4-baseline.json](benchmarks/raptors-0.4-baseline.json). The latest frozen 21-sample run after the semantic update is [raptors-0.4-current.json](benchmarks/raptors-0.4-current.json): normalization is `0.912x` NumPy (95% CI `0.887–1.000`), grouped counting is `0.348x` (`0.326–0.362`), and assembly/reordering is `2.683x` (`2.344–3.047`). Only one of the three workflows meets the required median ratio of `0.80`; the application performance gate therefore fails. The earlier post-optimization measurements remain in [raptors-0.4-optimized.json](benchmarks/raptors-0.4-optimized.json) for history. Do not tag or describe this as a 0.4 release.

Remaining release closure includes the full signature/warning/error/alias audit, extended-native-host coverage, the common-operation guardrail against the pre-optimization baseline, CPython/wheel matrix, confirmation architecture, sanitizer/Miri/lifetime and clean-install gates, allocation/peak-memory measurements, and a successful three-workflow performance result. The optimized report is honest release evidence, not a passed release gate.
