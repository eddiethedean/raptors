#!/usr/bin/env python3
"""Render or check the README stats for the newest published Raptors release."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CONTRACTS = ROOT / "compat"
START = "<!-- BEGIN GENERATED RELEASE STATS -->"
END = "<!-- END GENERATED RELEASE STATS -->"
VERSION_RE = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)\Z")
OPERATION_LABELS = {
    "create": "Create from the same Python list",
    "slice": "Slice a view",
    "assignment": "Overlapping assignment",
    "copy": "Independent copy",
}


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {path.relative_to(ROOT)}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{path.relative_to(ROOT)} must contain a JSON object")
    return value


def latest_published_contract() -> tuple[str, Path, dict]:
    published = []
    for path in CONTRACTS.glob("raptors-*.json"):
        contract = read_json(path)
        if contract.get("status") != "published":
            continue
        version = contract.get("package_version")
        match = VERSION_RE.fullmatch(version) if isinstance(version, str) else None
        if match is None:
            raise SystemExit(f"published contract {path.name} needs an exact package_version")
        if contract.get("release") != ".".join(match.groups()[:2]):
            raise SystemExit(f"published contract {path.name} release does not match {version}")
        published.append((tuple(map(int, match.groups())), version, path, contract))
    if not published:
        raise SystemExit("no published Raptors compatibility contract was found")
    versions = [entry[0] for entry in published]
    if len(versions) != len(set(versions)):
        raise SystemExit("multiple published compatibility contracts use the same package_version")
    _, version, path, contract = max(published, key=lambda entry: entry[0])
    return version, path, contract


def benchmark_data(contract: dict, version: str) -> tuple[dict | None, dict | None, list[str]]:
    evidence = contract.get("evidence", {}).get("performance_baseline", {})
    if evidence.get("status") in {"not_measured", "unavailable"}:
        return None, None, []
    relative_path = evidence.get("path")
    if not isinstance(relative_path, str):
        raise SystemExit(f"published release v{version} must explicitly record benchmark status")
    path = (ROOT / relative_path).resolve()
    if ROOT not in path.parents:
        raise SystemExit("benchmark path must stay inside the repository")
    report = read_json(path)
    if report.get("schema_version") != 1:
        raise SystemExit(f"unsupported benchmark schema in {relative_path}")
    operations = report.get("measurement", {}).get("operations")
    if not isinstance(operations, list) or not operations or any(not isinstance(op, str) for op in operations):
        raise SystemExit(f"benchmark {relative_path} must list its operations")
    if len(operations) != len(set(operations)):
        raise SystemExit(f"benchmark {relative_path} contains duplicate operation names")
    rows = report.get("results")
    if not isinstance(rows, list):
        raise SystemExit(f"benchmark {relative_path} must contain a results array")

    numpy_version = contract.get("reference", {}).get("version")
    indexed = {}
    environments = set()
    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit(f"benchmark {relative_path} contains an invalid result")
        backend, operation = row.get("backend"), row.get("operation")
        key = (backend, operation)
        if backend not in {"numpy", "raptors"} or operation not in operations or key in indexed:
            raise SystemExit(f"unexpected or duplicate benchmark result: {key}")
        if row.get("raptors") != version or row.get("numpy") != numpy_version:
            raise SystemExit(
                f"benchmark {relative_path} is not pinned to Raptors {version} and NumPy {numpy_version}"
            )
        if not isinstance(row.get("median_latency_ns"), int) or row["median_latency_ns"] <= 0:
            raise SystemExit(f"benchmark {relative_path} has an invalid median")
        environments.add(
            (
                row.get("python"),
                row.get("platform"),
                row.get("machine"),
                row.get("count"),
                row.get("repeats"),
            )
        )
        indexed[key] = row

    expected = {(backend, operation) for backend in ("numpy", "raptors") for operation in operations}
    if set(indexed) != expected:
        raise SystemExit(f"benchmark {relative_path} must have one NumPy and Raptors result per operation")
    if len(environments) != 1:
        raise SystemExit(f"benchmark {relative_path} mixes incompatible environments or inputs")
    return indexed, next(iter(environments)), operations


def format_latency(nanoseconds: int) -> str:
    if nanoseconds < 1_000:
        return f"{nanoseconds} ns"
    if nanoseconds < 1_000_000:
        value, unit = nanoseconds / 1_000, "µs"
    else:
        value, unit = nanoseconds / 1_000_000, "ms"
    if value >= 100:
        number = f"{value:.0f}"
    elif value >= 10:
        number = f"{value:.1f}"
    else:
        number = f"{value:.2f}"
    return f"{number} {unit}"


def format_ratio(raptors_ns: int, numpy_ns: int) -> str:
    if raptors_ns == numpy_ns:
        return "same median"
    if raptors_ns > numpy_ns:
        ratio = raptors_ns / numpy_ns
        delta = format_latency(raptors_ns - numpy_ns)
        direction = "slower"
    else:
        ratio = numpy_ns / raptors_ns
        delta = format_latency(numpy_ns - raptors_ns)
        direction = "faster"
    if ratio >= 100:
        number = f"{ratio:.0f}"
    elif ratio >= 10:
        number = f"{ratio:.1f}"
    else:
        number = f"{ratio:.2f}"
    return f"{number}× {direction} ({delta} {'extra' if direction == 'slower' else 'less'})"


def compact_platform(platform: str, machine: str) -> str:
    label = platform
    for marker in ("-arm64", "-x86_64", "-AMD64"):
        if marker in label:
            label = label.split(marker, maxsplit=1)[0]
            break
    return f"{label.replace('-', ' ')} {machine.upper()}"


def render_stats() -> str:
    version, contract_path, contract = latest_published_contract()
    numpy_version = contract["reference"]["version"]
    contract_link = contract_path.relative_to(ROOT).as_posix()
    release_url = f"https://github.com/eddiethedean/raptors/releases/tag/v{version}"
    pypi_url = f"https://pypi.org/project/raptors/{version}/"

    api = contract.get("api", {})
    dtypes = ", ".join(f"`{dtype}`" for dtype in api.get("dtypes", []))
    creation = [
        item["signature"]
        for key in ("array_function", "zeros_function", "empty_function")
        if (item := api.get(key, {})).get("status") == "implemented" and item.get("signature")
    ]
    creation_text = ", ".join(f"`{item}`" for item in creation)
    metadata = ", ".join(f"`{item}`" for item in api.get("metadata", []))
    indexing = ", ".join(api.get("indexing", []))
    mutation = ", ".join(api.get("mutation", []))
    unsupported = ", ".join(contract.get("unsupported", []))
    tests = contract.get("evidence", {}).get("python_differential_tests", {})
    python_versions = tests.get("python_versions", [])
    passed = tests.get("passed_per_version")
    skipped = tests.get("skipped_per_version")
    if not python_versions or not isinstance(passed, int) or not isinstance(skipped, int):
        raise SystemExit(f"published contract v{version} is missing versioned Python test evidence")
    versions_text = ", ".join(f"CPython {item}" for item in python_versions)
    inventory = contract.get("api_inventory", {})
    inventory_count = inventory.get("entry_count")

    if not dtypes or not creation_text or not metadata or not indexing or not mutation:
        raise SystemExit(f"published contract v{version} is missing declared API scope")
    if not isinstance(inventory_count, int):
        raise SystemExit(f"published contract v{version} is missing the API inventory entry count")
    unsupported_text = (
        f"Unsupported areas: {unsupported}."
        if unsupported
        else "The contract lists no explicitly unsupported areas."
    )

    lines = [
        f"**Latest published release: [`v{version}`]({release_url})** ([PyPI]({pypi_url})).",
        "",
        "### Compatibility",
        "",
        f"Raptors {version} is a narrow preview, not a drop-in NumPy replacement. Its verified surface includes explicit {dtypes} arrays; {creation_text}; metadata ({metadata}); indexing ({indexing}); and mutation/copy behavior ({mutation}).",
        "",
        f"The preview differential/property suite reports **{passed} passed and {skipped} skipped per Python version** on {versions_text}, compared with NumPy {numpy_version}. Those tests cover only the declared preview contract. {unsupported_text}",
        "",
        f"There is **no meaningful whole-NumPy parity percentage**. The {inventory_count:,}-entry API inventory is a preliminary name/member and planning inventory, not behavioral conformance evidence. See the [release contract]({contract_link}) and [inventory limits](compat/README.md).",
        "",
        "### Performance",
        "",
    ]

    indexed, environment, operations = benchmark_data(contract, version)
    if indexed is None:
        lines.extend(
            [
                f"No benchmark has been published for Raptors {version}; no performance comparison is claimed for this release.",
                "",
            ]
        )
    else:
        python_version, platform, machine, count, repeats = environment
        host = compact_platform(platform, machine)
        comparison = [
            (
                operation,
                indexed[("raptors", operation)]["median_latency_ns"]
                - indexed[("numpy", operation)]["median_latency_ns"],
            )
            for operation in operations
        ]
        slower = sum(delta > 0 for _, delta in comparison)
        faster = sum(delta < 0 for _, delta in comparison)
        equal = sum(delta == 0 for _, delta in comparison)
        if slower == len(operations):
            result_summary = f"NumPy had the lower median latency on **all {len(operations)} measured operations**"
        elif faster == len(operations):
            result_summary = f"Raptors had the lower median latency on **all {len(operations)} measured operations**"
        else:
            result_summary = f"Raptors had lower medians on **{faster} of {len(operations)} operations**, higher medians on **{slower}**, and equal medians on **{equal}**"
        lines.extend(
            [
                f"On one {host} host (CPython {python_version}, NumPy {numpy_version}), with {count:,} int64-compatible values and {repeats} repetitions, {result_summary}:",
                "",
                f"| Operation | NumPy {numpy_version} median | Raptors {version} median | Raptors vs NumPy |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for operation in operations:
            numpy_ns = indexed[("numpy", operation)]["median_latency_ns"]
            raptors_ns = indexed[("raptors", operation)]["median_latency_ns"]
            label = OPERATION_LABELS.get(operation, operation.replace("_", " ").capitalize())
            lines.append(
                f"| {label} | {format_latency(numpy_ns)} | {format_latency(raptors_ns)} | {format_ratio(raptors_ns, numpy_ns)} |"
            )
        if "slice" in operations:
            slice_delta = abs(
                indexed[("raptors", "slice")]["median_latency_ns"]
                - indexed[("numpy", "slice")]["median_latency_ns"]
            )
            slice_note = f" The slice delta is {format_latency(slice_delta)} in absolute terms."
        else:
            slice_note = ""
        lines.extend(
            [
                "",
                f"This is one local host and one input size; it does not establish performance for other workloads.{slice_note} Memory is not claimed as a win: `tracemalloc` omits Rust/native buffers, and the recorded process RSS deltas are too coarse for a reliable comparison.",
                "",
                f"See the [raw benchmark report]({contract['evidence']['performance_baseline']['path']}) and [benchmark methodology](docs/PERFORMANCE.md).",
                "",
            ]
        )

    lines.append(f"Evidence is pinned by [the {version} compatibility contract]({contract_link}); this section is generated by [`scripts/update_readme_release_stats.py`](scripts/update_readme_release_stats.py).")
    return "\n".join(lines)


def replace_section(readme: str, content: str) -> str:
    if readme.count(START) != 1 or readme.count(END) != 1:
        raise SystemExit("README must contain exactly one generated release stats marker pair")
    start = readme.index(START) + len(START)
    end = readme.index(END)
    if end < start:
        raise SystemExit("README release stats markers are out of order")
    return readme[:start] + "\n\n" + content + "\n\n" + readme[end:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the README section is stale")
    args = parser.parse_args()
    current = README.read_text()
    expected = replace_section(current, render_stats())
    if args.check:
        if current != expected:
            print("README release stats are stale; run python scripts/update_readme_release_stats.py", file=sys.stderr)
            return 1
        print("README release stats match the latest published contract and benchmark.")
        return 0
    README.write_text(expected)
    print("Updated README release stats from the latest published contract.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
