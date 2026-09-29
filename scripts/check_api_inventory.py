#!/usr/bin/env python3
"""Validate the generated NumPy inventory and the 0.1-0.3 release contracts."""
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "compat/numpy-api-2.5.3.json"
CONTRACT = ROOT / "compat/raptors-0.1.json"
DRAFT_CONTRACT = ROOT / "compat/raptors-0.2.json"
UFUNC_CONTRACT = ROOT / "compat/raptors-0.3.json"
DTYPE_PLAN = ROOT / "docs/DTYPE_ARCHITECTURE.md"
UFUNC_KERNELS = ROOT / "raptors-storage/src/ufunc.rs"
UFUNC_SIGNATURES = ROOT / "raptors-storage/src/ufunc_signatures.rs"
UFUNC_RESOLVER = ROOT / "raptors-storage/src/ufunc_loop_resolver.rs"
RELEASES = {f"0.{minor}" for minor in range(1, 9)}
PREVIEW = {"numpy.array", "numpy.zeros", "numpy.empty"}


def main():
    inventory = json.loads(INVENTORY.read_text())
    contract = json.loads(CONTRACT.read_text())
    draft_contract = json.loads(DRAFT_CONTRACT.read_text())
    ufunc_contract = json.loads(UFUNC_CONTRACT.read_text())
    entries = inventory.get("entries")
    if not isinstance(entries, list):
        raise SystemExit("inventory entries must be a JSON array")
    if inventory.get("schema_version") != 1:
        raise SystemExit("unsupported API inventory schema")
    if inventory.get("reference") != {
        "distribution": "numpy",
        "version": "2.5.3",
        "source_tag": "v2.5.3",
        "source_commit": "dd88c0c19b54ad9ed3533224221285bf0873249a",
        "python_lock": "raptors-python/uv.lock",
    }:
        raise SystemExit("API inventory reference pin changed unexpectedly")
    generation = inventory.get("generation", {})
    if generation.get("entry_count") != len(entries):
        raise SystemExit("inventory entry_count does not match entries")
    if generation.get("import_failures"):
        raise SystemExit(f"inventory has import failures: {generation['import_failures']}")
    names = [entry.get("name") for entry in entries]
    if names != sorted(names) or len(names) != len(set(names)):
        raise SystemExit("inventory names must be unique and sorted")
    for entry in entries:
        if entry.get("status") not in {"partial", "not_implemented"}:
            raise SystemExit(f"invalid inventory status for {entry.get('name')}")
        if entry.get("target_release") not in RELEASES:
            raise SystemExit(f"invalid target release for {entry.get('name')}")
        if not entry.get("required_case_plan") or not entry.get("known_limits"):
            raise SystemExit(f"missing case plan or known limits for {entry.get('name')}")
        if entry.get("case_plan_status") not in {"authored", "planned_not_authored"}:
            raise SystemExit(f"invalid case plan status for {entry.get('name')}")
    release_by_name = {entry["name"]: entry["target_release"] for entry in entries}
    expected_release = {
        "numpy.add": "0.3",
        "numpy.maximum": "0.3",
        "numpy.reciprocal": "0.3",
        "numpy.ufunc": "0.3",
        "numpy.ufunc.reduce": "0.3",
        "numpy.errstate": "0.3",
        "numpy.max": "0.4",
        "numpy.matmul": "0.5",
        "numpy.matvec": "0.5",
        "numpy.ndarray.__matmul__": "0.5",
        "numpy.vecdot": "0.5",
        "numpy.vecmat": "0.5",
        "numpy.matrix": "0.8",
        "numpy.matlib.add": "0.8",
    }
    for name, release in expected_release.items():
        if release_by_name.get(name) != release:
            raise SystemExit(f"{name} must be assigned to {release}, got {release_by_name.get(name)!r}")
    numeric_ufunc_count = sum(
        entry["kind"] == "ufunc"
        and entry["target_release"] == "0.3"
        and entry["name"].count(".") == 1
        for entry in entries
    )
    if numeric_ufunc_count != 101:
        raise SystemExit(f"expected 101 top-level 0.3 ufunc names, got {numeric_ufunc_count}")
    preview_entries = {entry["name"]: entry for entry in entries if entry["name"] in PREVIEW}
    if set(preview_entries) != PREVIEW:
        raise SystemExit(f"preview API names missing from inventory: {sorted(PREVIEW - set(preview_entries))}")
    for entry in preview_entries.values():
        if entry["status"] != "partial" or entry["target_release"] != "0.1" or entry["case_plan_status"] != "authored":
            raise SystemExit(f"preview API state is inconsistent: {entry['name']}")
    contract_inventory = contract.get("api_inventory", {})
    if contract_inventory.get("entry_count") != len(entries):
        raise SystemExit("0.1 contract inventory count is stale")
    if draft_contract.get("release") != "0.2":
        raise SystemExit("0.2 compatibility contract has the wrong release identifier")
    release_status = draft_contract.get("status")
    if release_status not in {"in_progress", "release_candidate", "published"}:
        raise SystemExit(f"invalid 0.2 compatibility contract status: {release_status!r}")
    if draft_contract.get("package_version") != "0.2.0":
        raise SystemExit("0.2 compatibility contract must identify package version 0.2.0")
    draft_inventory = draft_contract.get("api_inventory", {})
    if draft_inventory.get("entry_count") != len(entries):
        raise SystemExit("0.2 contract inventory count is stale")
    draft_reference = draft_contract.get("reference", {})
    if any(
        draft_reference.get(key) != inventory["reference"].get(key)
        for key in ("distribution", "version", "source_tag", "source_commit")
    ):
        raise SystemExit("0.2 snapshot NumPy reference does not match the generated inventory")
    numeric_kinds = set(draft_contract.get("scope", {}).get("dtype_kinds", []))
    if numeric_kinds != {"b", "i", "u", "f", "c"}:
        raise SystemExit("0.2 snapshot must remain scoped to the five numeric dtype kinds")
    release_gates = draft_contract.get("evidence", {}).get("release_gates")
    if release_status == "in_progress" and release_gates not in {"pending", "not_passed"}:
        raise SystemExit("0.2 work-in-progress contract has inconsistent release gate evidence")
    if release_status in {"release_candidate", "published"} and release_gates != "passed":
        raise SystemExit("0.2 release candidate or published release must have passed release gates")
    if release_status == "published":
        evidence = draft_contract.get("evidence", {})
        commit = evidence.get("release_commit", "")
        release = evidence.get("tagged_release", {})
        artifacts = evidence.get("published_artifacts", {})
        wheels = artifacts.get("wheels", [])
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise SystemExit("published 0.2 contract must identify its 40-character release commit")
        if release.get("tag") != "v0.2.0" or release.get("commit") != commit:
            raise SystemExit("published 0.2 tag and release commit evidence are inconsistent")
        if release.get("status") != "published" or release.get("publish_job") != "success":
            raise SystemExit("published 0.2 contract must record a successful PyPI publish job")
        if artifacts.get("distribution") != "raptors==0.2.0" or len(wheels) != 8:
            raise SystemExit("published 0.2 contract must list all eight published wheels")
        if len(set(wheels)) != 8 or any(not wheel.endswith(".whl") for wheel in wheels):
            raise SystemExit("published 0.2 wheel filenames must be unique wheel artifacts")

    if ufunc_contract.get("schema_version") != 1:
        raise SystemExit("unsupported 0.3 ufunc contract schema")
    if ufunc_contract.get("release") != "0.3" or ufunc_contract.get("package_version") != "0.3.0":
        raise SystemExit("0.3 ufunc contract must identify release and package version 0.3.0")
    ufunc_status = ufunc_contract.get("status")
    if ufunc_status not in {"implementation_in_progress", "release_candidate", "published"}:
        raise SystemExit(f"invalid 0.3 ufunc contract status: {ufunc_status!r}")
    ufunc_reference = ufunc_contract.get("reference", {})
    if any(
        ufunc_reference.get(key) != inventory["reference"].get(key)
        for key in ("distribution", "version", "source_tag", "source_commit")
    ):
        raise SystemExit("0.3 ufunc contract NumPy reference does not match the generated inventory")

    ufunc_scope = ufunc_contract.get("scope", {})
    public_names = ufunc_scope.get("top_level_public_ufunc_names", [])
    public_name_map = {item.get("name"): item.get("object_name") for item in public_names}
    if (
        len(public_names) != 101
        or ufunc_scope.get("public_name_count") != len(public_names)
        or len(public_name_map) != len(public_names)
        or None in public_name_map
    ):
        raise SystemExit("0.3 contract must list exactly 101 unique top-level ufunc names")
    inventory_names = {
        entry["name"].removeprefix("numpy.")
        for entry in entries
        if entry["kind"] == "ufunc"
        and entry["target_release"] == "0.3"
        and entry["name"].count(".") == 1
    }
    if set(public_name_map) != inventory_names:
        raise SystemExit("0.3 contract ufunc names do not match the 0.3 API inventory assignments")

    ufunc_objects = ufunc_scope.get("ufunc_objects", [])
    object_map = {item.get("name"): item for item in ufunc_objects}
    if (
        len(object_map) != len(ufunc_objects)
        or len(ufunc_objects) != ufunc_scope.get("canonical_object_count")
    ):
        raise SystemExit("0.3 contract canonical ufunc object count is inconsistent")
    mapped_public_names = {}
    for name, item in object_map.items():
        names = item.get("public_names", [])
        if not name or not names or item.get("nin") not in (1, 2) or item.get("nargs") != item.get("nin") + item.get("nout"):
            raise SystemExit(f"invalid ufunc object metadata in 0.3 contract: {name!r}")
        if not item.get("numeric_types"):
            raise SystemExit(f"0.3 ufunc object has no recorded numeric loops: {name}")
        for public_name in names:
            if public_name in mapped_public_names:
                raise SystemExit(f"duplicate 0.3 public ufunc alias: {public_name}")
            mapped_public_names[public_name] = name
    if mapped_public_names != public_name_map:
        raise SystemExit("0.3 public-name aliases do not match their canonical ufunc objects")

    kernel_source = UFUNC_KERNELS.read_text()
    runtime_names_match = re.search(
        r"pub const TOP_LEVEL_UFUNC_NAMES:\s*&\[&str\]\s*=\s*&\[(.*?)\];",
        kernel_source,
        re.DOTALL,
    )
    if not runtime_names_match:
        raise SystemExit("Rust ufunc kernel name table is missing")
    runtime_names = re.findall(r'"([a-z0-9_]+)"', runtime_names_match.group(1))
    if len(runtime_names) != len(set(runtime_names)) or set(runtime_names) != set(public_name_map):
        raise SystemExit("Rust ufunc names do not match the 0.3 contract public-name list")
    signature_names = set(re.findall(r'^\s*"([a-z0-9_]+)"\s*=>', UFUNC_SIGNATURES.read_text(), re.MULTILINE))
    resolver_names = set(re.findall(r'Entry\s*\{\s*name:\s*"([a-z0-9_]+)"', UFUNC_RESOLVER.read_text()))
    canonical_names = set(object_map)
    if signature_names != canonical_names or resolver_names != canonical_names:
        raise SystemExit("generated Rust signature or loop tables do not match the 0.3 canonical ufunc objects")

    if set(ufunc_scope.get("dtype_kinds", [])) != {"b", "i", "u", "f", "c"}:
        raise SystemExit("0.3 ufunc contract must use the five supported numeric dtype kinds")
    if ufunc_scope.get("runtime_numpy_dependency") is not False:
        raise SystemExit("0.3 ufunc contract must keep the runtime NumPy dependency disabled")
    if not isinstance(ufunc_contract.get("known_gaps"), list):
        raise SystemExit("0.3 ufunc contract known_gaps must be a JSON array")
    if not isinstance(ufunc_contract.get("known_limits"), list):
        raise SystemExit("0.3 ufunc contract known_limits must be a JSON array")
    ufunc_evidence = ufunc_contract.get("evidence", {})
    if ufunc_evidence.get("cargo_check", {}).get("status") not in {"not_run", "passed_locally"}:
        raise SystemExit("0.3 contract has invalid Cargo check evidence")
    if ufunc_status == "implementation_in_progress":
        if ufunc_evidence.get("release_gate") != "pending":
            raise SystemExit("0.3 implementation-in-progress status requires a pending release gate")
        differential = ufunc_evidence.get("differential_cases", {})
        differential_status = differential.get("status")
        if differential_status == "not_run":
            if differential.get("passed", 0) != 0 or differential.get("skipped", 0) != 0:
                raise SystemExit("0.3 not-run differential evidence must not contain test counts")
        elif differential_status == "passed":
            if differential.get("passed", 0) <= 0 or differential.get("skipped") != 0:
                raise SystemExit("0.3 passed differential evidence requires passing cases and zero skips")
        else:
            raise SystemExit("0.3 implementation-in-progress status has invalid differential evidence")
    else:
        differential = ufunc_evidence.get("differential_cases", {})
        if ufunc_evidence.get("release_gate") != "passed":
            raise SystemExit("0.3 release candidate or published status requires passed release gates")
        if ufunc_evidence.get("cargo_check", {}).get("status") != "passed_locally":
            raise SystemExit("0.3 release candidate or published status requires a passed Cargo check")
        if differential.get("status") != "passed" or differential.get("passed", 0) <= 0 or differential.get("skipped") != 0:
            raise SystemExit("0.3 release candidate or published status requires complete differential evidence")
        if ufunc_contract.get("known_gaps"):
            raise SystemExit("0.3 release candidate or published status cannot retain known gaps")
    planned_kinds = set(
        re.findall(r"(?m)^\|\s*`([biufcmMOSUVT])`\s*\|", DTYPE_PLAN.read_text())
    )
    if planned_kinds != set("biufcmMOSUVT"):
        raise SystemExit("dtype plan must cover the eleven legacy families and NumPy 2.x StringDType")
    print(
        f"Validated {len(entries)} NumPy 2.5.3 inventory entries, the 0.1 preview contract, "
        "the 0.2 numeric scope, the 0.3 ufunc contract, and the complete dtype-family plan."
    )


if __name__ == "__main__":
    main()
