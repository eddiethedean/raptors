#!/usr/bin/env python3
"""Validate the generated NumPy inventory and its link to the 0.1 contract."""
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "compat/numpy-api-2.5.3.json"
CONTRACT = ROOT / "compat/raptors-0.1.json"
DRAFT_CONTRACT = ROOT / "compat/raptors-0.2.json"
DTYPE_PLAN = ROOT / "docs/DTYPE_ARCHITECTURE.md"
RELEASES = {f"0.{minor}" for minor in range(1, 9)}
PREVIEW = {"numpy.array", "numpy.zeros", "numpy.empty"}


def main():
    inventory = json.loads(INVENTORY.read_text())
    contract = json.loads(CONTRACT.read_text())
    draft_contract = json.loads(DRAFT_CONTRACT.read_text())
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
        raise SystemExit("0.2 release candidate must have passed release gates")
    planned_kinds = set(
        re.findall(r"(?m)^\|\s*`([biufcmMOSUVT])`\s*\|", DTYPE_PLAN.read_text())
    )
    if planned_kinds != set("biufcmMOSUVT"):
        raise SystemExit("dtype plan must cover the eleven legacy families and NumPy 2.x StringDType")
    print(
        f"Validated {len(entries)} NumPy 2.5.3 inventory entries, the 0.1 preview contract, "
        "the 0.2 numeric scope, and the complete dtype-family plan."
    )


if __name__ == "__main__":
    main()
