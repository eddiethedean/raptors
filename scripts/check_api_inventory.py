#!/usr/bin/env python3
"""Validate the generated NumPy inventory and its link to the 0.1 contract."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "compat/numpy-api-2.5.3.json"
CONTRACT = ROOT / "compat/raptors-0.1.json"
RELEASES = {f"0.{minor}" for minor in range(1, 9)}
PREVIEW = {"numpy.array", "numpy.zeros", "numpy.empty"}


def main():
    inventory = json.loads(INVENTORY.read_text())
    contract = json.loads(CONTRACT.read_text())
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
    print(f"Validated {len(entries)} NumPy 2.5.3 inventory entries and the 0.1 preview contract.")


if __name__ == "__main__":
    main()
