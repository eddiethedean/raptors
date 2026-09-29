#!/usr/bin/env python3
"""Generate the pinned numeric ufunc metadata tables and 0.3 contract draft."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PINNED_VERSION = "2.5.3"
SOURCE_COMMIT = "dd88c0c19b54ad9ed3533224221285bf0873249a"
NUMERIC_CODES = set("?bBhHiIlLqQefdgFDG")
DTYPES = [
    np.dtype(np.bool_), np.dtype(np.int8), np.dtype(np.uint8),
    np.dtype(np.int16), np.dtype(np.uint16), np.dtype(np.int32),
    np.dtype(np.uint32), np.dtype(np.int64), np.dtype(np.uint64),
    np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64),
    np.dtype(np.complex64), np.dtype(np.complex128),
    np.dtype(np.longdouble), np.dtype(np.clongdouble),
]
DTYPE_NAMES = [
    "bool", "int8", "uint8", "int16", "uint16", "int32", "uint32",
    "int64", "uint64", "float16", "float32", "float64", "complex64",
    "complex128", "longdouble", "clongdouble",
]


def rust_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def numeric_signatures(ufunc) -> list[str]:
    return [
        signature for signature in ufunc.types
        if all(code in NUMERIC_CODES for code in signature.replace("->", ""))
    ]


def dtype_index(dtype) -> int | None:
    kind, size, code = dtype.kind, dtype.itemsize, dtype.char
    if kind == "b":
        return 0
    if kind == "i":
        return {1: 1, 2: 3, 4: 5, 8: 7}.get(size)
    if kind == "u":
        return {1: 2, 2: 4, 4: 6, 8: 8}.get(size)
    if kind == "f":
        if size == 2:
            return 9
        if size == 4:
            return 10
        if code == "g":
            return 14
        if size == 8:
            return 11
    if kind == "c":
        if size == 8:
            return 12
        if code == "G":
            return 15
        if size == 16:
            return 13
    return None


def resolver_cell(ufunc, input_dtypes) -> str:
    try:
        resolved = ufunc.resolve_dtypes(
            tuple(input_dtypes) + (None,) * ufunc.nout,
            casting="same_kind",
        )
    except (TypeError, ValueError):
        return ""
    indexes = [dtype_index(dtype) for dtype in resolved]
    if any(index is None for index in indexes):
        return ""
    return "".join(format(index, "x") for index in indexes)


def generate_signatures(objects: dict[str, object]) -> str:
    lines = [
        "//! Numeric ufunc loop metadata extracted from NumPy 2.5.3 on macOS arm64.",
        "//! Non-numeric datetime, object, and string loops are intentionally omitted.",
        "//! The table is data only; Raptors executes the corresponding Rust scalar kernels.",
        "",
        "pub fn signatures(name: &str) -> Vec<String> {",
        "    let signatures: &[&str] = match name {",
    ]
    for name, ufunc in sorted(objects.items()):
        items = ", ".join(rust_string(value) for value in numeric_signatures(ufunc))
        lines.append(f"        {rust_string(name)} => &[{items}],")
    lines += [
        "        _ => &[],",
        "    };",
        "    signatures.iter().map(|signature| target_signature(signature)).collect()",
        "}",
        "",
        "fn target_signature(signature: &str) -> String {",
        "    if !cfg!(target_os = \"windows\") { return signature.to_owned(); }",
        "    signature.chars().map(|code| match code {",
        "        'i' => 'l', 'I' => 'L', 'l' => 'q', 'L' => 'Q', other => other,",
        "    }).collect()",
        "}",
    ]
    return "\n".join(lines) + "\n"


def generate_resolver(objects: dict[str, object]) -> str:
    lines = [
        "//! Complete numeric ufunc loop-resolution tables captured from NumPy 2.5.3 on macOS arm64.",
        "//! Each cell stores resolved input and output DType indexes; an empty cell is unsupported.",
        "//! DType indexes follow the order in raptors_storage::DType.",
        "",
        "use crate::DType;",
        "",
        "const DTYPES: [DType; 16] = [",
        "    DType::Bool, DType::Int8, DType::UInt8, DType::Int16, DType::UInt16,",
        "    DType::Int32, DType::UInt32, DType::Int64, DType::UInt64, DType::Float16,",
        "    DType::Float32, DType::Float64, DType::Complex64, DType::Complex128,",
        "    DType::LongDouble, DType::ComplexLongDouble,",
        "];",
        "",
        "struct Entry { name: &'static str, nin: usize, nout: usize, cells: &'static str }",
        "static ENTRIES: &[Entry] = &[",
    ]
    for name, ufunc in sorted(objects.items()):
        input_pairs = [
            (left,) if ufunc.nin == 1 else (left, right)
            for left in DTYPES for right in ([None] if ufunc.nin == 1 else DTYPES)
        ]
        cells = ".".join(resolver_cell(ufunc, pair) for pair in input_pairs) + "."
        lines.append(
            f"    Entry {{ name: {rust_string(name)}, nin: {ufunc.nin}, "
            f"nout: {ufunc.nout}, cells: {rust_string(cells)} }},"
        )
    lines += [
        "];",
        "",
        "pub fn resolve(name: &str, inputs: &[DType]) -> Option<Vec<DType>> {",
        "    let entry = ENTRIES.iter().find(|entry| entry.name == name)?;",
        "    if inputs.len() != entry.nin { return None; }",
        "    let left = dtype_index(inputs[0])?;",
        "    let cell_index = if entry.nin == 1 { left } else {",
        "        left * 16 + dtype_index(inputs[1])?",
        "    };",
        "    let cell = entry.cells.split('.').nth(cell_index)?;",
        "    if cell.len() != entry.nin + entry.nout { return None; }",
        "    cell.bytes().map(|code| decode(code).map(|index| DTYPES[index])).collect()",
        "}",
        "",
        "fn dtype_index(dtype: DType) -> Option<usize> {",
        "    DTYPES.iter().position(|candidate| *candidate == dtype)",
        "}",
        "",
        "fn decode(code: u8) -> Option<usize> {",
        "    match code {",
        "        b'0'..=b'9' => Some((code - b'0') as usize),",
        "        b'a'..=b'f' => Some((code - b'a' + 10) as usize),",
        "        _ => None,",
        "    }",
        "}",
    ]
    return "\n".join(lines) + "\n"


def identity_value(value):
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return "-inf" if value < 0 else "inf" if value > 0 else "nan"
    return value


def main() -> None:
    if np.__version__ != PINNED_VERSION:
        raise SystemExit(f"expected NumPy {PINNED_VERSION}, found {np.__version__}")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("generate pinned ufunc tables on macOS arm64")
    inventory = json.loads((ROOT / "compat/numpy-api-2.5.3.json").read_text())
    entries = [
        entry for entry in inventory["entries"]
        if entry.get("kind") == "ufunc" and entry.get("target_release") == "0.3"
    ]
    if len(entries) != 101:
        raise SystemExit(f"expected 101 phase 0.3 ufunc names, found {len(entries)}")
    contract_path = ROOT / "compat/raptors-0.3.json"
    previous_contract = json.loads(contract_path.read_text()) if contract_path.exists() else {}
    objects: dict[str, object] = {}
    public = []
    by_object: defaultdict[str, list[str]] = defaultdict(list)
    for entry in entries:
        name = entry["name"].removeprefix("numpy.")
        ufunc = getattr(np, name)
        objects.setdefault(ufunc.__name__, ufunc)
        by_object[ufunc.__name__].append(name)
        public.append({
            "name": name,
            "object_name": ufunc.__name__,
            "signature": entry.get("signature"),
        })

    signatures_path = ROOT / "raptors-storage/src/ufunc_signatures.rs"
    resolver_path = ROOT / "raptors-storage/src/ufunc_loop_resolver.rs"
    signatures_path.write_text(generate_signatures(objects))
    resolver_path.write_text(generate_resolver(objects))
    subprocess.run(
        ["rustfmt", "--edition", "2021", str(signatures_path), str(resolver_path)],
        cwd=ROOT,
        check=True,
    )

    object_contract = []
    for name, ufunc in sorted(objects.items()):
        object_contract.append({
            "name": name,
            "public_names": sorted(by_object[name]),
            "nin": ufunc.nin,
            "nout": ufunc.nout,
            "nargs": ufunc.nargs,
            "identity": identity_value(ufunc.identity),
            "signature": ufunc.signature,
            "numeric_types": numeric_signatures(ufunc),
        })

    contract = {
        "schema_version": 1,
        "release": "0.3",
        "package_version": "0.3.0",
        "status": previous_contract.get("status", "implementation_in_progress"),
        "reference": {
            "distribution": "numpy",
            "version": PINNED_VERSION,
            "source_tag": "v2.5.3",
            "source_commit": SOURCE_COMMIT,
            "metadata_platform": "macOS arm64",
        },
        "scope": {
            "module": "raptors",
            "top_level_public_ufunc_names": sorted(public, key=lambda entry: entry["name"]),
            "public_name_count": len(public),
            "canonical_object_count": len(objects),
            "ufunc_objects": object_contract,
            "dtype_kinds": ["b", "i", "u", "f", "c"],
            "call_controls": ["out", "where", "dtype", "casting", "order", "subok", "signature", "sig"],
            "methods": ["reduce", "accumulate", "reduceat", "outer", "at"],
            "operators": ["arithmetic", "comparisons", "bitwise", "reflected", "in-place", "divmod"],
            "floating_error_api": ["geterr", "seterr", "errstate", "geterrcall", "seterrcall"],
            "runtime_numpy_dependency": False,
        },
        "known_gaps": [
            "Longdouble and clongdouble scalar kernels currently compute through binary64 conversions on platforms where those formats are wider.",
            "Fresh outputs with false where positions are zero-initialized internally; those positions remain outside the result-value contract.",
            "Complex inverse-trig differential cases cover branch cuts, signed zero, small imaginary values, and large finite values; exact bit-for-bit equality across platform libm implementations is not promised.",
            "Floating-error flag detection uses scalar result heuristics and still needs NumPy differential coverage for all error modes.",
            "Combined dtype/signature, casting, where, out, and order cases are covered for single- and multi-output ufuncs; the full invalid-argument precedence matrix remains pending.",
            "Raptors has no ndarray subclasses, so subok does not alter results; third-party dispatch remains outside the 0.3 input domain.",
        ],
        "evidence": previous_contract.get("evidence", {
            "generation_platform": f"{platform.system()} {platform.machine()}",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cargo_check": {"status": "not_run", "command": "cargo check --offline -p raptors-python"},
            "differential_cases": {"status": "not_run", "passed": 0, "skipped": 0},
            "release_gate": "pending",
        }),
    }
    contract_path.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n"
    )
    print("Generated 0.3 ufunc signatures, loop resolver, and contract draft.")


if __name__ == "__main__":
    main()
