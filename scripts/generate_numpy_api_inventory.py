#!/usr/bin/env python3
"""Generate the public NumPy 2.5.3 API inventory from the pinned install."""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import pkgutil
import re
from pathlib import Path

import numpy as np

EXPECTED_VERSION = "2.5.3"
PREVIEW_NAMES = {"array", "zeros", "empty"}
NUMERIC_TYPES = {
    "ndarray", "dtype", "generic", "number", "integer", "signedinteger",
    "unsignedinteger", "inexact", "floating", "complexfloating", "bool_",
    "int8", "int16", "int32", "int64", "intp", "uint8", "uint16",
    "uint32", "uint64", "uintp", "float16", "float32", "float64",
    "longdouble", "complex64", "complex128", "clongdouble", "byte",
    "ubyte", "short", "ushort", "intc", "uintc", "long", "ulong",
    "longlong", "ulonglong", "single", "double", "half", "csingle",
    "cdouble", "int_", "uint", "bool",
}
SPECIALIZED_TYPES = {
    "datetime64", "timedelta64", "str_", "bytes_", "void", "object_",
    "character", "flexible", "record", "recarray",
}
NUMERIC_OPERATOR_METHODS = {
    "__add__", "__sub__", "__mul__", "__matmul__", "__truediv__",
    "__floordiv__", "__mod__", "__divmod__", "__pow__", "__neg__",
    "__pos__", "__abs__", "__invert__", "__and__", "__or__", "__xor__",
    "__lshift__", "__rshift__", "__lt__", "__le__", "__eq__", "__ne__",
    "__gt__", "__ge__",
}
REDUCTION_METHODS = {
    "sum", "prod", "min", "max", "mean", "std", "var", "argmin",
    "argmax", "any", "all", "cumsum", "cumprod", "reduce", "accumulate",
    "reduceat", "outer", "at",
}
IO_FUNCTIONS = {
    "from_dlpack", "frombuffer", "fromfile", "fromiter", "fromstring",
    "genfromtxt", "load", "loadtxt", "save", "savetxt", "savez",
    "savez_compressed",
}
SCALAR_PROTOCOLS = {
    "__array__", "__array_function__", "__array_ufunc__", "__bool__", "__bytes__",
    "__eq__", "__getitem__", "__hash__", "__iter__", "__len__", "__lt__",
    "__matmul__", "__ne__", "__repr__", "__setitem__", "__str__",
} | NUMERIC_OPERATOR_METHODS

NUMERIC_FOUNDATION_ARRAY_MEMBERS = {
    "__getitem__", "__setitem__", "__len__", "shape", "ndim", "size",
    "strides", "itemsize", "nbytes", "flags", "dtype", "item", "copy",
    "reshape", "transpose", "T", "astype",
}
NUMERIC_FOUNDATION_DTYPE_MEMBERS = {
    "name", "kind", "char", "itemsize", "alignment", "byteorder",
    "isnative", "str", "type", "newbyteorder",
}
NUMERIC_FOUNDATION_SCALAR_MEMBERS = {
    "dtype", "item", "__bool__", "__int__", "__float__", "__complex__",
    "__index__",
}
ARRAY_INTEROP_MEMBERS = {
    "__array__", "__array_interface__", "__array_struct__", "data", "base",
    "ctypes", "from_dlpack", "register_dlpack_dtype",
}
NUMERIC_DTYPE_DESCRIPTORS = {
    "kind", "name", "char", "itemsize", "alignment", "byteorder", "isnative",
    "str", "type", "newbyteorder", "__eq__", "__ne__", "__repr__", "__str__",
}
SPECIALIZED_DTYPE_CLASSES = {
    "StringDType", "DateTime64DType", "TimeDelta64DType", "ObjectDType",
    "BytesDType", "StrDType", "VoidDType",
}
GENERALIZED_NUMERIC_UFUNCS = {"matmul", "matvec", "vecdot", "vecmat"}
UFUNC_ERROR_STATE_API = {"errstate", "geterr", "seterr", "geterrcall", "seterrcall"}


def in_module(path, module):
    return path == module or path.startswith(module + ".")


def api_kind(value):
    if inspect.ismodule(value):
        return "module"
    if isinstance(value, np.ufunc):
        return "ufunc"
    if inspect.isclass(value):
        return "class"
    if inspect.isroutine(value) or callable(value):
        return "callable"
    return "attribute"


def target_release(path, value, module_name):
    name = path.rsplit(".", 1)[-1]
    parent = path.rsplit(".", 1)[0] if "." in path else ""
    if in_module(path, "numpy.matlib"):
        return "0.8"
    if in_module(path, "numpy.matrix"):
        return "0.8"
    if path == "numpy.ufunc" or (module_name == "numpy" and name in UFUNC_ERROR_STATE_API):
        return "0.3"
    if module_name == "numpy" and len(path.split(".")) <= 3 and path.split(".")[1] in GENERALIZED_NUMERIC_UFUNCS:
        return "0.5"
    if parent == "numpy.ndarray" and name == "__matmul__":
        return "0.5"
    if name in PREVIEW_NAMES and module_name == "numpy":
        return "0.1"
    if name in {"__array_function__", "__array_ufunc__", "__array_namespace__", "__array_namespace_info__", "__array_priority__"}:
        return "0.8"
    if name in ARRAY_INTEROP_MEMBERS:
        return "0.6"
    path_parts = path.split(".")
    if len(path_parts) >= 3 and path_parts[0] == "numpy":
        ufunc = getattr(np, path_parts[1], None)
        if isinstance(ufunc, np.ufunc):
            return "0.3"
    if parent == "numpy.ndarray":
        if name in NUMERIC_OPERATOR_METHODS or name in REDUCTION_METHODS:
            return "0.4" if name in REDUCTION_METHODS else "0.3"
        if name in NUMERIC_FOUNDATION_ARRAY_MEMBERS:
            return "0.2"
        return "0.4"
    if parent == "numpy.dtype":
        if name in NUMERIC_FOUNDATION_DTYPE_MEMBERS:
            return "0.2"
        if name in {"fields", "names", "subdtype", "hasobject", "isalignedstruct"}:
            return "0.7"
        return "0.8"
    if parent == "numpy.ufunc":
        return "0.3"
    if parent == "numpy.generic":
        if name in NUMERIC_OPERATOR_METHODS:
            return "0.3"
        return "0.2" if name in NUMERIC_FOUNDATION_SCALAR_MEMBERS else "0.8"
    if parent.startswith("numpy.") and parent.rsplit(".", 1)[-1] in NUMERIC_TYPES:
        if name in NUMERIC_OPERATOR_METHODS:
            return "0.3"
        return "0.2" if name in NUMERIC_FOUNDATION_SCALAR_MEMBERS else "0.8"
    if parent.startswith("numpy.") and parent.rsplit(".", 1)[-1] in SPECIALIZED_TYPES:
        return "0.7"
    if any(in_module(path, module) for module in ("numpy.linalg", "numpy.fft", "numpy.polynomial", "numpy.random")):
        return "0.5"
    if any(in_module(path, module) for module in ("numpy.char", "numpy.strings", "numpy.ma", "numpy.rec")):
        return "0.7"
    if any(in_module(path, module) for module in ("numpy.lib.format", "numpy.lib.npyio", "numpy.lib._iotools", "numpy.ctypeslib", "numpy.memmap")):
        return "0.6"
    if path.startswith("numpy.testing") or path.startswith("numpy.f2py") or path == "numpy.test":
        return "0.8"
    if name in SPECIALIZED_TYPES or name in {"datetime_as_string", "datetime_data", "isnat"}:
        return "0.7"
    if module_name == "numpy.dtypes":
        parent = path.rsplit(".", 1)[0] if "." in path else ""
        if parent == "numpy.dtypes":
            return "0.7" if name in SPECIALIZED_DTYPE_CLASSES else "0.2"
        if parent.startswith("numpy.dtypes."):
            dtype_class = parent.rsplit(".", 1)[-1]
            if name in NUMERIC_DTYPE_DESCRIPTORS:
                return "0.7" if dtype_class in SPECIALIZED_DTYPE_CLASSES else "0.2"
            return "0.7" if dtype_class in SPECIALIZED_DTYPE_CLASSES else "0.8"
    if name in IO_FUNCTIONS:
        return "0.6"
    if isinstance(value, np.ufunc) or "umath" in (getattr(value, "__module__", "") or ""):
        return "0.3"
    if path == f"numpy.{name}" and name in NUMERIC_TYPES:
        return "0.2"
    if path.startswith("numpy.dtypes."):
        return "0.7"
    return "0.4"


def signature_of(value):
    try:
        signature = str(inspect.signature(value))
        return re.sub(r"0x[0-9a-fA-F]+", "<address>", signature)
    except (TypeError, ValueError):
        return None


def required_case_plan(path, kind, module_name):
    if path in {"numpy.array", "numpy.zeros", "numpy.empty"}:
        return ["tests/preview/test_differential.py", "tests/preview/test_properties.py"]
    if kind == "module":
        return ["importability", "documented_public_exports", "versioned_export_changes"]
    if kind == "attribute":
        return ["value_and_type", "immutability_or_mutation_contract", "versioned_export_changes"]
    if kind == "class":
        return ["constructor_and_defaults", "properties_and_scalar_behavior", "errors_and_boundaries", "ownership_and_interoperation"]
    if kind == "ufunc":
        return ["signature_and_defaults", "values_and_dtypes", "broadcasting_and_strides", "out_where_and_methods", "errors_and_warnings"]
    return ["signature_and_defaults", "values_and_result_type", "shape_and_dtype", "errors_and_warnings", "mutation_when_documented"]


def class_is_defined_or_exported_here(path, value, module_name):
    if not inspect.isclass(value):
        return False
    owner = getattr(value, "__module__", "") or ""
    if module_name == "numpy":
        return owner == "numpy" or path in {"numpy.ndarray", "numpy.dtype", "numpy.generic", "numpy.ufunc"}
    return owner == module_name or owner.startswith(module_name + ".")


def entry(path, value, module_name):
    name = path.rsplit(".", 1)[-1]
    is_preview = module_name == "numpy" and name in PREVIEW_NAMES
    return {
        "name": path,
        "kind": api_kind(value),
        "origin": getattr(value, "__module__", None) or module_name,
        "signature": signature_of(value) if callable(value) else None,
        "status": "partial" if is_preview else "not_implemented",
        "target_release": target_release(path, value, module_name),
        "required_cases": ["test_differential.py"] if is_preview else [],
        "required_case_plan": required_case_plan(path, api_kind(value), module_name),
        "case_plan_status": "authored" if is_preview else "planned_not_authored",
        "known_limits": ["Preview scope is recorded in compat/raptors-0.1.json"] if is_preview else ["Semantic cases and limits require review"],
    }


def is_public_module(name):
    parts = name.split(".")
    return all(not part.startswith("_") and part not in {"tests", "conftest"} for part in parts)


def build_inventory():
    if np.__version__ != EXPECTED_VERSION:
        raise SystemExit(f"expected NumPy {EXPECTED_VERSION}, found {np.__version__}")
    modules = {"numpy": np}
    import_failures = []
    for info in pkgutil.walk_packages(np.__path__, prefix="numpy."):
        if not is_public_module(info.name):
            continue
        try:
            modules[info.name] = importlib.import_module(info.name)
        except Exception as error:  # retained as inventory evidence rather than silently omitted
            import_failures.append({"module": info.name, "error": f"{type(error).__name__}: {error}"})

    inventory = {}
    for module_name, module in sorted(modules.items()):
        if module_name == "numpy":
            names = set(getattr(module, "__all__", ()))
            names.update(name for name in dir(module) if not name.startswith("_"))
        else:
            names = {name for name in dir(module) if not name.startswith("_")}
        for name in sorted(names):
            try:
                value = getattr(module, name)
            except Exception as error:
                import_failures.append({"name": f"{module_name}.{name}", "error": f"{type(error).__name__}: {error}"})
                continue
            path = f"{module_name}.{name}"
            inventory[path] = entry(path, value, module_name)
            if isinstance(value, np.ufunc) and (
                module_name == "numpy"
                or (getattr(value, "__module__", "") or "").startswith(module_name + ".")
                or getattr(value, "__module__", "") == module_name
            ):
                for member_name in ("reduce", "accumulate", "reduceat", "outer", "at"):
                    member_path = f"{path}.{member_name}"
                    member = getattr(value, member_name, None)
                    if member is not None:
                        inventory[member_path] = entry(member_path, member, module_name)
            if class_is_defined_or_exported_here(path, value, module_name):
                for member_name in dir(value):
                    is_protocol = member_name in SCALAR_PROTOCOLS
                    if member_name.startswith("_") and not is_protocol:
                        continue
                    member_path = f"{path}.{member_name}"
                    try:
                        member = getattr(value, member_name)
                    except Exception as error:
                        import_failures.append({"name": member_path, "error": f"{type(error).__name__}: {error}"})
                        continue
                    inventory[member_path] = entry(member_path, member, module_name)

    return {
        "schema_version": 1,
        "reference": {
            "distribution": "numpy",
            "version": np.__version__,
            "source_tag": "v2.5.3",
            "source_commit": "dd88c0c19b54ad9ed3533224221285bf0873249a",
            "python_lock": "raptors-python/uv.lock",
        },
        "generation": {
            "script": "scripts/generate_numpy_api_inventory.py",
            "method": "public top-level exports and importable non-private non-test package modules, their public members, and selected methods on classes defined or exported by that module",
            "import_failures": import_failures,
            "entry_count": len(inventory),
        },
        "entries": list(sorted(inventory.values(), key=lambda item: item["name"])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="compat/numpy-api-2.5.3.json")
    args = parser.parse_args()
    result = build_inventory()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {result['generation']['entry_count']} API inventory entries to {output}")


if __name__ == "__main__":
    main()
