#!/usr/bin/env python3
"""Install a wheel in a fresh environment and verify it imports without NumPy."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--python", default=sys.executable, help="interpreter to use for the clean environment")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="raptors-no-numpy-") as temporary:
        environment = Path(temporary) / "venv"
        uv = shutil.which("uv")
        if uv is None:
            raise SystemExit("uv must be available to create the isolated no-dependency environment")
        subprocess.run([uv, "venv", "--python", args.python, str(environment)], check=True)
        executable = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run([uv, "pip", "install", "--python", str(executable), "--no-deps", str(args.wheel.resolve())], check=True)
        check = (
            "import importlib.util, raptors; "
            "assert importlib.util.find_spec('numpy') is None; "
            "a=raptors.array([[1, 2], [3, 4]], dtype=raptors.int64); "
            "v=a[::-1, 1]; v[0]=9; "
            "assert int(a[0, 1]) == 2 and int(a[1, 1]) == 9; "
            "print(raptors.__file__)"
        )
        subprocess.run([str(executable), "-I", "-c", check], check=True)


if __name__ == "__main__":
    main()
