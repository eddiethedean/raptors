#!/usr/bin/env python3
"""Fail if a built wheel contradicts the declared Raptors 0.1 contract."""
import argparse
from email.parser import Parser
from pathlib import Path
from zipfile import ZipFile

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import parse_wheel_filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--python", required=True, help="expected interpreter minor, e.g. 3.14")
    args = parser.parse_args()
    distribution, version, build, tags = parse_wheel_filename(args.wheel.name)
    assert str(distribution) == "raptors", distribution
    assert str(version) == "0.1.0", version
    expected = "cp" + "".join(args.python.split("."))
    assert any(tag.interpreter == expected and tag.abi == expected for tag in tags), tags
    with ZipFile(args.wheel) as wheel:
        metadata_name = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(wheel.read(metadata_name).decode())
        names = set(wheel.namelist())
    assert SpecifierSet(metadata["Requires-Python"]) == SpecifierSet(">=3.12,<3.15"), metadata["Requires-Python"]
    runtime_requirements = []
    for requirement_text in metadata.get_all("Requires-Dist", []):
        requirement = Requirement(requirement_text)
        if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
            runtime_requirements.append(requirement_text)
    assert not runtime_requirements, runtime_requirements
    assert any(name.endswith(".dist-info/licenses/LICENSE") or name.endswith(".dist-info/LICENSE") for name in names), "MIT license file missing from wheel"
    assert any(name.startswith("raptors/") and name.endswith(".so") or name.startswith("raptors/") and name.endswith(".pyd") for name in names), "native extension missing"
    print(f"Validated {args.wheel.name}: CPython {args.python}, no runtime dependencies, license included")


if __name__ == "__main__":
    main()
