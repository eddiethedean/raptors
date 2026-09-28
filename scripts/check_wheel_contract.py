#!/usr/bin/env python3
"""Fail if a built wheel contradicts the declared Raptors package contract."""
import argparse
from email.parser import Parser
from pathlib import Path
import tomllib
from zipfile import ZipFile

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import parse_wheel_filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--python", required=True, help="minimum CPython minor in the abi3 tag, e.g. 3.12")
    parser.add_argument("--platform-family", required=True, choices=("manylinux", "musllinux", "macosx", "win"))
    parser.add_argument("--platform-fragment", required=True, help="expected architecture fragment in the wheel platform tag")
    args = parser.parse_args()
    distribution, version, build, tags = parse_wheel_filename(args.wheel.name)
    assert str(distribution) == "raptors", distribution
    project = tomllib.loads(Path("raptors-python/pyproject.toml").read_text())["project"]
    assert str(version) == project["version"], (version, project["version"])
    expected = "cp" + "".join(args.python.split("."))
    assert any(tag.interpreter == expected and tag.abi == "abi3" for tag in tags), tags
    assert any(tag.platform.startswith(args.platform_family) and args.platform_fragment in tag.platform for tag in tags), tags
    with ZipFile(args.wheel) as wheel:
        metadata_name = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(wheel.read(metadata_name).decode())
        names = set(wheel.namelist())
    assert SpecifierSet(metadata["Requires-Python"]) == SpecifierSet(project["requires-python"]), metadata["Requires-Python"]
    runtime_requirements = []
    for requirement_text in metadata.get_all("Requires-Dist", []):
        requirement = Requirement(requirement_text)
        if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
            runtime_requirements.append(requirement_text)
    assert not runtime_requirements, runtime_requirements
    assert any(name.endswith(".dist-info/licenses/LICENSE") or name.endswith(".dist-info/LICENSE") for name in names), "MIT license file missing from wheel"
    assert any(name.startswith("raptors/") and name.endswith((".so", ".pyd")) for name in names), "native extension missing"
    print(f"Validated {args.wheel.name}: CPython {args.python}+ abi3, {args.platform_family}/{args.platform_fragment}, no runtime dependencies, license included")


if __name__ == "__main__":
    main()
