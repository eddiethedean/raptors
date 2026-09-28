"""Fail closed when the native extension is missing or imports incorrectly."""
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

# Deliberately do not skip: a missing native build is a failed preview run.
import raptors  # noqa: E402,F401
