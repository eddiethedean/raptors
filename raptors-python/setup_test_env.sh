#!/bin/bash
# Setup script for Python test environment
# This script configures the environment to run Python linking tests

set -e

# Get Python executable
if [ -z "$PYO3_PYTHON" ]; then
    PYO3_PYTHON=$(which python3)
fi

echo "Using Python: $PYO3_PYTHON"

# Get Python library directory
LIBDIR=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
echo "Python library directory: $LIBDIR"

# Get Python version for library name
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VERSION"

# Get the workspace root (parent directory)
WORKSPACE_ROOT=$(cd "$(dirname "$0")/.." && pwd)

# Create .cargo directory if it doesn't exist
mkdir -p "$WORKSPACE_ROOT/.cargo"

# Generate config.toml in workspace root
cat > "$WORKSPACE_ROOT/.cargo/config.toml" << EOF
[env]
PYO3_PYTHON = "$PYO3_PYTHON"

[target.'cfg(target_os = "macos")']
rustflags = [
    "-L", "$LIBDIR",
    "-l", "python$PYTHON_VERSION",
]
EOF

echo "Created $WORKSPACE_ROOT/.cargo/config.toml"
echo ""
echo "You can now run Python tests with:"
echo "  cargo test --package raptors-python"

