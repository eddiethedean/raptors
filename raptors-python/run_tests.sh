#!/usr/bin/env bash
# Run the supported Python preview suite and every Rust workspace test.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if command -v uv >/dev/null 2>&1; then
    TEST_ENV_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/raptors-test-env.XXXXXX")"
    export UV_PROJECT_ENVIRONMENT="$TEST_ENV_ROOT/venv"
    trap 'rm -rf "$TEST_ENV_ROOT"' EXIT
    uv sync --project "$SCRIPT_DIR" --extra dev --locked --no-install-project
    uv run --project "$SCRIPT_DIR" --extra dev --no-sync \
        maturin develop --manifest-path "$SCRIPT_DIR/Cargo.toml" --release
    uv run --project "$SCRIPT_DIR" --extra dev --no-sync \
        python -m pytest "$SCRIPT_DIR/tests/preview" -v
else
    if ! python -c "import raptors" >/dev/null 2>&1; then
        if ! command -v maturin >/dev/null 2>&1; then
            echo "maturin is required to build the raptors extension" >&2
            exit 1
        fi
        (cd "$WORKSPACE_ROOT" && maturin develop --manifest-path "$SCRIPT_DIR/Cargo.toml" --release)
    fi
    python -m pytest "$SCRIPT_DIR/tests/preview" -v
fi

cargo test --offline --locked --manifest-path "$WORKSPACE_ROOT/Cargo.toml" \
    --workspace -- --test-threads=1
