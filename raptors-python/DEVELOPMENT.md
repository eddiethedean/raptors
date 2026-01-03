# Development Guide for Raptors Python

This guide explains how to set up and work on the Raptors Python bindings.

## Development Setup

### Prerequisites

- Rust toolchain (install from https://rustup.rs/)
- Python 3.7 or later
- maturin (`pip install maturin` or `cargo install maturin`)
- pytest (for running Python tests)

### Initial Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/raptors.git
   cd raptors
   ```

2. Set up the test environment (required for running Rust tests):
   ```bash
   cd raptors-python
   ./setup_test_env.sh
   ```
   This script configures the Python library paths needed for linking tests.

3. Install the package in editable mode:
   ```bash
   maturin develop
   ```

4. Install development dependencies:
   ```bash
   pip install -e .[dev]
   ```

## Project Structure

```
raptors-python/
├── Cargo.toml          # Rust crate configuration
├── pyproject.toml      # Python package configuration
├── src/                # Rust source code
│   ├── lib.rs         # Module entry point
│   ├── array.rs       # Array bindings
│   ├── dtype.rs       # DType bindings
│   ├── ufunc.rs       # Ufunc bindings
│   ├── iterators.rs   # Iterator bindings
│   └── numpy_interop.rs  # NumPy interop
├── tests/              # Test files
│   ├── *.rs          # Rust unit tests
│   └── test_*.py     # Python pytest tests
├── examples/          # Example scripts
└── README.md         # Package documentation
```

## Running Tests

### Quick Start

Run all tests (recommended):

```bash
# Using the test runner script
./run_tests.sh

# Or using Make
make test
```

### Rust Tests

Run Rust unit tests:

```bash
cargo test --lib
```

Run specific test:

```bash
cargo test test_array_creation
```

Run with output:

```bash
cargo test --lib -- --nocapture
```

### Python Tests

**Important**: Ensure the module is built before running Python tests:

```bash
# Build the module first (if not already built)
maturin develop
```

Then run pytest tests:

```bash
pytest tests/
```

Run specific test file:

```bash
pytest tests/test_array.py
```

Run with verbose output:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=raptors
```

### All Tests

Run both Rust and Python tests:

```bash
# Using the test runner script
./run_tests.sh

# Or manually
cargo test --lib && pytest tests/
```

## Development Workflow

### Making Changes

1. Make changes to Rust code in `src/`
2. Rebuild automatically (if using `maturin develop`):
   ```bash
   maturin develop
   ```
3. Test your changes:
   ```bash
   cargo test
   pytest tests/
   ```

### Testing Changes

1. Build in release mode for performance testing:
   ```bash
   maturin develop --release
   ```

2. Test in Python:
   ```python
   import raptors
   # Test your changes
   ```

3. Run examples:
   ```bash
   python examples/basic_usage.py
   ```

## Code Style

### Rust

- Use `rustfmt` for formatting:
  ```bash
  cargo fmt
  ```

- Use `clippy` for linting:
  ```bash
  cargo clippy
  ```

### Python

- Follow PEP 8 style guide
- Use `black` for formatting (if configured)
- Use `pylint` or `flake8` for linting (if configured)

## Debugging

### Rust Debugging

1. Build with debug symbols:
   ```bash
   maturin develop
   ```

2. Use `println!` or `dbg!` for debugging
3. Use a Rust debugger (gdb, lldb)

### Python Debugging

1. Use Python debugger:
   ```python
   import pdb; pdb.set_trace()
   ```

2. Use IDE debugger (VS Code, PyCharm, etc.)

### Common Issues

**Import errors:**
- Rebuild: `maturin develop`
- Check Python path
- Verify installation: `pip show raptors`

**Build errors:**
- Update Rust: `rustup update`
- Update maturin: `pip install --upgrade maturin`
- Clean build: `cargo clean && maturin develop`

**Test failures:**
- Check Python version compatibility
- Verify dependencies are installed
- Check test environment setup

**Linker errors when running tests:**
If you see linker errors like "symbol(s) not found for architecture", you need to configure the Python library path:
```bash
# Run the setup script (recommended)
./setup_test_env.sh

# Or manually create .cargo/config.toml with Python library paths
# The setup script will do this automatically
```

## Building Locally

### Development Build

```bash
maturin develop
```

### Release Build

```bash
maturin develop --release
```

### Build Wheel

```bash
maturin build --release
```

See [BUILD.md](BUILD.md) for more details.

## Testing PyPI Uploads Locally

1. Build the package:
   ```bash
   maturin build --release
   ```

2. Test installation from local wheel:
   ```bash
   pip install target/wheels/raptors-*.whl
   ```

3. Test in clean environment:
   ```bash
   python -m venv test_env
   source test_env/bin/activate
   pip install target/wheels/raptors-*.whl
   python -c "import raptors; print(raptors.__version__)"
   deactivate
   rm -rf test_env
   ```

## Adding New Features

### Adding a New Python Binding

1. Add Rust code in `src/`:
   ```rust
   #[pyfunction]
   fn my_new_function() -> PyResult<PyObject> {
       // Implementation
   }
   ```

2. Register in `src/lib.rs`:
   ```rust
   m.add_function(wrap_pyfunction!(my_new_function, m)?)?;
   ```

3. Add tests in `tests/`
4. Update documentation

### Adding Tests

1. Add Rust test in `tests/*_test.rs`
2. Add Python test in `tests/test_*.py`
3. Run tests to verify

## Version Management

When updating the version:

1. Update `Cargo.toml` (workspace root)
2. Update `raptors-python/Cargo.toml`
3. Update `raptors-python/pyproject.toml`
4. Update changelog (if maintained)

## Documentation

- Update `README.md` for user-facing changes
- Update docstrings in Rust code (they appear in Python)
- Update examples if API changes

## Getting Help

- Check [BUILD.md](BUILD.md) for build issues
- Check main [README.md](../README.md) for project overview
- Check [CONTRIBUTING.md](../docs/CONTRIBUTING.md) for contribution guidelines
- Open an issue on GitHub

