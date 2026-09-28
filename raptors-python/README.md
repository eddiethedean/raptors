# Raptors Python

Python bindings for the experimental Raptors array engine. The project is rebuilding toward NumPy's public functionality through:

```python
import raptors as np
```

**That is the compatibility goal, not a claim that the current package is a drop-in replacement.** The [rebuild plan](../docs/REBUILD_PLAN.md) defines the design and acceptance gates.

## Current package

The extension uses PyO3 and delegates array work to the local `raptors-core` crate. It exposes array/dtype classes, constructors, selected operations and reductions, iteration, and NumPy conversion helpers. See the [API guide](../docs/API_GUIDE.md) for source locations and limitations.

Known issues include unsafe narrow-dtype list construction and unproven view/concurrency contracts. Existing modules need differential validation before they count as compatible.

The current Python metadata depends on NumPy. The rebuild targets native numeric execution without NumPy fallback, with NumPy serving as the reference oracle and optional interoperability dependency. That dependency change has not yet been implemented.

## Development

Follow [DEVELOPMENT.md](DEVELOPMENT.md) to create an isolated environment and explicitly rebuild the extension. Use [BUILD.md](BUILD.md) for wheel validation and [TESTING.md](TESTING.md) for the existing test commands and planned gates.

This documentation does not assert that a public PyPI release is available or ready. The historical Python version classifiers and publishing matrix have not been validated as the rebuilt support policy.

## Compatibility and safety

The final scope includes NumPy's public dtypes, functions, methods, ufunc behavior, and submodules. Earlier releases must state their supported subset. Compiled extensions requiring NumPy objects need explicit adapters; NumPy binary-interface replacement is outside the release requirement.

Memory safety depends on checked storage and controlled mutation, including aliases and foreign buffers. Performance claims require equivalent Python workloads, including allocation and conversion costs.

## Documentation

- [Project overview](../README.md)
- [Documentation index](../docs/README.md)
- [Architecture proposal](../docs/ARCHITECTURE.md)
- [Contribution guide](../docs/CONTRIBUTING.md)
- [Verification record](../docs/NUMPY_TEST_VERIFICATION.md)

The metadata currently declares MIT; a top-level license file and consistent declarations are still needed before release.
