"""Small differential comparator shared by the preview oracle cases."""
import numpy as np

SCALAR_NAMES = {
    np.bool_: "BoolScalar",
    np.int64: "Int64Scalar",
    np.uint64: "UInt64Scalar",
    np.float32: "Float32Scalar",
    np.float64: "Float64Scalar",
}


def _coordinates(shape):
    if not shape:
        return [()]
    return np.ndindex(shape)


def _python_value(value):
    # Extended scalar wrappers cannot return a built-in value from ``item``.
    # Preserve long-double precision by parsing its decimal representation;
    # complex-long-double ``__complex__`` is sufficient for the exactly
    # representable values used by the broad dtype/cast matrix. The dedicated
    # long-double case checks extended integer precision where the platform
    # exposes it.
    scalar_name = type(value).__name__
    if scalar_name == "LongDoubleScalar":
        return np.longdouble(str(value))
    if scalar_name == "ComplexLongDoubleScalar":
        return complex(value)
    return value.item() if hasattr(value, "item") else value


def assert_array_matches(reference, candidate, *, values=True):
    """Compare type, layout, scalar classes, values, and signed zero exactly."""
    for attribute in (
        "name",
        "kind",
        "char",
        "itemsize",
        "alignment",
        "byteorder",
        "isnative",
        "str",
    ):
        expected = getattr(reference.dtype, attribute)
        actual = getattr(candidate.dtype, attribute)
        assert actual == expected, f"dtype mismatch: {attribute}: {actual!r} != {expected!r}"
    assert tuple(candidate.shape) == tuple(reference.shape), "shape mismatch"
    assert tuple(candidate.strides) == tuple(reference.strides), "stride mismatch"
    for attribute in ("c_contiguous", "f_contiguous", "writeable"):
        assert getattr(candidate.flags, attribute) == getattr(reference.flags, attribute), (
            f"array {attribute} flag mismatch"
        )
    if not values:
        return
    for coordinates in _coordinates(reference.shape):
        coordinates = tuple(int(i) for i in coordinates)
        try:
            expected = reference[coordinates]
            actual = candidate[coordinates]
        except Exception as exc:
            raise AssertionError(f"indexing failed at {coordinates}: {exc}") from exc
        expected_type = SCALAR_NAMES.get(type(expected))
        if expected_type is not None:
            assert type(actual).__name__ == expected_type, (
                f"scalar type mismatch at {coordinates}: {type(actual).__name__} != {expected_type}"
            )
        expected_value = _python_value(expected)
        actual_value = _python_value(actual)
        if reference.dtype.kind == "f":
            if np.isnan(expected_value):
                assert np.isnan(actual_value), f"expected NaN at {coordinates}"
            else:
                assert actual_value == expected_value, f"value mismatch at {coordinates}"
                if expected_value == 0:
                    assert bool(np.signbit(actual_value)) == bool(np.signbit(expected_value)), (
                        f"signed zero mismatch at {coordinates}"
                    )
        else:
            assert actual_value == expected_value, f"value mismatch at {coordinates}"


def assert_candidate_observable(candidate):
    """Turn crashes, expired owners, and access failures into a harness failure."""
    shape = tuple(candidate.shape)
    try:
        for coordinates in _coordinates(shape):
            candidate[tuple(int(i) for i in coordinates)]
    except Exception as exc:
        raise AssertionError(f"candidate data access failed: {exc}") from exc
