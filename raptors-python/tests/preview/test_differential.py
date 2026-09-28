import gc

import numpy as np
import pytest
import raptors

from harness import assert_array_matches


@pytest.mark.parametrize(
    "dtype, values",
    [
        (raptors.bool_, [False, True, 0, 3]),
        (raptors.int64, [-(2**53 + 1), 2**53 + 1, -1, 0]),
        (raptors.uint64, [0, 2**53 + 1, 2**64 - 1]),
        (raptors.float32, [1 / 3, -0.0, float("nan"), 1.25]),
        (raptors.float64, [1 / 3, -0.0, float("nan"), float("inf")]),
    ],
)
def test_explicit_dtype_values_and_scalar_types(dtype, values):
    name = dtype.name
    reference = np.array(values, dtype=name)
    candidate = raptors.array(values, dtype=dtype)
    assert_array_matches(reference, candidate)


@pytest.mark.parametrize("dtype", [raptors.bool_, raptors.int64, raptors.uint64, raptors.float32, raptors.float64])
def test_zero_dimensional_arrays(dtype):
    value = {"bool": True, "int64": -9, "uint64": 9, "float32": 0.25, "float64": 0.25}[dtype.name]
    reference = np.array(value, dtype=dtype.name)
    candidate = raptors.array(value, dtype=dtype)
    assert_array_matches(reference, candidate)


@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (1, 3), (2, 3)])
def test_zeros_and_empty_metadata(shape):
    reference = np.zeros(shape, dtype=np.float64)
    candidate = raptors.zeros(shape, dtype=raptors.float64)
    assert_array_matches(reference, candidate)
    assert_array_matches(reference, raptors.zeros(shape, dtype=None))

    uninitialized = raptors.empty(shape, dtype=raptors.float64)
    assert tuple(uninitialized.shape) == reference.shape
    assert tuple(uninitialized.strides) == reference.strides
    # The preview initializes `empty`; values remain unspecified by its contract.
    for index in range(uninitialized.size):
        value = uninitialized[np.unravel_index(index, shape)] if shape else uninitialized[()]
        assert value.dtype.name == "float64"


@pytest.mark.parametrize("values,dtype,error", [
    ([2**63], raptors.int64, OverflowError),
    ([-1], raptors.uint64, OverflowError),
    ([float("nan")], raptors.int64, ValueError),
    ([float("inf")], raptors.uint64, OverflowError),
    ([[1, 2], [3]], raptors.int64, ValueError),
])
def test_conversion_and_ragged_errors(values, dtype, error):
    with pytest.raises(error):
        raptors.array(values, dtype=dtype)
    with pytest.raises(error):
        np.array(values, dtype=dtype.name)


@pytest.mark.parametrize(
    "shape,error",
    [(-1, ValueError), ((2, -1), ValueError), (1.5, TypeError), (True, TypeError)],
)
def test_shape_errors_match_numpy(shape, error):
    with pytest.raises(error):
        np.zeros(shape)
    with pytest.raises(error):
        raptors.zeros(shape)


def test_shape_dimensions_reject_booleans():
    with pytest.raises(TypeError):
        np.zeros((True,))
    with pytest.raises(TypeError):
        raptors.zeros((True,))


def test_reshape_infers_one_dimension():
    reference = np.arange(24, dtype=np.int64)
    candidate = raptors.array(reference.tolist(), dtype=raptors.int64)
    assert_array_matches(reference.reshape((2, -1)), candidate.reshape((2, -1)))
    assert_array_matches(reference.reshape((-1, 3, 2)), candidate.reshape((-1, 3, 2)))
    assert raptors.zeros((0,), dtype=raptors.int64).reshape((-1,)).shape == (0,)

    for shape in ((2, -1, -1), (5, -1), (0, -1)):
        with pytest.raises(ValueError):
            reference.reshape(shape)
        with pytest.raises(ValueError):
            candidate.reshape(shape)


def test_array_copy_none_accepts_copy_if_needed_semantics():
    source = raptors.array([1, 2, 3], dtype=raptors.int64)
    reused = raptors.array(source, copy=None)
    assert_array_matches(np.array([1, 2, 3], dtype=np.int64), reused)

    copied_from_sequence = raptors.array([1, 2, 3], dtype=raptors.int64, copy=None)
    assert_array_matches(np.array([1, 2, 3], dtype=np.int64), copied_from_sequence)


def test_astype_preserves_fortran_layout_by_default():
    reference = np.asfortranarray(np.arange(12, dtype=np.int64).reshape(3, 4))
    candidate = raptors.array(reference.tolist(), dtype=raptors.int64, order="F")

    actual = candidate.astype("float32")
    expected = reference.astype(np.float32)
    assert_array_matches(expected, actual)
    assert actual.f_contiguous and not actual.c_contiguous

    c_order = candidate.astype("float32", order="C")
    assert c_order.c_contiguous and not c_order.f_contiguous


def test_longlong_dtype_alias_metadata_survives_arrays_and_scalars():
    for code, constant_name, scalar_name in (
        ("q", "longlong", "LongLongScalar"),
        ("Q", "ulonglong", "ULongLongScalar"),
    ):
        expected = np.dtype(code)
        descriptor = raptors.DType(code)
        assert descriptor.char == expected.char
        assert descriptor.name == expected.name
        assert getattr(raptors, constant_name).char == expected.char

        if expected.char != getattr(raptors, "int64" if code == "q" else "uint64").char:
            assert descriptor.type.__name__ == scalar_name

        value = -3 if code == "q" else 3
        candidate = raptors.array([value], dtype=code)
        assert candidate.dtype.char == expected.char
        if expected.char != getattr(raptors, "int64" if code == "q" else "uint64").char:
            assert type(candidate[0]).__name__ == scalar_name


def test_ellipsis_counts_nested_boolean_list_rank():
    values = np.arange(24, dtype=np.int64).reshape(2, 3, 2, 2)
    mask = [[True, False], [False, True]]
    candidate = raptors.array(values.tolist(), dtype=raptors.int64)
    expected = values[..., mask]
    actual = candidate[..., mask]
    assert tuple(actual.shape) == expected.shape
    for index in np.ndindex(expected.shape):
        assert actual[index] == expected[index]


def test_dtype_metadata_and_explicit_dtype_requirement():
    for dtype in (raptors.bool_, raptors.int64, raptors.uint64, raptors.float32, raptors.float64):
        assert dtype.name in {"bool", "int64", "uint64", "float32", "float64"}
        assert dtype.itemsize == np.dtype(dtype.name).itemsize
        assert dtype.kind == np.dtype(dtype.name).kind
    assert raptors.array([[1, 2], [3, 4]], dtype="int64").dtype == raptors.int64
    with pytest.raises(TypeError):
        raptors.array([1, 2])


def test_float32_overflow_warning_matches_reference():
    values = [1e100, 1e100]
    with pytest.warns(RuntimeWarning, match="overflow encountered in cast") as reference_warnings:
        reference = np.array(values, dtype=np.float32)
    with pytest.warns(RuntimeWarning, match="overflow encountered in cast") as candidate_warnings:
        candidate = raptors.array(values, dtype=raptors.float32)
    assert len(candidate_warnings) == len(reference_warnings) == 2
    assert_array_matches(reference, candidate)


@pytest.mark.parametrize("key", [slice(None, None, -1), slice(1, 5, 2), slice(5, 5, -1)])
def test_negative_stepped_and_empty_slices(key):
    values = np.arange(6, dtype=np.int64)
    reference = values[key]
    candidate = raptors.array(values.tolist(), dtype=raptors.int64)[key]
    assert_array_matches(reference, candidate)


def test_tuple_integer_and_slice_indexing():
    values = np.arange(24, dtype=np.float32).reshape(4, 6)
    reference = values[1:4:2, ::-2]
    candidate = raptors.array(values.tolist(), dtype=raptors.float32)[1:4:2, ::-2]
    assert_array_matches(reference, candidate)

    scalar = raptors.array(values.tolist(), dtype=raptors.float32)[-1, -2]
    expected = values[-1, -2]
    assert type(scalar).__name__ == "Float32Scalar"
    assert scalar == expected
    assert scalar.dtype.name == "float32"


def test_invalid_indices_match_numpy_errors():
    reference = np.arange(3, dtype=np.int64)
    candidate = raptors.array([0, 1, 2], dtype=raptors.int64)
    for key in (3, (0, 1), 1.5):
        with pytest.raises(IndexError) as reference_error:
            _ = reference[key]
        with pytest.raises(type(reference_error.value)):
            _ = candidate[key]
    with pytest.raises(ValueError):
        _ = reference[::0]
    with pytest.raises(ValueError):
        _ = candidate[::0]


def test_scalar_assignment_views_copy_and_overlapping_assignment():
    expected = np.arange(6, dtype=np.int64)
    actual = raptors.array(expected.tolist(), dtype=raptors.int64)
    expected[1:] = expected[:-1]
    actual[1:] = actual[:-1]
    assert_array_matches(expected, actual)

    expected[::-2] = 40
    actual[::-2] = 40
    assert_array_matches(expected, actual)

    expected[1:3] = np.array([8, 9], dtype=np.int64)
    actual[1:3] = raptors.array([8, 9], dtype=raptors.int64)
    assert_array_matches(expected, actual)

    copied = actual.copy()
    copied[0] = -99
    assert actual[0] == 0
    assert copied[0] == -99


@pytest.mark.parametrize(
    "dtype,values",
    [
        (raptors.bool_, [False, True]),
        (raptors.int64, [-9, 17]),
        (raptors.uint64, [2**63 + 1, 2**64 - 1]),
        (raptors.float32, [1 / 3, -0.0]),
        (raptors.float64, [-0.0, 1 / 3]),
    ],
)
def test_typed_scalar_results_can_be_reused_as_values(dtype, values):
    reference = np.array(values, dtype=dtype.name)
    candidate = raptors.array(values, dtype=dtype)
    expected = np.zeros(1, dtype=dtype.name)
    actual = raptors.zeros(1, dtype=dtype)
    expected[0] = reference[1]
    actual[0] = candidate[1]
    assert_array_matches(expected, actual)
    assert_array_matches(np.array([reference[1]], dtype=dtype.name), raptors.array([candidate[1]], dtype=dtype))


def test_array_assignment_rejects_unsupported_casting_and_shape_changes():
    actual = raptors.array([1, 2, 3], dtype=raptors.int64)
    with pytest.raises(TypeError, match="dtypes must match"):
        actual[:] = raptors.array([8, 9, 10], dtype=raptors.uint64)
    with pytest.raises(ValueError, match="shapes must match exactly"):
        actual[:] = raptors.array([8, 9], dtype=raptors.int64)


def test_view_keeps_allocation_alive_after_original_is_deleted():
    expected_owner = np.arange(5, dtype=np.uint64)
    expected_view = expected_owner[::-1][1:]
    actual_owner = raptors.array(expected_owner.tolist(), dtype=raptors.uint64)
    actual_view = actual_owner[::-1][1:]
    del actual_owner
    gc.collect()
    expected_view[0] = 77
    actual_view[0] = 77
    assert_array_matches(expected_view, actual_view)


def test_unsupported_surface_is_absent_or_fails_clearly():
    assert not hasattr(raptors, "add")
    assert not hasattr(raptors.Array, "reshape")
    value = raptors.array([1, 2], dtype=raptors.int64)
    with pytest.raises(TypeError):
        _ = value + value
