import gc
import sys
import warnings

import numpy as np
import pytest
import raptors

from harness import assert_array_matches


NUMERIC_DTYPE_CASES = [
    ("bool", raptors.bool_, [False, True]),
    ("int8", raptors.int8, [0, 1, 2]),
    ("uint8", raptors.uint8, [0, 1, 2]),
    ("int16", raptors.int16, [0, 1, 2]),
    ("uint16", raptors.uint16, [0, 1, 2]),
    ("int32", raptors.int32, [0, 1, 2]),
    ("uint32", raptors.uint32, [0, 1, 2]),
    ("int64", raptors.int64, [0, 1, 2]),
    ("uint64", raptors.uint64, [0, 1, 2]),
    ("float16", raptors.float16, [0.0, 1.0, 2.0]),
    ("float32", raptors.float32, [0.0, 1.0, 2.0]),
    ("float64", raptors.float64, [0.0, 1.0, 2.0]),
    ("complex64", raptors.complex64, [0j, 1 + 2j, 2 + 0j]),
    ("complex128", raptors.complex128, [0j, 1 + 2j, 2 + 0j]),
    ("longdouble", raptors.DType("longdouble"), [0.0, 1.0, 2.0]),
    ("clongdouble", raptors.DType("clongdouble"), [0j, 1 + 2j, 2 + 0j]),
]

NUMERIC_DTYPE_ALIASES = [
    "bool_", "?", "b1",
    "i1", "b", "byte", "u1", "B", "ubyte",
    "i2", "h", "short", "u2", "H", "ushort",
    "i4", "i", "intc", "u4", "I", "uintc",
    "i8", "q", "int", "int_", "long", "intp", "p", "n", "longlong",
    "u8", "Q", "uint", "ulong", "uintp", "P", "N", "ulonglong",
    "f2", "e", "half", "f4", "f", "single", "f8", "d", "double", "float",
    "c8", "F", "csingle", "c16", "D", "complex", "cdouble", "G",
    "longdouble", "g", "clongdouble",
]
if np.dtype("longdouble").itemsize > np.dtype("float64").itemsize:
    NUMERIC_DTYPE_ALIASES.extend(("float128", "complex256", "c32"))


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


@pytest.mark.parametrize("value", [10**100, -(10**100)])
def test_boolean_conversion_accepts_arbitrary_size_python_integers(value):
    expected_scalar = np.array(value, dtype=np.bool_)
    actual_scalar = raptors.array(value, dtype=raptors.bool_)
    assert_array_matches(expected_scalar, actual_scalar)

    expected_sequence = np.array([value], dtype=np.bool_)
    actual_sequence = raptors.array([value], dtype=raptors.bool_)
    assert_array_matches(expected_sequence, actual_sequence)

    expected_assignment = np.zeros(1, dtype=np.bool_)
    actual_assignment = raptors.zeros(1, dtype=raptors.bool_)
    expected_assignment[0] = value
    actual_assignment[0] = value
    assert_array_matches(expected_assignment, actual_assignment)


def test_half_dtype_alias_matches_numpy_float16():
    expected_dtype = np.dtype("half")
    actual_dtype = raptors.DType("half")
    assert actual_dtype.name == expected_dtype.name
    assert actual_dtype.itemsize == expected_dtype.itemsize
    assert actual_dtype.char == expected_dtype.char
    assert_array_matches(
        np.array([1.25, -2.5], dtype="half"),
        raptors.array([1.25, -2.5], dtype="half"),
    )


def test_csingle_dtype_alias_matches_numpy_complex64():
    expected_dtype = np.dtype("csingle")
    actual_dtype = raptors.DType("csingle")
    assert actual_dtype.name == expected_dtype.name
    assert actual_dtype.itemsize == expected_dtype.itemsize
    assert actual_dtype.char == expected_dtype.char

    expected = np.array([1 + 2**-30 + 2j], dtype="csingle")
    actual = raptors.array([1 + 2**-30 + 2j], dtype="csingle")
    assert_array_matches(expected, actual)


@pytest.mark.parametrize("alias", NUMERIC_DTYPE_ALIASES)
def test_numeric_dtype_aliases_match_numpy(alias):
    expected_dtype = np.dtype(alias)
    actual_dtype = raptors.DType(alias)
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
        assert getattr(actual_dtype, attribute) == getattr(expected_dtype, attribute), (
            alias,
            attribute,
        )

    values = {
        "b": [False, True],
        "i": [0, 1, 2],
        "u": [0, 1, 2],
        "f": [0.0, 1.25, -2.5],
        "c": [0j, 1 + 2j, 3 - 4j],
    }[expected_dtype.kind]
    expected = np.array(values, dtype=expected_dtype)
    actual = raptors.array(values, dtype=alias)
    assert_array_matches(expected, actual)
    assert actual_dtype.type.__name__ == type(actual[0]).__name__


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
    ([128], raptors.int8, OverflowError),
    ([-129], raptors.int8, OverflowError),
    ([256], raptors.uint8, OverflowError),
    ([-1], raptors.uint8, OverflowError),
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


def test_ragged_shape_error_precedes_element_cast_error():
    values = [[1e100], [1, 2]]
    with pytest.raises(ValueError):
        np.array(values, dtype=np.int8)
    with pytest.raises(ValueError):
        raptors.array(values, dtype=raptors.int8)

    reference = np.zeros(2, dtype=np.int8)
    candidate = raptors.zeros(2, dtype=raptors.int8)
    reference_index = np.array([0, 1], dtype=np.int64)
    candidate_index = raptors.array([0, 1], dtype=raptors.int64)
    with pytest.raises(ValueError):
        reference[reference_index] = values
    with pytest.raises(ValueError):
        candidate[candidate_index] = values
    assert_array_matches(reference, candidate)


@pytest.mark.parametrize(
    "value,dtype",
    [(1e20, raptors.int64), (-1e20, raptors.uint64)],
)
def test_float_to_integer_overflow_matches_numpy(value, dtype):
    with pytest.raises(OverflowError):
        np.array([value], dtype=dtype.name)
    with pytest.raises(OverflowError):
        raptors.array([value], dtype=dtype)

    expected = np.zeros(1, dtype=dtype.name)
    actual = raptors.zeros(1, dtype=dtype)
    with pytest.raises(OverflowError):
        expected[0] = value
    with pytest.raises(OverflowError):
        actual[0] = value


@pytest.mark.parametrize(
    "values,dtype,error",
    [
        ([1e20], raptors.int64, OverflowError),
        ([-1e20], raptors.uint64, OverflowError),
        ([1 + 2j], raptors.float64, TypeError),
    ],
)
@pytest.mark.parametrize("fancy", [False, True])
def test_sequence_assignment_checks_destination_cast(values, dtype, error, fancy):
    reference = np.zeros(1, dtype=dtype.name)
    candidate = raptors.zeros(1, dtype=dtype)
    reference_key = np.array([0]) if fancy else slice(None)
    candidate_key = raptors.array([0], dtype=raptors.int64) if fancy else slice(None)

    with pytest.raises(error):
        reference[reference_key] = values
    with pytest.raises(error):
        candidate[candidate_key] = values
    assert_array_matches(np.zeros(1, dtype=dtype.name), candidate)


def test_sequence_assignment_warns_when_values_overflow_float32():
    values = [1e100, -1e100]
    reference = np.zeros(2, dtype=np.float32)
    candidate = raptors.zeros(2, dtype=raptors.float32)

    with pytest.warns(RuntimeWarning, match="overflow encountered in cast") as expected_warnings:
        reference[:] = values
    with pytest.warns(RuntimeWarning, match="overflow encountered in cast") as actual_warnings:
        candidate[:] = values

    assert len(actual_warnings) == len(expected_warnings) == 2
    assert_array_matches(reference, candidate)


def test_fractional_float_to_unsigned_integer_truncates_toward_zero():
    assert np.array([-0.7], dtype=np.uint8)[0] == 0
    assert raptors.array([-0.7], dtype=raptors.uint8)[0] == 0


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


def test_scalar_shapes_accept_the_integer_index_protocol():
    class IndexLike:
        def __index__(self):
            return 3

    values = np.array([1, 2, 3], dtype=np.int64)
    candidate = raptors.array([1, 2, 3], dtype=raptors.int64)
    for shape in (np.int64(3), IndexLike()):
        assert raptors.zeros(shape).shape == np.zeros(shape).shape == (3,)
        assert raptors.empty(shape).shape == np.empty(shape).shape == (3,)
        assert_array_matches(values.reshape(shape), candidate.reshape(shape))


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

    empty_reference = np.empty((0,), dtype=np.int64)
    empty_candidate = raptors.empty((0,), dtype=raptors.int64)
    for shape, order in [((0, 1), "C"), ((0, 2, 1), "F")]:
        assert_array_matches(
            empty_reference.reshape(shape, order=order),
            empty_candidate.reshape(shape, order=order),
        )
    for shape, order in [((0, 1), "C"), ((2, 0, 3), "C"), ((2, 0, 3), "F")]:
        assert_array_matches(
            empty_reference.reshape(shape, order=order, copy=True),
            empty_candidate.reshape(shape, order=order, copy=True),
        )


def test_reshape_views_compatible_stepped_source():
    reference_owner = np.arange(6, dtype=np.int64)
    candidate_owner = raptors.array(reference_owner.tolist(), dtype=raptors.int64)
    reference_source = reference_owner[::2]
    candidate_source = candidate_owner[::2]

    expected = reference_source.reshape((1, 3), copy=False)
    actual = candidate_source.reshape((1, 3), copy=False)
    assert_array_matches(expected, actual)

    expected[0, 1] = 71
    actual[0, 1] = 71
    assert_array_matches(reference_owner, candidate_owner)

    reference_matrix = np.arange(24, dtype=np.int64).reshape(4, 6)
    candidate_matrix = raptors.array(reference_matrix.tolist(), dtype=raptors.int64)
    expected_stepped_view = reference_matrix[::2].reshape((2, 1, 6), copy=False)
    actual_stepped_view = candidate_matrix[::2].reshape((2, 1, 6), copy=False)
    assert_array_matches(expected_stepped_view, actual_stepped_view)
    expected_stepped_view[1, 0, 0] = 89
    actual_stepped_view[1, 0, 0] = 89
    assert_array_matches(reference_matrix, candidate_matrix)

    expected_fortran_view = reference_matrix.reshape((2, 2, 6), order="F", copy=False)
    actual_fortran_view = candidate_matrix.reshape((2, 2, 6), order="F", copy=False)
    assert_array_matches(expected_fortran_view, actual_fortran_view)
    expected_fortran_view[1, 0, 0] = 91
    actual_fortran_view[1, 0, 0] = 91
    assert_array_matches(reference_matrix, candidate_matrix)


@pytest.mark.parametrize(
    "source_shape,source_kind,target_shape,order,shares",
    [
        ((4, 6), "base", (2, 12), "C", True),
        ((4, 6), "base", (2, 12), "F", True),
        ((4, 6), "transpose", (2, 12), "C", False),
        ((4, 6), "transpose", (2, 12), "F", True),
        ((24,), "reverse", (2, 12), "C", True),
        ((4, 6), "row_step", (12,), "C", False),
    ],
)
def test_reshape_copy_and_view_behavior_matches_numpy(
    source_shape, source_kind, target_shape, order, shares
):
    values = np.arange(np.prod(source_shape), dtype=np.int64).reshape(source_shape)
    owner_order = "F" if source_kind == "base" and order == "F" else "C"
    reference_owner = np.array(values, order=owner_order, copy=True)
    candidate_owner = raptors.array(
        values.tolist(), dtype=raptors.int64, order=owner_order
    )

    if source_kind == "transpose":
        reference_source, candidate_source = reference_owner.T, candidate_owner.T
    elif source_kind == "reverse":
        reference_source, candidate_source = reference_owner[::-1], candidate_owner[::-1]
    elif source_kind == "row_step":
        reference_source, candidate_source = reference_owner[::2], candidate_owner[::2]
    else:
        reference_source, candidate_source = reference_owner, candidate_owner

    expected = reference_source.reshape(target_shape, order=order)
    actual = candidate_source.reshape(target_shape, order=order)
    assert_array_matches(expected, actual)
    assert np.shares_memory(expected, reference_owner) is shares

    if shares:
        assert_array_matches(
            reference_source.reshape(target_shape, order=order, copy=False),
            candidate_source.reshape(target_shape, order=order, copy=False),
        )
    else:
        with pytest.raises(ValueError):
            reference_source.reshape(target_shape, order=order, copy=False)
        with pytest.raises(ValueError):
            candidate_source.reshape(target_shape, order=order, copy=False)

    expected.flat[0] = 97
    actual[tuple(0 for _ in target_shape)] = 97
    assert_array_matches(reference_owner, candidate_owner)

def test_array_copy_none_accepts_copy_if_needed_semantics():
    source = raptors.array([1, 2, 3], dtype=raptors.int64)
    reused = raptors.array(source, copy=None)
    assert_array_matches(np.array([1, 2, 3], dtype=np.int64), reused)

    copied_from_sequence = raptors.array([1, 2, 3], dtype=raptors.int64, copy=None)
    assert_array_matches(np.array([1, 2, 3], dtype=np.int64), copied_from_sequence)


@pytest.mark.parametrize("copy", [False, None])
def test_array_reuses_equivalent_native_byteorder_descriptor(copy):
    order = "<" if sys.byteorder == "little" else ">"
    target_dtype = np.dtype("i2").newbyteorder(order)
    source = raptors.array([1, 2], dtype=raptors.DType("=i2"))
    expected_source = np.array([1, 2], dtype=np.dtype("=i2"))

    actual = raptors.array(
        source,
        dtype=raptors.DType("i2").newbyteorder(order),
        copy=copy,
    )
    expected = np.array(expected_source, dtype=target_dtype, copy=copy)
    assert_array_matches(expected, actual)
    assert actual.dtype.byteorder == expected.dtype.byteorder

    actual[0] = 41
    assert source[0] == 41
    source[1] = 73
    assert actual[1] == 73


@pytest.mark.parametrize("copy", [False, None])
@pytest.mark.parametrize(
    "source_dtype,target_dtype,values,replacement,convert",
    [
        ("float64", "longdouble", [1.25, -2.5], 7.5, float),
        ("complex128", "clongdouble", [1 + 2j, 3 - 4j], 7 + 8j, complex),
    ],
)
def test_array_reuses_equal_width_float_dtype_aliases(
    copy, source_dtype, target_dtype, values, replacement, convert
):
    source = raptors.array(values, dtype=raptors.DType(source_dtype))
    expected_dtype = np.dtype(target_dtype)
    if np.dtype(source_dtype) != expected_dtype:
        if copy is False:
            with pytest.raises(ValueError):
                raptors.array(source, dtype=raptors.DType(target_dtype), copy=False)
            return
        expected = np.array(values, dtype=np.dtype(source_dtype)).astype(expected_dtype)
        actual = raptors.array(source, dtype=raptors.DType(target_dtype), copy=None)
        assert_array_matches(expected, actual)
        actual[0] = replacement
        assert convert(source[0]) == convert(values[0])
        return

    actual = raptors.array(source, dtype=raptors.DType(target_dtype), copy=copy)
    assert actual.dtype.name == expected_dtype.name
    assert actual.dtype.char == expected_dtype.char
    assert actual.dtype.byteorder == expected_dtype.byteorder
    assert type(actual[0]).__name__ == (
        "LongDoubleScalar" if expected_dtype.kind == "f" else "ComplexLongDoubleScalar"
    )

    actual[0] = replacement
    assert convert(source[0]) == convert(replacement)
    source[1] = replacement
    assert convert(actual[1]) == convert(replacement)

    for derived in (actual.copy(), actual[[1, 0]]):
        assert derived.dtype.char == expected_dtype.char
        assert convert(derived[0]) == convert(replacement)


def test_astype_preserves_fortran_layout_by_default():
    reference = np.asfortranarray(np.arange(12, dtype=np.int64).reshape(3, 4))
    candidate = raptors.array(reference.tolist(), dtype=raptors.int64, order="F")

    actual = candidate.astype("float32")
    expected = reference.astype(np.float32)
    assert_array_matches(expected, actual)
    assert actual.f_contiguous and not actual.c_contiguous

    c_order = candidate.astype("float32", order="C")
    assert c_order.c_contiguous and not c_order.f_contiguous


def test_keep_order_preserves_noncontiguous_axis_permutation():
    reference = np.arange(24, dtype=np.int64).reshape(4, 6).T[::2]
    candidate = raptors.array(
        np.arange(24, dtype=np.int64).reshape(4, 6).tolist(), dtype=raptors.int64
    ).T[::2]

    expected_cast = reference.astype(np.float32, order="K")
    actual_cast = candidate.astype(raptors.float32, order="K")
    assert_array_matches(expected_cast, actual_cast)

    expected_copy = np.array(reference, order="K", copy=True)
    actual_copy = raptors.array(candidate, order="K", copy=True)
    assert_array_matches(expected_copy, actual_copy)


def test_any_order_uses_c_layout_when_singleton_array_is_both_contiguous():
    reference = np.empty((1, 3), dtype=np.int64)
    candidate = raptors.array(reference.tolist(), dtype=raptors.int64)
    assert reference.flags.c_contiguous and reference.flags.f_contiguous

    assert_array_matches(
        np.array(reference, order="A", copy=True),
        raptors.array(candidate, order="A", copy=True),
    )
    assert_array_matches(
        reference.astype(np.float32, order="A"),
        candidate.astype(raptors.float32, order="A"),
    )
    assert_array_matches(
        reference.reshape((3, 1), order="A"),
        candidate.reshape((3, 1), order="A"),
    )


def test_longdouble_dtype_alias_equality_matches_numpy_on_each_platform():
    for long_name, alias_name in (
        ("longdouble", "float64"),
        ("clongdouble", "complex128"),
    ):
        expected_long = np.dtype(long_name)
        expected_alias = np.dtype(alias_name)
        actual_long = raptors.DType(long_name)
        actual_alias = raptors.DType(alias_name)
        assert (expected_long == expected_alias) == (actual_long == actual_alias)
        assert (expected_alias == expected_long) == (actual_alias == actual_long)
        assert (expected_long == expected_alias) == (actual_long == alias_name)
        assert (expected_alias == expected_long) == (actual_alias == long_name)

        long_array = raptors.array([1], dtype=raptors.DType(long_name))
        alias_array = raptors.array([1], dtype=raptors.DType(alias_name))
        assert (long_array.dtype == alias_array.dtype) == (expected_long == expected_alias)
        assert_array_matches(np.array([1], dtype=expected_long), long_array)
        assert_array_matches(np.array([1], dtype=expected_alias), alias_array)


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
            scalar = candidate[0]
            assert raptors.array([scalar]).dtype.char == expected.char
            assert raptors.promote_types(code, code).char == expected.char
            assert raptors.promote_types(code, expected.name).char == np.promote_types(
                code, expected.name
            ).char


@pytest.mark.parametrize(("code", "constant_name"), [("i", "intc"), ("I", "uintc")])
def test_c_integer_dtype_codes_survive_arrays_scalars_and_promotion(code, constant_name):
    expected = np.dtype(code)
    descriptor = raptors.DType(code)
    assert descriptor.char == expected.char
    assert getattr(raptors, constant_name).char == expected.char

    expected_array = np.array([1, 2], dtype=expected)
    actual_array = raptors.array([1, 2], dtype=code)
    assert_array_matches(expected_array, actual_array)
    scalar = actual_array[0]
    assert scalar.dtype.char == expected.char
    assert raptors.array([scalar]).dtype.char == expected.char
    assert raptors.promote_types(code, code).char == np.promote_types(code, code).char


def test_fixed_width_int32_uses_c_int_typecode_on_windows():
    if sys.platform != "win32":
        pytest.skip("Windows distinguishes C int from C long typecodes")

    expected_int = np.dtype("int32")
    expected_uint = np.dtype("uint32")
    assert raptors.DType("int32").char == expected_int.char == "i"
    assert raptors.DType("uint32").char == expected_uint.char == "I"
    assert raptors.DType("long").char == np.dtype("long").char == "l"
    assert raptors.DType("ulong").char == np.dtype("ulong").char == "L"


def test_integer_promotion_uses_numpy_c_int_descriptor_when_needed():
    expected = np.promote_types(np.dtype("int8"), np.dtype("uint16"))
    actual = raptors.promote_types("int8", "uint16")
    assert actual.name == expected.name
    assert actual.char == expected.char


@pytest.mark.parametrize("complex_dtype", [raptors.complex64, raptors.complex128])
def test_complex_and_longdouble_promotion_matches_numpy(complex_dtype):
    expected = np.promote_types(np.dtype(complex_dtype.name), np.dtype(np.longdouble))
    actual = raptors.promote_types(complex_dtype, raptors.DType("longdouble"))
    assert actual.kind == expected.kind
    assert actual.itemsize == expected.itemsize
    assert actual.char == expected.char


def test_ellipsis_counts_nested_boolean_list_rank():
    values = np.arange(24, dtype=np.int64).reshape(2, 3, 2, 2)
    mask = [[True, False], [False, True]]
    candidate = raptors.array(values.tolist(), dtype=raptors.int64)
    expected = values[..., mask]
    actual = candidate[..., mask]
    assert tuple(actual.shape) == expected.shape
    for index in np.ndindex(expected.shape):
        assert actual[index] == expected[index]


def test_dtype_metadata_and_inferred_dtype():
    for dtype in (raptors.bool_, raptors.int64, raptors.uint64, raptors.float32, raptors.float64):
        assert dtype.name in {"bool", "int64", "uint64", "float32", "float64"}
        assert dtype.itemsize == np.dtype(dtype.name).itemsize
        assert dtype.kind == np.dtype(dtype.name).kind
    assert raptors.array([[1, 2], [3, 4]], dtype="int64").dtype == raptors.int64
    assert_array_matches(np.array([1, 2]), raptors.array([1, 2]))


@pytest.mark.parametrize("dtype_name,dtype,values", NUMERIC_DTYPE_CASES)
def test_all_numeric_dtype_metadata_and_round_trip(dtype_name, dtype, values):
    expected_dtype = np.dtype(dtype_name)
    actual_dtype = raptors.DType(dtype_name)

    assert actual_dtype.name == expected_dtype.name
    assert actual_dtype.kind == expected_dtype.kind
    assert actual_dtype.char == expected_dtype.char
    assert actual_dtype.itemsize == expected_dtype.itemsize
    assert actual_dtype.alignment == expected_dtype.alignment
    assert actual_dtype.byteorder == expected_dtype.byteorder
    assert actual_dtype.isnative == expected_dtype.isnative
    assert actual_dtype.str == expected_dtype.str

    expected = np.array(values, dtype=expected_dtype)
    actual = raptors.array(values, dtype=dtype)
    scalar_name = type(actual[0]).__name__
    assert actual_dtype.type.__name__ == scalar_name
    assert actual[0].dtype.type.__name__ == scalar_name
    assert_array_matches(expected, actual)
    assert_array_matches(expected.copy(), actual.copy())


@pytest.mark.parametrize(
    "values,mask",
    [
        ([3, 5, 7, 9], [True, False, True, False]),
        ([3, 5, 7, 9], [False, False, False, False]),
        ([], []),
    ],
)
def test_boolean_mask_indexing_and_assignment_matches_numpy(values, mask):
    reference = np.array(values, dtype=np.int64)
    candidate = raptors.array(values, dtype=raptors.int64)
    reference_mask = np.array(mask, dtype=np.bool_)
    candidate_mask = raptors.array(mask, dtype=raptors.bool_)

    expected_selection = reference[reference_mask]
    actual_selection = candidate[candidate_mask]
    assert_array_matches(expected_selection, actual_selection)
    if expected_selection.size:
        expected_selection[0] = -11
        actual_selection[0] = -11
        assert_array_matches(reference, candidate)

    replacement = np.arange(expected_selection.size, dtype=np.int64) + 20
    reference[reference_mask] = replacement
    candidate[candidate_mask] = raptors.array(replacement.tolist(), dtype=raptors.int64)
    assert_array_matches(reference, candidate)


def test_boolean_mask_rank_shape_errors_match_numpy():
    reference = np.arange(6, dtype=np.int64).reshape(2, 3)
    candidate = raptors.array(reference.tolist(), dtype=raptors.int64)
    for mask in (np.array([True, False, True]), np.ones((2, 2), dtype=np.bool_)):
        with pytest.raises(IndexError) as expected_error:
            _ = reference[mask]
        with pytest.raises(type(expected_error.value)):
            _ = candidate[raptors.array(mask.tolist(), dtype=raptors.bool_)]


def test_fancy_index_copy_duplicate_writes_and_overlapping_source_match_numpy():
    reference = np.arange(5, dtype=np.int64)
    candidate = raptors.array(reference.tolist(), dtype=raptors.int64)
    key_values = [1, 1, -1]
    reference_key = np.array(key_values, dtype=np.int64)
    candidate_key = raptors.array(key_values, dtype=raptors.int64)

    expected_selection = reference[reference_key]
    actual_selection = candidate[candidate_key]
    assert_array_matches(expected_selection, actual_selection)
    expected_selection[0] = 99
    actual_selection[0] = 99
    assert_array_matches(reference, candidate)

    reference[reference_key] = [10, 20, 30]
    candidate[candidate_key] = [10, 20, 30]
    assert_array_matches(reference, candidate)

    reference = np.arange(5, dtype=np.int64)
    candidate = raptors.array(reference.tolist(), dtype=raptors.int64)
    reference[[0, 1, 2, 3, 4]] = reference[[4, 3, 2, 1, 0]]
    candidate[[0, 1, 2, 3, 4]] = candidate[[4, 3, 2, 1, 0]]
    assert_array_matches(reference, candidate)


@pytest.mark.parametrize("source_name,source_dtype,values", NUMERIC_DTYPE_CASES)
@pytest.mark.parametrize("target_name,target_dtype,_", NUMERIC_DTYPE_CASES)
def test_numeric_array_cast_matrix_matches_numpy(
    source_name, source_dtype, values, target_name, target_dtype, _
):
    expected_source = np.array(values, dtype=source_name)
    actual_source = raptors.array(values, dtype=source_dtype)

    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter("always")
        expected = expected_source.astype(target_name)
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter("always")
        actual = actual_source.astype(target_dtype)

    assert [str(item.message) for item in actual_warnings] == [
        str(item.message) for item in expected_warnings
    ]
    assert [type(item.message).__name__ for item in actual_warnings] == [
        type(item.message).__name__ for item in expected_warnings
    ]
    assert_array_matches(expected, actual)


@pytest.mark.parametrize("left_name,left_dtype,_", NUMERIC_DTYPE_CASES)
@pytest.mark.parametrize("right_name,right_dtype,__", NUMERIC_DTYPE_CASES)
def test_numeric_dtype_promotion_matrix_matches_numpy(
    left_name, left_dtype, _, right_name, right_dtype, __
):
    expected = np.promote_types(np.dtype(left_name), np.dtype(right_name))
    actual = raptors.promote_types(left_dtype, right_dtype)

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
        assert getattr(actual, attribute) == getattr(expected, attribute), attribute


def test_float32_overflow_warning_matches_reference():
    values = [1e100, 1e100]
    with pytest.warns(RuntimeWarning, match="overflow encountered in cast") as reference_warnings:
        reference = np.array(values, dtype=np.float32)
    with pytest.warns(RuntimeWarning, match="overflow encountered in cast") as candidate_warnings:
        candidate = raptors.array(values, dtype=raptors.float32)
    assert len(candidate_warnings) == len(reference_warnings) == 2
    assert_array_matches(reference, candidate)


def test_float16_overflow_warning_starts_at_rounding_boundary():
    values = [65505.0, 65520.0]
    with pytest.warns(RuntimeWarning, match="overflow encountered in cast") as reference_warnings:
        reference = np.array(values, dtype=np.float16)
    with pytest.warns(RuntimeWarning, match="overflow encountered in cast") as candidate_warnings:
        candidate = raptors.array(values, dtype=raptors.float16)
    assert len(candidate_warnings) == len(reference_warnings) == 1
    assert_array_matches(reference, candidate)


def test_array_casts_warn_when_finite_values_overflow_float32():
    values = [1e100, -1e100]
    reference_source = np.array(values, dtype=np.float64)
    candidate_source = raptors.array(values, dtype=raptors.float64)

    with pytest.warns(RuntimeWarning, match="overflow encountered in cast") as reference_warnings:
        expected = reference_source.astype(np.float32)
    with pytest.warns(RuntimeWarning, match="overflow encountered in cast") as candidate_warnings:
        actual = candidate_source.astype(raptors.float32)
    assert len(candidate_warnings) == len(reference_warnings) == 1
    assert_array_matches(expected, actual)

    with pytest.warns(RuntimeWarning, match="overflow encountered in cast") as candidate_warnings:
        copied = raptors.array(candidate_source, dtype=raptors.float32)
    assert len(candidate_warnings) == 1
    assert_array_matches(expected, copied)


@pytest.mark.parametrize(
    "dtype",
    [
        raptors.int8,
        raptors.uint8,
        raptors.int16,
        raptors.uint16,
        raptors.int32,
        raptors.uint32,
        raptors.int64,
        raptors.uint64,
    ],
)
def test_float_array_to_integer_cast_matches_numpy_for_in_range_values(dtype):
    values = [-1.75, 0.0, 1.75] if dtype.kind == "i" else [0.0, 1.75, 42.5]
    reference_source = np.array(values, dtype=np.float64)
    candidate_source = raptors.array(values, dtype=raptors.float64)

    expected = reference_source.astype(dtype.name)
    actual = candidate_source.astype(dtype)
    assert_array_matches(expected, actual)

    expected_assignment = np.zeros(len(values), dtype=dtype.name)
    actual_assignment = raptors.zeros(len(values), dtype=dtype)
    expected_assignment[:] = reference_source
    actual_assignment[:] = candidate_source
    assert_array_matches(expected_assignment, actual_assignment)


@pytest.mark.parametrize(
    "dtype",
    [
        raptors.int8,
        raptors.uint8,
        raptors.int16,
        raptors.uint16,
        raptors.int32,
        raptors.uint32,
        raptors.int64,
        raptors.uint64,
    ],
)
def test_float_array_to_integer_invalid_cast_warns_without_requiring_undefined_values(dtype):
    values = [1e20, -1e20, float("inf"), float("-inf"), float("nan")]
    reference_source = np.array(values, dtype=np.float64)
    candidate_source = raptors.array(values, dtype=raptors.float64)

    with pytest.warns(RuntimeWarning, match="invalid value encountered in cast") as expected_warnings:
        expected = reference_source.astype(dtype.name)
    with pytest.warns(RuntimeWarning, match="invalid value encountered in cast") as actual_warnings:
        actual = candidate_source.astype(dtype)
    assert len(actual_warnings) == len(expected_warnings) == 1
    assert actual.dtype.name == expected.dtype.name == dtype.name
    assert tuple(actual.shape) == tuple(expected.shape)

    expected_assignment = np.zeros(len(values), dtype=dtype.name)
    actual_assignment = raptors.zeros(len(values), dtype=dtype)
    with pytest.warns(RuntimeWarning, match="invalid value encountered in cast") as expected_warnings:
        expected_assignment[:] = reference_source
    with pytest.warns(RuntimeWarning, match="invalid value encountered in cast") as actual_warnings:
        actual_assignment[:] = candidate_source
    assert len(actual_warnings) == len(expected_warnings) == 1
    assert actual_assignment.dtype.name == expected_assignment.dtype.name == dtype.name
    assert tuple(actual_assignment.shape) == tuple(expected_assignment.shape)


def test_complex_to_real_scalar_conversion_raises_type_error():
    with pytest.raises(TypeError):
        np.array([1 + 2j], dtype=np.float32)
    with pytest.raises(TypeError):
        raptors.array([1 + 2j], dtype=raptors.float32)

    reference = np.zeros(1, dtype=np.float32)
    candidate = raptors.zeros(1, dtype=raptors.float32)
    with pytest.raises(TypeError):
        reference[0] = 1 + 2j
    with pytest.raises(TypeError):
        candidate[0] = 1 + 2j


def test_complex_to_real_array_cast_emits_complex_warning():
    reference_source = np.array([1 + 2j, 3 + 0j], dtype=np.complex128)
    candidate_source = raptors.array([1 + 2j, 3 + 0j], dtype=raptors.complex128)

    with pytest.warns(np.exceptions.ComplexWarning, match="discards the imaginary part") as expected_warnings:
        expected = reference_source.astype(np.float32)
    with pytest.warns(raptors.ComplexWarning, match="discards the imaginary part") as actual_warnings:
        actual = candidate_source.astype(raptors.float32)
    assert len(actual_warnings) == len(expected_warnings) == 1
    assert_array_matches(expected, actual)

    with pytest.warns(raptors.ComplexWarning, match="discards the imaginary part"):
        copied = raptors.array(candidate_source, dtype=raptors.float32)
    assert_array_matches(expected, copied)

    reference_assignment = np.zeros(2, dtype=np.float32)
    candidate_assignment = raptors.zeros(2, dtype=raptors.float32)
    with pytest.warns(np.exceptions.ComplexWarning, match="discards the imaginary part"):
        reference_assignment[:] = reference_source
    with pytest.warns(raptors.ComplexWarning, match="discards the imaginary part"):
        candidate_assignment[:] = candidate_source
    assert_array_matches(reference_assignment, candidate_assignment)


@pytest.mark.parametrize("operation", ["astype", "array", "assignment"])
def test_complex_to_unsigned_cast_warning_order_matches_numpy(operation):
    reference_source = np.array([1 + 1j], dtype=np.complex128)
    candidate_source = raptors.array([1 + 1j], dtype=raptors.complex128)

    def apply_numpy():
        if operation == "astype":
            return reference_source.astype(np.uint64)
        if operation == "array":
            return np.array(reference_source, dtype=np.uint64)
        result = np.zeros(1, dtype=np.uint64)
        result[:] = reference_source
        return result

    def apply_raptors():
        if operation == "astype":
            return candidate_source.astype(raptors.uint64)
        if operation == "array":
            return raptors.array(candidate_source, dtype=raptors.uint64)
        result = raptors.zeros(1, dtype=raptors.uint64)
        result[:] = candidate_source
        return result

    with pytest.warns() as expected_warnings:
        expected = apply_numpy()
    with pytest.warns() as actual_warnings:
        actual = apply_raptors()

    assert [str(w.message) for w in actual_warnings] == [
        str(w.message) for w in expected_warnings
    ]
    assert_array_matches(expected, actual)


@pytest.mark.parametrize("operation", ["astype", "array", "assignment"])
def test_invalid_complex_to_unsigned_cast_keeps_complex_warning(operation):
    reference_source = np.array([-1 + 1j], dtype=np.complex128)
    candidate_source = raptors.array([-1 + 1j], dtype=raptors.complex128)

    def apply_numpy():
        if operation == "astype":
            return reference_source.astype(np.uint64)
        if operation == "array":
            return np.array(reference_source, dtype=np.uint64)
        result = np.zeros(1, dtype=np.uint64)
        result[:] = reference_source
        return result

    def apply_raptors():
        if operation == "astype":
            return candidate_source.astype(raptors.uint64)
        if operation == "array":
            return raptors.array(candidate_source, dtype=raptors.uint64)
        result = raptors.zeros(1, dtype=raptors.uint64)
        result[:] = candidate_source
        return result

    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter("always")
        expected = apply_numpy()
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter("always")
        actual = apply_raptors()

    assert any(issubclass(item.category, np.exceptions.ComplexWarning) for item in expected_warnings)
    assert any(issubclass(item.category, raptors.ComplexWarning) for item in actual_warnings)
    assert expected.dtype == np.dtype(np.uint64)
    assert actual.dtype.name == "uint64"
    assert tuple(actual.shape) == tuple(expected.shape)


def test_complex64_overflow_emits_runtime_warning():
    reference_source = np.array([1e100, -1e100], dtype=np.float64)
    candidate_source = raptors.array([1e100, -1e100], dtype=raptors.float64)

    with pytest.warns(RuntimeWarning, match="overflow encountered in cast") as expected_warnings:
        expected = reference_source.astype(np.complex64)
    with pytest.warns(RuntimeWarning, match="overflow encountered in cast") as actual_warnings:
        actual = candidate_source.astype(raptors.complex64)
    assert len(actual_warnings) == len(expected_warnings)
    assert np.isinf(actual[0].item().real) and np.isinf(actual[1].item().real)
    assert_array_matches(expected, actual)

    with pytest.warns(RuntimeWarning, match="overflow encountered in cast"):
        constructed = raptors.array([1e100 + 0j], dtype=raptors.complex64)
    assert np.isinf(constructed[0].item().real)


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


def test_scalar_integer_indices_participate_in_advanced_axis_placement():
    values = np.arange(3 * 4 * 5, dtype=np.int64).reshape(3, 4, 5)
    reference = values
    candidate = raptors.array(values.tolist(), dtype=raptors.int64)
    fancy = np.array([0, 2], dtype=np.int64)
    fancy_candidate = raptors.array([0, 2], dtype=raptors.int64)

    for reference_key, candidate_key in [
        ((0, slice(None), fancy), (0, slice(None), fancy_candidate)),
        ((0, None, fancy), (0, None, fancy_candidate)),
    ]:
        expected = reference[reference_key]
        actual = candidate[candidate_key]
        assert tuple(actual.shape) == expected.shape
        for coordinates in np.ndindex(expected.shape):
            assert int(actual[tuple(map(int, coordinates))]) == int(expected[coordinates])


def test_raptors_boolean_scalar_is_a_boolean_index():
    reference = np.arange(6, dtype=np.int64).reshape(2, 3)
    candidate = raptors.array(reference.tolist(), dtype=raptors.int64)
    for value in (False, True):
        reference_key = np.array(value, dtype=np.bool_)[()]
        candidate_key = raptors.array(value, dtype=raptors.bool_)[()]
        assert type(candidate_key).__name__ == "BoolScalar"
        assert_array_matches(reference[reference_key], candidate[candidate_key])
        assert_array_matches(reference[(reference_key, ...)], candidate[(candidate_key, ...)])


@pytest.mark.parametrize("order", ["C", "F"])
def test_advanced_index_result_strides_match_numpy(order):
    values_2d = np.arange(12, dtype=np.int64).reshape((3, 4), order=order)
    candidate_2d = raptors.array(values_2d.tolist(), dtype=raptors.int64, order=order)
    index = raptors.array([0, 2], dtype=raptors.int64)
    assert_array_matches(values_2d[:, [0, 2]], candidate_2d[:, index])

    values_3d = np.arange(24, dtype=np.int64).reshape((2, 3, 4), order=order)
    candidate_3d = raptors.array(values_3d.tolist(), dtype=raptors.int64, order=order)
    assert_array_matches(values_3d[:, :, [0, 2]], candidate_3d[:, :, index])


@pytest.mark.parametrize("index_order", ["C", "F"])
def test_advanced_index_result_order_follows_index_arrays_for_singleton_subspace(index_order):
    values = np.arange(4, dtype=np.int64)
    candidate = raptors.array(values.tolist(), dtype=raptors.int64)
    index_values = np.array([[0, 1], [2, 3]], dtype=np.int64, order=index_order)
    index = raptors.array(index_values.tolist(), dtype=raptors.int64, order=index_order)

    assert_array_matches(values[index_values], candidate[index])


@pytest.mark.parametrize(
    "first_order, second_order",
    [("C", "C"), ("F", "F"), ("C", "F"), ("F", "C")],
)
def test_advanced_index_result_order_with_multiple_index_operands(first_order, second_order):
    values = np.arange(24, dtype=np.int64).reshape(4, 6)
    candidate = raptors.array(values.tolist(), dtype=raptors.int64)
    first_values = np.array([[0, 1], [2, 3]], dtype=np.int64, order=first_order)
    second_values = np.array([[1, 2], [3, 4]], dtype=np.int64, order=second_order)
    first = raptors.array(first_values.tolist(), dtype=raptors.int64, order=first_order)
    second = raptors.array(second_values.tolist(), dtype=raptors.int64, order=second_order)

    assert_array_matches(values[first_values, second_values], candidate[first, second])


@pytest.mark.parametrize("use_array_index", [False, True])
def test_uint64_fancy_indices_wrap_to_platform_index_width(use_array_index):
    reference = np.arange(12, dtype=np.int64)
    candidate = raptors.array(reference.tolist(), dtype=raptors.int64)
    value = 2**64 - 1
    reference_index = np.array([value], dtype=np.uint64)
    candidate_index = (
        raptors.array([value], dtype=raptors.uint64) if use_array_index else [value]
    )

    assert_array_matches(reference[reference_index], candidate[candidate_index])
    reference[reference_index] = 77
    candidate[candidate_index] = 77
    assert_array_matches(reference, candidate)


def test_zero_dimensional_integer_array_index_returns_scalar():
    reference = np.array([10, 20], dtype=np.int64)
    candidate = raptors.array([10, 20], dtype=raptors.int64)
    expected = reference[np.array(1, dtype=np.int64)]
    actual = candidate[raptors.array(1, dtype=raptors.int64)]
    assert np.isscalar(expected)
    assert type(actual).__name__ == "Int64Scalar"
    assert actual == expected


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


@pytest.mark.parametrize("name", ["bool", "int8", "uint8"])
def test_one_byte_newbyteorder_remains_native_and_equal(name):
    expected_base = np.dtype(name)
    expected = expected_base.newbyteorder(">")
    base = raptors.DType(name)
    actual = base.newbyteorder(">")
    assert actual.byteorder == expected.byteorder == "|"
    assert actual.isnative == expected.isnative
    assert actual == base


@pytest.mark.parametrize("dtype_spec", ["i2", ">i2", "<f4", "c16", "longdouble"])
@pytest.mark.parametrize("order", ["<", "=", ">", "|", "S"])
def test_newbyteorder_metadata_and_equality_match_numpy(dtype_spec, order):
    expected_base = np.dtype(dtype_spec)
    actual_base = raptors.DType(dtype_spec)
    expected = expected_base.newbyteorder(order)
    actual = actual_base.newbyteorder(order)

    for attribute in ("name", "itemsize", "kind", "char", "byteorder", "isnative", "str"):
        assert getattr(actual, attribute) == getattr(expected, attribute)
    assert (actual == actual_base) == (expected == expected_base)

    expected_array = np.array([1, 2], dtype=expected)
    actual_array = raptors.array([1, 2], dtype=actual)
    assert actual_array.dtype.byteorder == expected_array.dtype.byteorder
    assert actual_array.dtype.str == expected_array.dtype.str
    if dtype_spec == "longdouble":
        assert [float(actual_array[index]) for index in range(2)] == [
            float(expected_array[index]) for index in range(2)
        ]
    else:
        assert_array_matches(expected_array, actual_array)


def test_non_numpy_dtype_spellings_are_rejected():
    with pytest.raises(TypeError):
        np.dtype("uint_")
    with pytest.raises(ValueError):
        raptors.DType("uint_")
    assert not hasattr(raptors, "uint_")
    with pytest.raises(TypeError):
        np.dtype("long_double")
    with pytest.raises(ValueError):
        raptors.DType("long_double")


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


def test_array_assignment_casts_values_and_rejects_shape_changes():
    expected = np.array([1, 2, 3], dtype=np.int64)
    actual = raptors.array([1, 2, 3], dtype=raptors.int64)
    expected[:] = np.array([8, 9, 10], dtype=np.uint64)
    actual[:] = raptors.array([8, 9, 10], dtype=raptors.uint64)
    assert_array_matches(expected, actual)
    with pytest.raises(ValueError):
        actual[:] = raptors.array([8, 9], dtype=raptors.int64)


def test_basic_sequence_assignment_keeps_successful_prefix_on_conversion_error():
    reference = np.zeros(3, dtype=np.int64)
    candidate = raptors.zeros(3, dtype=raptors.int64)
    values = [1, 1e20, 3]

    with pytest.raises(OverflowError):
        reference[:] = values
    with pytest.raises(OverflowError):
        candidate[:] = values
    assert_array_matches(reference, candidate)
    assert reference.tolist() == [1, 0, 0]


def test_broadcast_sequence_assignment_cast_error_does_not_partially_write():
    reference = np.zeros((2, 2), dtype=np.uint8)
    candidate = raptors.zeros((2, 2), dtype=raptors.uint8)
    values = [[1], [1e100]]

    with pytest.raises(OverflowError):
        reference[:, :] = values
    with pytest.raises(OverflowError):
        candidate[:, :] = values
    assert_array_matches(reference, candidate)


def test_basic_assignment_checks_excess_sequence_rank_before_cast():
    reference = np.zeros((2, 3), dtype=np.int8)
    candidate = raptors.zeros((2, 3), dtype=raptors.int8)
    values = [[1e100]]

    with pytest.raises(ValueError):
        reference[0] = values
    with pytest.raises(ValueError):
        candidate[0] = values
    assert_array_matches(reference, candidate)


@pytest.mark.parametrize(
    "dtype,error",
    [
        (raptors.int8, TypeError),
        (raptors.float32, ValueError),
        (raptors.complex64, TypeError),
    ],
)
def test_scalar_target_sequence_assignment_rejects_sequence_before_cast(dtype, error):
    reference = np.zeros(1, dtype=dtype.name)
    candidate = raptors.zeros(1, dtype=dtype)

    with pytest.raises(error):
        reference[0] = [1e100]
    with pytest.raises(error):
        candidate[0] = [1e100]
    assert_array_matches(reference, candidate)


def test_scalar_boolean_assignment_uses_sequence_truthiness():
    reference = np.zeros(2, dtype=np.bool_)
    candidate = raptors.zeros(2, dtype=raptors.bool_)

    reference[0] = [False]
    candidate[0] = [False]
    reference[1] = []
    candidate[1] = []
    assert_array_matches(reference, candidate)


def test_zero_dimensional_ellipsis_assignment_preserves_array_semantics():
    reference = np.zeros((), dtype=np.bool_)
    candidate = raptors.zeros((), dtype=raptors.bool_)

    with pytest.raises(ValueError):
        reference[...] = [False]
    with pytest.raises(ValueError):
        candidate[...] = [False]
    assert_array_matches(reference, candidate)

    reference[()] = [False]
    candidate[()] = [False]
    assert_array_matches(reference, candidate)


@pytest.mark.parametrize(
    "dtype,value",
    [
        (raptors.int8, [1e100]),
        (raptors.float32, [1 + 2j]),
    ],
)
def test_zero_dimensional_array_rank_error_precedes_value_cast(dtype, value):
    reference = np.zeros((), dtype=dtype.name)
    candidate = raptors.zeros((), dtype=dtype)

    with pytest.raises(ValueError):
        reference[...] = value
    with pytest.raises(ValueError):
        candidate[...] = value
    assert_array_matches(reference, candidate)


def test_sequence_assignment_conversion_error_precedes_broadcast_error():
    reference = np.zeros(3, dtype=np.int64)
    candidate = raptors.zeros(3, dtype=raptors.int64)
    values = [1, 1e20]

    with pytest.raises(OverflowError):
        reference[:] = values
    with pytest.raises(OverflowError):
        candidate[:] = values
    assert_array_matches(reference, candidate)


def test_fancy_sequence_assignment_does_not_partially_write_on_conversion_error():
    reference = np.zeros(3, dtype=np.int64)
    candidate = raptors.zeros(3, dtype=raptors.int64)
    reference_key = np.array([0, 1, 2], dtype=np.int64)
    candidate_key = raptors.array([0, 1, 2], dtype=raptors.int64)
    values = [1, 1e20, 3]

    with pytest.raises(OverflowError):
        reference[reference_key] = values
    with pytest.raises(OverflowError):
        candidate[candidate_key] = values
    assert_array_matches(reference, candidate)
    assert reference.tolist() == [0, 0, 0]


def test_python_integer_to_longdouble_preserves_extended_precision():
    value = 2**53 + 1
    reference = np.array([value], dtype=np.longdouble)
    candidate = raptors.array([value], dtype=raptors.DType("longdouble"))

    if np.dtype("longdouble").itemsize > np.dtype("float64").itemsize:
        assert np.longdouble(str(candidate[0])) == reference[0]
    else:
        assert float(candidate[0]) == float(reference[0])


@pytest.mark.parametrize("fancy", [False, True])
def test_array_assignment_warns_when_finite_values_overflow_float32(fancy):
    values = [1e100, -1e100]
    reference_source = np.array(values, dtype=np.float64)
    candidate_source = raptors.array(values, dtype=raptors.float64)
    reference = np.zeros(2, dtype=np.float32)
    candidate = raptors.zeros(2, dtype=raptors.float32)
    reference_key = np.array([0, 1]) if fancy else slice(None)
    candidate_key = raptors.array([0, 1], dtype=raptors.int64) if fancy else slice(None)

    with pytest.warns(RuntimeWarning, match="overflow encountered in cast") as expected_warnings:
        reference[reference_key] = reference_source
    with pytest.warns(RuntimeWarning, match="overflow encountered in cast") as actual_warnings:
        candidate[candidate_key] = candidate_source

    assert len(actual_warnings) == len(expected_warnings) == 1
    assert_array_matches(reference, candidate)


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


def test_phase_03_ufuncs_work_while_later_matrix_operations_stay_absent():
    assert not hasattr(raptors, "matmul")
    value = raptors.array([1, 2], dtype=raptors.int64)
    assert_array_matches(np.array([1, 2], dtype=np.int64).reshape((1, 2)), value.reshape((1, 2)))
    assert_array_matches(np.array([2, 4], dtype=np.int64), value + value)
