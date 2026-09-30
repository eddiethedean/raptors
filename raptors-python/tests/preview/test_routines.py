import gc
import inspect
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import raptors

from harness import assert_array_matches


def capture_warnings(call):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = call()
    return result, [(type(item.message), str(item.message)) for item in captured]


def test_frozen_0_4_contract_names_are_public():
    contract_path = Path(__file__).resolve().parents[3] / "compat" / "raptors-0.4.json"
    contract = json.loads(contract_path.read_text())
    scope = contract["scope"]
    routines = {name for names in scope["families"].values() for name in names}
    missing_routines = sorted(name for name in routines if not hasattr(raptors, name))
    missing_methods = sorted(
        name for name in scope["array_methods"] if not hasattr(raptors.Array, name)
    )
    assert not missing_routines, f"contract routines are not exported: {missing_routines}"
    assert not missing_methods, f"contract ndarray methods are missing: {missing_methods}"


def test_phase_04_shape_and_join_signatures_match_numpy():
    expected = {
        "flip": "(m, axis=None)",
        "fliplr": "(m)",
        "flipud": "(m)",
        "reshape": "(a, /, shape, order='C', *, copy=None)",
        "tile": "(A, reps)",
        "concat": "(arrays, /, axis=0, out=None, *, dtype=None, casting='same_kind')",
        "concatenate": "(arrays, /, axis=0, out=None, *, dtype=None, casting='same_kind')",
        "hstack": "(tup, *, dtype=None, casting='same_kind')",
        "vstack": "(tup, *, dtype=None, casting='same_kind')",
        "dstack": "(tup)",
        "column_stack": "(tup)",
    }
    for name, signature in expected.items():
        assert str(inspect.signature(getattr(raptors, name))) == signature


def test_phase_04_selection_and_count_signatures_match_numpy():
    expected = {
        "where": "(condition, x=None, y=None, /)",
        "append": "(arr, values, axis=None)",
        "putmask": "(a, /, mask, values)",
        "bincount": "(x, /, weights=None, minlength=0)",
        "count_nonzero": "(a, axis=None, *, keepdims=False)",
    }
    for name, signature in expected.items():
        assert str(inspect.signature(getattr(raptors, name))) == signature


def test_phase_04_copyto_and_ndarray_logical_reduction_signatures_match_numpy():
    assert str(inspect.signature(raptors.copyto)) == (
        "(dst, src, casting='same_kind', where=True)"
    )
    for name in ("all", "any"):
        assert str(inspect.signature(getattr(raptors.Array, name))) == (
            "(self, /, axis=None, out=None, keepdims=False, *, where=True)"
        )


def test_phase_04_default_signatures_match_numpy():
    expected = {
        "arange": "(start_or_stop, /, stop=None, step=1, *, dtype=None, device=None, like=None)",
        "partition": "(a, kth, axis=-1, kind='introselect', order=None)",
        "argpartition": "(a, kth, axis=-1, kind='introselect', order=None)",
        "lexsort": "(keys, axis=-1)",
        "take_along_axis": "(arr, indices, axis=-1)",
        "select": "(condlist, choicelist, default=0)",
        "histogram": "(a, bins=10, range=None, density=None, weights=None)",
        "histogram2d": "(x, y, bins=10, range=None, density=None, weights=None)",
        "histogramdd": "(sample, bins=10, range=None, density=None, weights=None)",
    }
    for name, signature in expected.items():
        assert str(inspect.signature(getattr(raptors, name))) == signature
    for name in ("sort", "argsort"):
        assert (
            inspect.signature(getattr(raptors, name)).parameters["axis"].default
            == inspect.signature(getattr(np, name)).parameters["axis"].default
            == -1
        )
    for name in ("sort", "argsort"):
        assert str(inspect.signature(getattr(raptors.Array, name))) == (
            "(self, /, axis=-1, kind=None, order=None, *, stable=None, descending=None)"
        )
    for name in ("partition", "argpartition"):
        assert str(inspect.signature(getattr(raptors.Array, name))) == (
            "(self, kth, /, axis=-1, kind='introselect', order=None)"
        )
    for name in ("std", "var"):
        assert str(inspect.signature(getattr(raptors.Array, name))) == (
            "(self, /, axis=None, dtype=None, out=None, ddof=0, **kwargs)"
        )


@pytest.mark.parametrize(
    "name,values",
    [
        ("std", [1.0, 2.0, 3.0]),
        ("var", [1.0, 2.0, 3.0]),
        ("nanstd", [1.0, 2.0, np.nan]),
        ("nanvar", [1.0, 2.0, np.nan]),
    ],
)
def test_phase_04_top_level_ddof_signature_and_default_match_numpy(name, values):
    candidate_function = getattr(raptors, name)
    reference_function = getattr(np, name)
    candidate_default = inspect.signature(candidate_function).parameters["ddof"].default
    reference_default = inspect.signature(reference_function).parameters["ddof"].default
    assert candidate_default == reference_default == 0
    assert type(candidate_default) is type(reference_default) is int

    expected_input = np.array(values, dtype=np.float64)
    actual_input = raptors.array(values, dtype=raptors.float64)
    for kwargs in ({}, {"ddof": 1}):
        expected = reference_function(expected_input, **kwargs)
        actual = candidate_function(actual_input, **kwargs)
        assert actual.dtype.name == expected.dtype.name
        assert float(actual) == float(expected)


def test_phase_04_top_level_signature_shapes_and_documented_defaults_match_contract():
    contract_path = Path(__file__).resolve().parents[3] / "compat" / "raptors-0.4.json"
    contract = json.loads(contract_path.read_text())
    routine_names = sorted(
        {name for names in contract["scope"]["families"].values() for name in names}
    )
    signature_audit = contract["evidence"]["signature_audit"]
    expected_mismatches = set(signature_audit["top_level_mismatch_names"])
    observed_mismatches = set()
    array_method_names = contract["scope"]["array_methods"]
    concrete_defaults = {
        "keepdims": False,
        "initial": None,
        "where": Ellipsis,
        "mean": None,
        "correction": None,
        "descending": None,
    }

    assert len(routine_names) == signature_audit["top_level_count"] == 127
    for name in routine_names:
        reference = inspect.signature(getattr(np, name))
        candidate = inspect.signature(getattr(raptors, name))
        assert list(candidate.parameters) == list(reference.parameters), name
        for parameter_name, reference_parameter in reference.parameters.items():
            candidate_parameter = candidate.parameters[parameter_name]
            assert candidate_parameter.kind == reference_parameter.kind, (
                name,
                parameter_name,
            )
            if candidate_parameter.default == reference_parameter.default:
                continue

            observed_mismatches.add(name)
            if repr(reference_parameter.default) == "<no value>":
                assert parameter_name in concrete_defaults, (name, parameter_name)
                assert candidate_parameter.default is concrete_defaults[parameter_name], (
                    name,
                    parameter_name,
                    candidate_parameter.default,
                )
            else:
                assert (name, parameter_name) == ("eye", "dtype")
                assert candidate_parameter.default is None

    assert observed_mismatches == expected_mismatches
    assert len(observed_mismatches) == signature_audit["top_level_mismatches"]
    assert len(array_method_names) == signature_audit["array_method_count"] == 26
    for name in array_method_names:
        assert inspect.signature(getattr(raptors.Array, name)) == inspect.signature(
            getattr(np.ndarray, name)
        ), name


def test_phase_04_ndarray_positional_only_signatures_match_numpy():
    expected = {
        "put": "(self, indices, values, /, mode='raise')",
        "repeat": "(self, repeats, /, axis=None)",
        "searchsorted": "(self, v, /, side='left', sorter=None)",
        "swapaxes": "(self, axis1, axis2, /)",
        "take": "(self, indices, /, axis=None, out=None, mode='raise')",
    }
    for name, signature in expected.items():
        assert str(inspect.signature(getattr(raptors.Array, name))) == signature


def test_phase_04_arg_reduction_keepdims_is_keyword_only():
    expected = {
        "argmax": "(a, axis=None, out=None, *, keepdims=False)",
        "argmin": "(a, axis=None, out=None, *, keepdims=False)",
        "nanargmax": "(a, axis=None, out=None, *, keepdims=False)",
        "nanargmin": "(a, axis=None, out=None, *, keepdims=False)",
    }
    for name, signature in expected.items():
        assert str(inspect.signature(getattr(raptors, name))) == signature


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_atleast_functions_preserve_identity_when_rank_is_sufficient(rank):
    values = np.arange(6, dtype=np.int32).reshape(1, 2, 3)
    source = raptors.array(values.tolist(), dtype=raptors.int32)
    result = getattr(raptors, f"atleast_{rank}d")(source)

    assert result is source
    assert_array_matches(getattr(np, f"atleast_{rank}d")(values), result)


@pytest.mark.parametrize(
    "name,values,expected_shape,source_index,result_index",
    [
        ("atleast_1d", 7, (1,), (), (0,)),
        ("atleast_2d", [1, 2, 3], (1, 3), (0,), (0, 0)),
        ("atleast_3d", [[1, 2, 3], [4, 5, 6]], (2, 3, 1), (0, 0), (0, 0, 0)),
    ],
)
def test_atleast_rank_increase_returns_a_sharing_view(
    name, values, expected_shape, source_index, result_index
):
    source = raptors.array(values, dtype=raptors.int32)
    expected = getattr(np, name)(np.asarray(values, dtype=np.int32))
    result = getattr(raptors, name)(source)

    assert result is not source
    assert tuple(result.shape) == expected_shape
    assert_array_matches(expected, result)
    result[result_index] = 91
    assert source[source_index] == 91


def test_atleast_multiple_inputs_return_tuple_and_preserve_each_input_identity():
    first = raptors.array([[1, 2], [3, 4]], dtype=raptors.int16)
    second = raptors.array([5, 6], dtype=raptors.int16)

    first_result, second_result = raptors.atleast_2d(first, second)

    assert first_result is first
    assert second_result is not second
    assert_array_matches(np.atleast_2d(np.array([1, 2, 3, 4], dtype=np.int16).reshape(2, 2)), first_result)
    assert_array_matches(np.atleast_2d(np.array([5, 6], dtype=np.int16)), second_result)
    second_result[0, 0] = 73
    assert second[0] == 73


def test_histogram_density_none_matches_numpy_default():
    values = np.array([0.1, 0.2, 0.8, 0.9])
    expected = np.histogram(values, bins=2, density=None)
    actual = raptors.histogram(raptors.array(values.tolist()), bins=2, density=None)
    assert_array_matches(expected[0], actual[0])
    assert_array_matches(expected[1], actual[1])


def test_histogram_bins_none_is_not_confused_with_the_default():
    one_dimensional = np.array([0.1, 0.2, 0.8, 0.9])
    two_dimensional = np.column_stack((one_dimensional, one_dimensional))
    cases = (
        (np.histogram, raptors.histogram, (one_dimensional,)),
        (np.histogram2d, raptors.histogram2d, (one_dimensional, one_dimensional)),
        (np.histogramdd, raptors.histogramdd, (two_dimensional,)),
    )
    for reference, candidate, args in cases:
        with pytest.raises(TypeError):
            reference(*args, bins=None)
        with pytest.raises(TypeError):
            candidate(*args, bins=None)


def test_copyto_where_none_and_ndarray_reduction_masks_match_numpy():
    values = np.array([True, False])
    mask = np.array([False, True])
    source = raptors.array(values.tolist(), dtype=raptors.bool_)
    native_mask = raptors.array(mask.tolist(), dtype=raptors.bool_)
    assert source.all(where=native_mask) == values.all(where=mask)
    assert source.any(where=native_mask) == values.any(where=mask)
    with pytest.raises(TypeError):
        source.all(None, None, False, native_mask)
    with pytest.raises(TypeError):
        source.any(None, None, False, native_mask)

    expected = np.zeros(2, dtype=np.int64)
    np.copyto(expected, np.array([4, 5]), where=None)
    actual = raptors.zeros(2, dtype=raptors.int64)
    source_values = raptors.array([4, 5], dtype=raptors.int64)
    raptors.copyto(actual, source_values, where=None)
    assert_array_matches(expected, actual)

    positional_mask_expected = np.zeros(2, dtype=np.int64)
    np.copyto(positional_mask_expected, np.array([4, 5]), "same_kind", mask)
    positional_mask_actual = raptors.zeros(2, dtype=raptors.int64)
    raptors.copyto(positional_mask_actual, source_values, "same_kind", native_mask)
    assert_array_matches(positional_mask_expected, positional_mask_actual)


def test_reduction_where_call_forms_and_explicit_none_match_numpy():
    values = np.array([1, 2, 3], dtype=np.int64)
    mask = np.array([True, False, True])
    source = raptors.array(values.tolist(), dtype=raptors.int64)
    native_mask = raptors.array(mask.tolist(), dtype=raptors.bool_)

    expected = np.sum(values, None, None, None, False, 0, mask)
    actual = raptors.sum(source, None, None, None, False, 0, native_mask)
    assert int(actual) == int(expected)
    assert int(raptors.sum(source, where=None)) == int(np.sum(values, where=None))

    for reference, candidate in (
        (np.mean, raptors.mean),
        (np.var, raptors.var),
    ):
        with pytest.raises(TypeError):
            reference(values, where=None)
        with pytest.raises(TypeError):
            candidate(source, where=None)

    with pytest.raises(ValueError):
        np.min(values, where=None)
    with pytest.raises(ValueError):
        raptors.min(source, where=None)


@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (2, 3)])
@pytest.mark.parametrize("dtype", [raptors.bool_, raptors.int16, raptors.float32, raptors.complex128])
def test_ones_matches_numpy_for_numeric_dtypes_and_empty_shapes(shape, dtype):
    expected = np.ones(shape, dtype=dtype.name)
    actual = raptors.ones(shape, dtype=dtype)
    assert_array_matches(expected, actual)


@pytest.mark.parametrize("order", ["C", "F"])
def test_ones_preserves_requested_memory_order(order):
    expected = np.ones((2, 3, 4), dtype=np.float64, order=order)
    actual = raptors.ones((2, 3, 4), dtype=raptors.float64, order=order)
    assert_array_matches(expected, actual)


def test_cpu_device_keyword_is_explicit_and_unsupported_dispatch_is_rejected():
    assert_array_matches(np.ones((2, 3)), raptors.ones((2, 3), device="cpu"))
    assert_array_matches(np.full((2,), 7), raptors.full((2,), 7, device="cpu"))
    assert_array_matches(np.eye(3), raptors.eye(3, device="cpu"))
    assert_array_matches(np.arange(5), raptors.arange(5, device="cpu"))
    assert_array_matches(np.linspace(0, 1, 4), raptors.linspace(0, 1, 4, device="cpu"))
    assert_array_matches(
        np.ones((2, 3), dtype=np.int16),
        raptors.ones_like(raptors.array([[1, 2, 3], [4, 5, 6]], dtype=raptors.int16), device="cpu"),
    )

    with pytest.raises(ValueError, match="cpu.*device"):
        raptors.ones((2,), device="gpu")
    with pytest.raises(TypeError, match="like-based dispatch"):
        raptors.ones((2,), like=raptors.array([1, 2], dtype=raptors.int64))


def test_full_infers_numeric_scalar_dtype_and_broadcasts_values():
    integer = raptors.full((2, 3), 7)
    assert_array_matches(np.full((2, 3), 7), integer)

    row = raptors.array([1, 2, 3], dtype=raptors.int16)
    broadcast = raptors.full((2, 3), row)
    assert_array_matches(np.full((2, 3), np.array([1, 2, 3], dtype=np.int16)), broadcast)

    scalar = raptors.full((), 2.5, dtype=raptors.float32)
    assert_array_matches(np.full((), 2.5, dtype=np.float32), scalar)


def test_full_rejects_unbroadcastable_fill_value():
    with pytest.raises(ValueError):
        raptors.full((2, 3), [1, 2])


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ((5,), {}),
        ((2, 9, 3), {}),
        ((5, -2, -2), {}),
        ((0.0, 1.0, 0.25), {}),
        ((0, 5, 2), {"dtype": raptors.float32}),
        ((2**53 + 1, 2**53 + 4), {}),
        ((2**63, 2**63 + 3), {}),
    ],
)
def test_arange_matches_numpy(args, kwargs):
    expected = np.arange(*args, **{
        key: getattr(np, value.name) if isinstance(value, type(raptors.float32)) else value
        for key, value in kwargs.items()
    })
    actual = raptors.arange(*args, **kwargs)
    assert_array_matches(expected, actual)


@pytest.mark.parametrize("args", [(0, 5, 0), (0.0, 5.0, 0.0)])
def test_arange_rejects_zero_step(args):
    with pytest.raises(ValueError, match="step must not be zero"):
        raptors.arange(*args)


@pytest.mark.parametrize(
    "shape,k",
    [((3, 3), 0), ((2, 4), 1), ((4, 2), -1), ((0, 3), 0)],
)
def test_eye_matches_numpy(shape, k):
    rows, columns = shape
    expected = np.eye(rows, columns, k=k, dtype=np.float32)
    actual = raptors.eye(rows, columns, k=k, dtype=raptors.float32)
    assert_array_matches(expected, actual)


def test_identity_and_eye_fortran_order():
    expected = np.eye(3, dtype=np.complex64, order="F")
    actual = raptors.eye(3, dtype=raptors.complex64, order="F")
    assert_array_matches(expected, actual)
    assert tuple(actual.strides) == expected.strides
    assert_array_matches(np.identity(3, dtype=np.int8), raptors.identity(3, dtype=raptors.int8))


@pytest.mark.parametrize("args", [(-1,), (2, -1)])
def test_eye_rejects_negative_dimensions(args):
    with pytest.raises(ValueError, match="negative dimensions"):
        raptors.eye(*args)


@pytest.mark.parametrize(
    "name,reference",
    [
        ("sum", np.sum),
        ("prod", np.prod),
        ("min", np.min),
        ("max", np.max),
        ("any", np.any),
        ("all", np.all),
    ],
)
@pytest.mark.parametrize("axis,keepdims", [(None, False), (0, False), (-1, True), ((0, 2), True)])
def test_reductions_match_numpy(name, reference, axis, keepdims):
    values = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    candidate = getattr(raptors, name)(
        raptors.array(values.tolist(), dtype=raptors.int16),
        axis=axis,
        keepdims=keepdims,
    )
    expected = reference(values, axis=axis, keepdims=keepdims)
    if np.ndim(expected) == 0:
        expected_scalar = np.asarray(expected)
        assert candidate.dtype.name == expected_scalar.dtype.name
        assert candidate == expected_scalar[()]
    else:
        assert_array_matches(expected, candidate)


def test_sum_and_min_support_initial_where_and_out():
    values = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
    mask = np.array([[True, False, True], [False, True, False]])
    candidate_values = raptors.array(values.tolist(), dtype=raptors.int16)
    candidate_mask = raptors.array(mask.tolist(), dtype=raptors.bool_)
    expected_out = np.zeros(2, dtype=np.int64)
    candidate_out = raptors.zeros(2, dtype=raptors.int64)

    expected = np.sum(values, axis=1, where=mask, initial=10, out=expected_out)
    actual = raptors.sum(
        candidate_values,
        axis=1,
        where=candidate_mask,
        initial=10,
        out=candidate_out,
    )
    assert actual is candidate_out
    assert_array_matches(expected, candidate_out)

    expected_min = np.min(values, axis=1, where=mask, initial=100)
    actual_min = raptors.min(candidate_values, axis=1, where=candidate_mask, initial=100)
    assert_array_matches(expected_min, actual_min)


@pytest.mark.parametrize("name,reference", [("cumsum", np.cumsum), ("cumprod", np.cumprod)])
@pytest.mark.parametrize("axis", [None, 0, -1])
def test_cumulative_reductions_match_numpy(name, reference, axis):
    values = np.arange(1, 13, dtype=np.int16).reshape(3, 4)
    candidate = getattr(raptors, name)(
        raptors.array(values.tolist(), dtype=raptors.int16), axis=axis
    )
    expected = reference(values, axis=axis)
    assert_array_matches(expected, candidate)


def test_cumulative_reductions_support_dtype_and_out():
    values = np.arange(1, 5, dtype=np.int8)
    expected_out = np.zeros(4, dtype=np.float32)
    candidate_out = raptors.zeros(4, dtype=raptors.float32)
    expected = np.cumsum(values, dtype=np.float32, out=expected_out)
    actual = raptors.cumsum(
        raptors.array(values.tolist(), dtype=raptors.int8),
        dtype=raptors.float32,
        out=candidate_out,
    )
    assert actual is candidate_out
    assert_array_matches(expected, candidate_out)


@pytest.mark.parametrize("axis", [0, 1, -1, None])
def test_concatenate_matches_numpy(axis):
    left = np.arange(6, dtype=np.int16).reshape(2, 3)
    right = (np.arange(6, dtype=np.float32) + 10).reshape(2, 3)
    candidate = raptors.concatenate(
        [raptors.array(left.tolist(), dtype=raptors.int16),
         raptors.array(right.tolist(), dtype=raptors.float32)],
        axis=axis,
    )
    assert_array_matches(np.concatenate([left, right], axis=axis), candidate)


@pytest.mark.parametrize("axis", [0, 1, -1, None])
def test_int32_concatenate_copies_transposed_inputs_in_logical_order(axis):
    values = np.arange(24, dtype=np.int32).reshape(4, 6)
    transposed = values.T
    candidate_input = raptors.transpose(
        raptors.array(values.tolist(), dtype=raptors.int32)
    )
    actual = raptors.concatenate([candidate_input, candidate_input], axis=axis)
    expected = np.concatenate([transposed, transposed], axis=axis)
    assert_array_matches(expected, actual)
    assert actual.flags.c_contiguous == expected.flags.c_contiguous
    assert actual.flags.f_contiguous == expected.flags.f_contiguous


def test_concatenate_supports_alias_dtype_casting_and_out():
    values = [
        raptors.array([1, 2], dtype=raptors.int16),
        raptors.array([3, 4], dtype=raptors.int16),
    ]
    expected_out = np.zeros(4, dtype=np.float32)
    candidate_out = raptors.zeros(4, dtype=raptors.float32)
    actual = raptors.concat(values, dtype=raptors.float32, out=candidate_out)
    assert actual is candidate_out
    assert_array_matches(np.concatenate([np.array([1, 2], dtype=np.int16),
                                         np.array([3, 4], dtype=np.int16)],
                                        dtype=np.float32), candidate_out)
    assert_array_matches(expected_out + np.array([1, 2, 3, 4], dtype=np.float32), candidate_out)
    with pytest.raises(TypeError, match="cannot cast"):
        raptors.concatenate(values, dtype=raptors.float16, casting="safe")


@pytest.mark.parametrize("axis", [0, 1, -1, -2])
def test_stack_matches_numpy(axis):
    left = np.arange(6, dtype=np.int16).reshape(2, 3)
    right = left + 10
    candidate = raptors.stack(
        [raptors.array(left.tolist(), dtype=raptors.int16),
         raptors.array(right.tolist(), dtype=raptors.int16)],
        axis=axis,
    )
    assert_array_matches(np.stack([left, right], axis=axis), candidate)


def test_stack_empty_axes_and_shape_errors():
    with pytest.raises(ValueError, match="same shape"):
        raptors.stack([raptors.ones((2,)), raptors.ones((3,))])
    with pytest.raises(ValueError, match="axis"):
        raptors.stack([raptors.ones(())], axis=1)


@pytest.mark.parametrize("indices", [2, [1, 4], [-1]])
def test_split_matches_numpy_and_returns_shared_views(indices):
    original = raptors.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=raptors.int16)
    expected = np.split(np.arange(8, dtype=np.int16).reshape(4, 2), indices)
    actual = raptors.split(original, indices)
    assert len(actual) == len(expected)
    for expected_part, actual_part in zip(expected, actual):
        assert_array_matches(expected_part, actual_part)
    if actual[0].shape[0] > 0:
        actual[0][0, 0] = 99
        assert original[0, 0] == 99


def test_array_split_and_split_errors():
    values = np.arange(10, dtype=np.int32)
    candidate = raptors.array(values.tolist(), dtype=raptors.int32)
    actual = raptors.array_split(candidate, 3)
    expected = np.array_split(values, 3)
    assert len(actual) == len(expected)
    for expected_part, actual_part in zip(expected, actual):
        assert_array_matches(expected_part, actual_part)
    with pytest.raises(ValueError, match="equal division"):
        raptors.split(candidate, 3)
    with pytest.raises(ValueError, match="larger than 0"):
        raptors.array_split(candidate, 0)


@pytest.mark.parametrize("order", ["C", "F", "A"])
def test_reshape_ravel_and_flatten_match_numpy(order):
    values = np.arange(24, dtype=np.int32).reshape(2, 3, 4).transpose(1, 0, 2)
    candidate = raptors.array(values.tolist(), dtype=raptors.int32)
    assert_array_matches(np.reshape(values, (4, 6), order=order),
                         raptors.reshape(candidate, (4, 6), order=order))
    assert_array_matches(np.ravel(values, order=order), raptors.ravel(candidate, order=order))
    assert_array_matches(np.ndarray.flatten(values, order=order),
                         raptors.flatten(candidate, order=order))


def test_reshape_and_transpose_preserve_views_when_possible():
    original = raptors.array([[0, 1, 2], [3, 4, 5]], dtype=raptors.int16)
    reshaped = raptors.reshape(original, (3, 2))
    reshaped[0, 0] = 77
    assert original[0, 0] == 77
    transposed = raptors.transpose(original)
    transposed[1, 0] = 88
    assert original[0, 1] == 88


@pytest.mark.parametrize("axes", [None, (2, 0, 1), (-1, 0, 1)])
def test_transpose_axes_match_numpy(axes):
    values = np.arange(24).reshape(2, 3, 4)
    candidate = raptors.array(values.tolist(), dtype=raptors.int64)
    assert_array_matches(np.transpose(values, axes=axes), raptors.transpose(candidate, axes=axes))


@pytest.mark.parametrize("axis", [None, 0, -1, (0, 2)])
def test_squeeze_matches_numpy(axis):
    values = np.arange(6).reshape(1, 2, 1, 3, 1)
    candidate = raptors.array(values.tolist(), dtype=raptors.int64)
    assert_array_matches(np.squeeze(values, axis=axis), raptors.squeeze(candidate, axis=axis))


@pytest.mark.parametrize("axis", [0, -1, (0, 2), (-1, 0)])
def test_expand_dims_matches_numpy(axis):
    values = np.arange(6).reshape(2, 3)
    candidate = raptors.array(values.tolist(), dtype=raptors.int64)
    assert_array_matches(np.expand_dims(values, axis=axis), raptors.expand_dims(candidate, axis))


@pytest.mark.parametrize(
    "name,args",
    [
        ("swapaxes", (0, 2)),
        ("moveaxis", ((0, 2), (2, 0))),
        ("rollaxis", (2, 0)),
    ],
)
def test_axis_rearrangement_matches_numpy(name, args):
    values = np.arange(24).reshape(2, 3, 4)
    candidate = raptors.array(values.tolist(), dtype=raptors.int64)
    expected = getattr(np, name)(values, *args)
    actual = getattr(raptors, name)(candidate, *args)
    assert_array_matches(expected, actual)


@pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 2)])
def test_flip_matches_numpy(axis):
    values = np.arange(24).reshape(2, 3, 4)
    candidate = raptors.array(values.tolist(), dtype=raptors.int64)
    assert_array_matches(np.flip(values, axis=axis), raptors.flip(candidate, axis=axis))


def test_fliplr_and_flipud_match_numpy():
    values = np.arange(12).reshape(3, 4)
    candidate = raptors.array(values.tolist(), dtype=raptors.int64)
    assert_array_matches(np.fliplr(values), raptors.fliplr(candidate))
    assert_array_matches(np.flipud(values), raptors.flipud(candidate))


@pytest.mark.parametrize(
    "axis,repeats",
    [(None, 2), (0, 2), (1, 2), (-1, 2), (None, [1, 0, 2, 1, 0, 1]), (0, [1, 0, 2])],
)
def test_repeat_matches_numpy(axis, repeats):
    values = np.arange(6).reshape(3, 2)
    candidate = raptors.array(values.tolist(), dtype=raptors.int64)
    assert_array_matches(np.repeat(values, repeats, axis=axis),
                         raptors.repeat(candidate, repeats, axis=axis))


@pytest.mark.parametrize("reps", [2, (2,), (2, 1), (2, 1, 3)])
def test_tile_matches_numpy(reps):
    values = np.arange(6).reshape(2, 3)
    candidate = raptors.array(values.tolist(), dtype=raptors.int64)
    assert_array_matches(np.tile(values, reps), raptors.tile(candidate, reps))


def test_shape_routines_cover_empty_dimensions_and_invalid_axes():
    empty = raptors.zeros((2, 0, 3), dtype=raptors.int16)
    assert_array_matches(np.flip(np.zeros((2, 0, 3), dtype=np.int16), axis=1),
                         raptors.flip(empty, axis=1))
    assert_array_matches(np.tile(np.zeros((2, 0, 3), dtype=np.int16), (1, 2, 1)),
                         raptors.tile(empty, (1, 2, 1)))
    with pytest.raises(ValueError, match="size not equal to one"):
        raptors.squeeze(raptors.ones((2, 3)), axis=0)
    with pytest.raises(ValueError, match="repeated axis"):
        raptors.expand_dims(raptors.ones((2,)), axis=(0, 0))


@pytest.mark.parametrize("order", ["C", "F", "K"])
def test_like_constructors_match_numpy_shape_dtype_and_layout(order):
    values = np.arange(24, dtype=np.int16).reshape(2, 3, 4).transpose(1, 0, 2)
    candidate = raptors.transpose(
        raptors.array(np.arange(24, dtype=np.int16).reshape(2, 3, 4).tolist(), dtype=raptors.int16),
        axes=(1, 0, 2),
    )
    for name in ("zeros_like", "ones_like", "empty_like"):
        expected = getattr(np, name)(values, dtype=np.float32, order=order)
        actual = getattr(raptors, name)(candidate, dtype=raptors.float32, order=order)
        assert_array_matches(expected, actual, values=name != "empty_like")
    expected = np.full_like(values, 7, dtype=np.int32, order=order, shape=(2, 3))
    actual = raptors.full_like(candidate, 7, dtype=raptors.int32, order=order, shape=(2, 3))
    assert_array_matches(expected, actual)


def test_like_constructors_infer_dtype_and_full_like_broadcasts():
    values = raptors.array([[1, 2, 3], [4, 5, 6]], dtype=raptors.int16)
    assert_array_matches(np.zeros_like(np.arange(6, dtype=np.int16).reshape(2, 3)),
                         raptors.zeros_like(values))
    assert_array_matches(np.full_like(np.zeros((2, 3), dtype=np.int16), [3, 4, 5]),
                         raptors.full_like(values, [3, 4, 5]))


def test_linspace_matches_numpy_for_axis_broadcast_and_retstep():
    expected, expected_step = np.linspace(-1.0, 2.0, num=7, endpoint=False, retstep=True)
    actual, actual_step = raptors.linspace(-1.0, 2.0, num=7, endpoint=False, retstep=True)
    assert actual.dtype.name == expected.dtype.name
    assert tuple(actual.shape) == expected.shape
    np.testing.assert_allclose([float(actual[i]) for i in range(7)], expected, rtol=0, atol=1e-15)
    assert actual_step == expected_step

    start = np.array([[0.0], [10.0]])
    stop = np.array([[2.0, 4.0, 6.0]])
    expected = np.linspace(start, stop, num=4, axis=1)
    candidate_start = raptors.array(start.tolist(), dtype=raptors.float64)
    candidate_stop = raptors.array(stop.tolist(), dtype=raptors.float64)
    actual = raptors.linspace(candidate_start, candidate_stop, num=4, axis=1)
    assert tuple(actual.shape) == expected.shape
    for i in range(expected.shape[0]):
        for j in range(expected.shape[1]):
            for k in range(expected.shape[2]):
                assert actual[i, j, k] == expected[i, j, k]


def test_linspace_dtype_flooring_and_edge_cases():
    expected = np.linspace(-2.0, 2.0, num=5, dtype=np.int32)
    actual = raptors.linspace(-2, 2, num=5, dtype=raptors.int32)
    assert_array_matches(expected, actual)
    assert_array_matches(np.linspace(3, 9, num=0), raptors.linspace(3, 9, num=0))
    with pytest.raises(ValueError, match="non-negative"):
        raptors.linspace(0, 1, num=-1)


@pytest.mark.parametrize(
    "name,args",
    [
        ("logspace", (0.0, 3.0)),
        ("geomspace", (1.0, 1000.0)),
        ("geomspace", (-1.0, -1000.0)),
    ],
)
def test_logarithmic_and_geometric_spaces_match_numpy(name, args):
    expected = getattr(np, name)(*args, num=4)
    actual = getattr(raptors, name)(*args, num=4)
    assert tuple(actual.shape) == expected.shape
    assert actual.dtype.name == expected.dtype.name
    np.testing.assert_allclose([float(actual[i]) for i in range(4)], expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("name,reference", [("mean", np.mean), ("var", np.var), ("std", np.std), ("ptp", np.ptp)])
@pytest.mark.parametrize("axis,keepdims", [(None, False), (0, False), (-1, True), ((0, 2), True)])
def test_descriptive_reductions_match_numpy(name, reference, axis, keepdims):
    values = np.arange(1, 25, dtype=np.int16).reshape(2, 3, 4)
    candidate = raptors.array(values.tolist(), dtype=raptors.int16)
    expected = reference(values, axis=axis, keepdims=keepdims)
    actual = getattr(raptors, name)(candidate, axis=axis, keepdims=keepdims)
    if np.ndim(expected) == 0:
        assert actual.dtype.name == np.asarray(expected).dtype.name
        assert np.isclose(float(actual), expected)
    else:
        assert tuple(actual.shape) == expected.shape
        assert actual.dtype.name == expected.dtype.name
        for coordinates in np.ndindex(expected.shape):
            assert np.isclose(float(actual[coordinates]), expected[coordinates], rtol=1e-14, atol=1e-14)


def test_float32_axis0_statistics_handle_contiguous_offset_views():
    values = np.array(
        [
            [((row * 17 + column * 13) % 251 - 125) / 31 for column in range(8)]
            for row in range(258)
        ],
        dtype=np.float32,
    )
    owner = raptors.array(values.tolist(), dtype=raptors.float32)
    candidate = owner[1:257]
    expected_values = values[1:257]

    assert_array_matches(np.mean(expected_values, axis=0), raptors.mean(candidate, axis=0))
    assert_array_matches(np.var(expected_values, axis=0), raptors.var(candidate, axis=0))
    assert_array_matches(np.std(expected_values, axis=0), raptors.std(candidate, axis=0))


def test_float32_broadcast_binary_handles_contiguous_offset_views():
    values = np.array(
        [
            [((row * 17 + column * 13) % 251 - 125) / 31 for column in range(8)]
            for row in range(258)
        ],
        dtype=np.float32,
    )
    candidate = raptors.array(values.tolist(), dtype=raptors.float32)[1:257]
    expected_values = values[1:257]
    right_values = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)
    right = raptors.array(right_values.tolist(), dtype=raptors.float32)

    assert_array_matches(
        np.subtract(expected_values, right_values),
        raptors.subtract(candidate, right),
    )
    assert_array_matches(
        np.true_divide(expected_values, right_values),
        raptors.true_divide(candidate, right),
    )


def test_mean_and_variance_support_dtype_mask_out_and_correction():
    values = np.arange(1, 7, dtype=np.int16).reshape(2, 3)
    mask = np.array([[True, False, True], [False, True, True]])
    candidate = raptors.array(values.tolist(), dtype=raptors.int16)
    candidate_mask = raptors.array(mask.tolist(), dtype=raptors.bool_)
    expected_mean_out = np.zeros(2, dtype=np.float32)
    candidate_mean_out = raptors.zeros(2, dtype=raptors.float32)
    expected = np.mean(values, axis=1, where=mask, dtype=np.float32, out=expected_mean_out)
    actual = raptors.mean(
        candidate,
        axis=1,
        where=candidate_mask,
        dtype=raptors.float32,
        out=candidate_mean_out,
    )
    assert actual is candidate_mean_out
    assert_array_matches(expected, candidate_mean_out)

    expected_var = np.var(values, axis=1, where=mask, correction=1)
    actual_var = raptors.var(candidate, axis=1, where=candidate_mask, correction=1)
    for axis in range(2):
        assert np.isclose(float(actual_var[axis]), expected_var[axis])


def test_variance_and_standard_deviation_accept_broadcastable_mean():
    values = np.array([[1.0, 2.0, 4.0], [3.0, 5.0, 8.0]], dtype=np.float64)
    candidate = raptors.array(values.tolist(), dtype=raptors.float64)
    mean = np.mean(values, axis=0)
    candidate_mean = raptors.array(mean.tolist(), dtype=raptors.float64)

    assert_array_matches(
        np.var(values, axis=0, mean=mean),
        raptors.var(candidate, axis=0, mean=candidate_mean),
    )
    assert_array_matches(
        np.std(values, axis=0, mean=mean),
        raptors.std(candidate, axis=0, mean=candidate_mean),
    )

    mask = np.array([[True, False, True], [False, True, True]])
    masked_mean = np.mean(values, axis=0, where=mask, keepdims=True)
    candidate_mask = raptors.array(mask.tolist(), dtype=raptors.bool_)
    candidate_masked_mean = raptors.array(masked_mean.tolist(), dtype=raptors.float64)
    assert_array_matches(
        np.var(values, axis=0, where=mask, mean=masked_mean),
        raptors.var(
            candidate,
            axis=0,
            where=candidate_mask,
            mean=candidate_masked_mean,
        ),
    )

    nan_values = np.array([[1.0, np.nan], [3.0, 5.0], [5.0, 7.0]])
    candidate_nan_values = raptors.array(nan_values.tolist(), dtype=raptors.float64)
    nan_mean = np.nanmean(nan_values, axis=0)
    candidate_nan_mean = raptors.array(nan_mean.tolist(), dtype=raptors.float64)
    assert_array_matches(
        np.nanvar(nan_values, axis=0, mean=nan_mean),
        raptors.nanvar(candidate_nan_values, axis=0, mean=candidate_nan_mean),
    )
    assert_array_matches(
        np.nanstd(nan_values, axis=0, mean=nan_mean),
        raptors.nanstd(candidate_nan_values, axis=0, mean=candidate_nan_mean),
    )

    float32_values = np.array([[1.0, 2.0, 4.0], [3.0, 5.0, 8.0]], dtype=np.float32)
    candidate_float32 = raptors.array(float32_values.tolist(), dtype=raptors.float32)
    float32_mean = np.mean(float32_values, axis=0)
    candidate_float32_mean = raptors.array(float32_mean.tolist(), dtype=raptors.float32)
    assert_array_matches(
        np.var(float32_values, axis=0, mean=float32_mean),
        raptors.var(candidate_float32, axis=0, mean=candidate_float32_mean),
    )
    assert_array_matches(
        np.std(float32_values, axis=0, mean=float32_mean),
        raptors.std(candidate_float32, axis=0, mean=candidate_float32_mean),
    )
    with pytest.raises(ValueError):
        raptors.var(candidate, axis=0, mean=raptors.array([1.0, 2.0]))


def test_empty_and_degrees_of_freedom_reductions_match_numpy_warnings():
    empty = raptors.empty((0,), dtype=raptors.float64)
    expected_mean, expected_warnings = capture_warnings(
        lambda: np.mean(np.empty((0,), dtype=np.float64))
    )
    actual_mean, actual_warnings = capture_warnings(lambda: raptors.mean(empty))
    assert actual_warnings == expected_warnings
    assert actual_warnings == [
        (RuntimeWarning, "Mean of empty slice"),
        (RuntimeWarning, "invalid value encountered in scalar divide"),
    ]
    assert np.isnan(float(expected_mean))
    assert np.isnan(float(actual_mean))

    empty_float32 = raptors.empty((0, 2), dtype=raptors.float32)
    expected_float32_mean, expected_warnings = capture_warnings(
        lambda: np.mean(np.empty((0, 2), dtype=np.float32), axis=0)
    )
    actual_float32_mean, actual_warnings = capture_warnings(
        lambda: raptors.mean(empty_float32, axis=0)
    )
    assert actual_warnings == expected_warnings
    assert_array_matches(expected_float32_mean, actual_float32_mean)

    all_nan = raptors.array([np.nan, np.nan], dtype=raptors.float64)
    expected_nanmean, expected_warnings = capture_warnings(
        lambda: np.nanmean(np.array([np.nan, np.nan]))
    )
    actual_nanmean, actual_warnings = capture_warnings(lambda: raptors.nanmean(all_nan))
    assert actual_warnings == expected_warnings
    assert actual_warnings == [(RuntimeWarning, "Mean of empty slice")]
    assert np.isnan(float(expected_nanmean))
    assert np.isnan(float(actual_nanmean))

    values = np.array([1.0, 2.0], dtype=np.float64)
    candidate = raptors.array(values.tolist(), dtype=raptors.float64)
    expected_var, expected_warnings = capture_warnings(lambda: np.var(values, ddof=2))
    actual_var, actual_warnings = capture_warnings(lambda: raptors.var(candidate, ddof=2))
    assert actual_warnings == expected_warnings
    assert actual_warnings == [
        (RuntimeWarning, "Degrees of freedom <= 0 for slice"),
        (RuntimeWarning, "divide by zero encountered in scalar divide"),
    ]
    assert np.isinf(float(expected_var))
    assert np.isinf(float(actual_var))

    float32_values = np.array([[1.0, 2.0], [3.0, 7.0]], dtype=np.float32)
    candidate_float32 = raptors.array(
        float32_values.tolist(), dtype=raptors.float32
    )
    for name, reference in (("var", np.var), ("std", np.std)):
        expected, expected_warnings = capture_warnings(
            lambda: reference(float32_values, axis=0, ddof=2)
        )
        actual, actual_warnings = capture_warnings(
            lambda: getattr(raptors, name)(candidate_float32, axis=0, ddof=2)
        )
        assert actual_warnings == expected_warnings
        assert_array_matches(expected, actual)

    nan_values = np.array([np.nan, np.nan], dtype=np.float64)
    candidate_nan_values = raptors.array(nan_values.tolist(), dtype=raptors.float64)
    expected_nanvar, expected_warnings = capture_warnings(lambda: np.nanvar(nan_values))
    actual_nanvar, actual_warnings = capture_warnings(lambda: raptors.nanvar(candidate_nan_values))
    assert actual_warnings == expected_warnings
    assert actual_warnings == [(RuntimeWarning, "Degrees of freedom <= 0 for slice.")]
    assert np.isnan(float(expected_nanvar))
    assert np.isnan(float(actual_nanvar))


def test_empty_and_all_nan_quantiles_match_numpy_warnings_and_errors():
    empty_values = np.array([], dtype=np.float64)
    candidate_empty = raptors.empty((0,), dtype=raptors.float64)
    with pytest.raises(IndexError):
        np.quantile(empty_values, 0.5)
    with pytest.raises(IndexError):
        raptors.quantile(candidate_empty, 0.5)

    for name, reference, arguments in (
        ("median", np.median, ()),
        ("nanmedian", np.nanmedian, ()),
        ("nanquantile", np.nanquantile, (0.5,)),
    ):
        expected, expected_warnings = capture_warnings(
            lambda: reference(empty_values, *arguments)
        )
        actual, actual_warnings = capture_warnings(
            lambda: getattr(raptors, name)(candidate_empty, *arguments)
        )
        assert actual_warnings == expected_warnings
        assert actual_warnings
        assert actual_warnings[0] == (RuntimeWarning, "Mean of empty slice")
        assert np.isnan(float(expected))
        assert np.isnan(float(actual))

    empty_matrix = np.empty((2, 0), dtype=np.float64)
    candidate_empty_matrix = raptors.empty((2, 0), dtype=raptors.float64)
    expected, expected_warnings = capture_warnings(lambda: np.median(empty_matrix, axis=1))
    actual, actual_warnings = capture_warnings(
        lambda: raptors.median(candidate_empty_matrix, axis=1)
    )
    assert actual_warnings == expected_warnings
    assert_array_matches(expected, actual)


@pytest.mark.parametrize(
    "name,reference,arguments",
    [
        ("median", np.median, ()),
        ("nanmedian", np.nanmedian, ()),
        ("quantile", np.quantile, (0.5,)),
        ("nanquantile", np.nanquantile, (0.5,)),
        ("percentile", np.percentile, (50.0,)),
        ("nanpercentile", np.nanpercentile, (50.0,)),
    ],
)
def test_quantile_family_honors_overwrite_input(name, reference, arguments):
    values = np.array([8.0, 1.0, 6.0, 2.0, 7.0, 3.0], dtype=np.float64)
    sorted_values = np.sort(values)
    candidate = raptors.array(values.tolist(), dtype=raptors.float64)

    expected = reference(values.copy(), *arguments, overwrite_input=True)
    actual = getattr(raptors, name)(candidate, *arguments, overwrite_input=True)
    if hasattr(actual, "shape"):
        assert_array_matches(expected, actual)
    else:
        assert actual.dtype.name == np.asarray(expected).dtype.name
        assert np.isclose(float(actual), float(expected))
    assert_array_matches(sorted_values, candidate)

    unchanged = raptors.array(values.tolist(), dtype=raptors.float64)
    getattr(raptors, name)(unchanged, *arguments)
    assert_array_matches(values, unchanged)

    all_nan = np.array([np.nan, np.nan], dtype=np.float64)
    candidate_all_nan = raptors.array(all_nan.tolist(), dtype=raptors.float64)
    for name, reference, arguments in (
        ("nanmedian", np.nanmedian, ()),
        ("nanquantile", np.nanquantile, (0.5,)),
    ):
        expected, expected_warnings = capture_warnings(
            lambda: reference(all_nan, *arguments)
        )
        actual, actual_warnings = capture_warnings(
            lambda: getattr(raptors, name)(candidate_all_nan, *arguments)
        )
        assert actual_warnings == expected_warnings
        assert actual_warnings == [(RuntimeWarning, "All-NaN slice encountered")]
        assert np.isnan(float(expected))
        assert np.isnan(float(actual))


@pytest.mark.parametrize("name,reference", [("argmin", np.argmin), ("argmax", np.argmax)])
@pytest.mark.parametrize("axis,keepdims", [(None, False), (0, False), (-1, True)])
def test_arg_reductions_match_numpy(name, reference, axis, keepdims):
    values = np.array([[8, 2, 7], [1, 9, 3]], dtype=np.int16)
    candidate = raptors.array(values.tolist(), dtype=raptors.int16)
    expected = reference(values, axis=axis, keepdims=keepdims)
    actual = getattr(raptors, name)(candidate, axis=axis, keepdims=keepdims)
    if np.ndim(expected) == 0:
        assert actual.dtype.name == np.asarray(expected).dtype.name
        assert int(actual) == int(expected)
    else:
        assert_array_matches(expected, actual)


def test_nonzero_argwhere_flatnonzero_and_count_nonzero_match_numpy():
    values = np.array([[0, 2, 0], [3, 0, 4]], dtype=np.int16)
    candidate = raptors.array(values.tolist(), dtype=raptors.int16)
    expected_coordinates = np.nonzero(values)
    actual_coordinates = raptors.nonzero(candidate)
    assert len(actual_coordinates) == len(expected_coordinates)
    for expected, actual in zip(expected_coordinates, actual_coordinates):
        assert_array_matches(expected, actual)
    assert_array_matches(np.argwhere(values), raptors.argwhere(candidate))
    assert_array_matches(np.flatnonzero(values), raptors.flatnonzero(candidate))
    assert_array_matches(np.count_nonzero(values, axis=1), raptors.count_nonzero(candidate, axis=1))


def test_coordinate_reductions_handle_scalar_and_empty_inputs():
    scalar = raptors.array(5, dtype=raptors.int16)
    expected = np.array(5)
    actual = raptors.argwhere(scalar)
    assert_array_matches(np.argwhere(expected), actual)
    with pytest.raises(ValueError, match="0d arrays is not allowed"):
        raptors.nonzero(scalar)
    empty = raptors.zeros((2, 0), dtype=raptors.float32)
    assert_array_matches(np.argwhere(np.zeros((2, 0))), raptors.argwhere(empty))


@pytest.mark.parametrize(
    "name,left,right",
    [
        ("hstack", [1, 2], [3, 4]),
        ("vstack", [1, 2], [3, 4]),
        ("dstack", [1, 2], [3, 4]),
        ("column_stack", [1, 2], [3, 4]),
    ],
)
def test_stacking_convenience_routines_match_numpy(name, left, right):
    expected = getattr(np, name)([left, right])
    actual = getattr(raptors, name)([left, right])
    assert_array_matches(expected, actual)


def test_stacking_convenience_handles_mixed_dimensions_and_aliases():
    left = np.arange(6).reshape(2, 3)
    row = np.array([7, 8, 9])
    column = np.array([[7], [8]])
    left_candidate = raptors.array(left.tolist(), dtype=raptors.int64)
    row_candidate = raptors.array(row.tolist(), dtype=raptors.int64)
    column_candidate = raptors.array(column.tolist(), dtype=raptors.int64)
    assert_array_matches(np.vstack([left, row]), raptors.vstack([left_candidate, row_candidate]))
    assert_array_matches(np.hstack([left, column]), raptors.hstack([left_candidate, column_candidate]))


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_atleast_dimension_helpers_match_numpy(rank):
    source = raptors.array(5, dtype=raptors.int16)
    expected = getattr(np, f"atleast_{rank}d")(np.array(5, dtype=np.int16))
    actual = getattr(raptors, f"atleast_{rank}d")(source)
    assert_array_matches(expected, actual)
    multiple = getattr(raptors, f"atleast_{rank}d")(source, raptors.array([1, 2]))
    assert isinstance(multiple, tuple)
    assert_array_matches(expected, multiple[0])


@pytest.mark.parametrize(
    "name,values,sections",
    [
        ("vsplit", [[0, 1], [2, 3], [4, 5], [6, 7]], 2),
        ("hsplit", [[0, 1, 2, 3], [4, 5, 6, 7]], 2),
        ("dsplit", np.arange(24).reshape(2, 3, 4).tolist(), 2),
    ],
)
def test_named_split_helpers_match_numpy(name, values, sections):
    expected = getattr(np, name)(np.asarray(values), sections)
    candidate = raptors.array(values, dtype=raptors.int64)
    actual = getattr(raptors, name)(candidate, sections)
    assert len(actual) == len(expected)
    for expected_part, actual_part in zip(expected, actual):
        assert_array_matches(expected_part, actual_part)


def test_where_and_copy_match_numpy_and_preserve_independence():
    condition = np.array([[True], [False]])
    left = np.arange(6, dtype=np.int16).reshape(2, 3)
    right = np.full((1, 3), 9, dtype=np.int16)
    candidate_condition = raptors.array(condition.tolist(), dtype=raptors.bool_)
    candidate_left = raptors.array(left.tolist(), dtype=raptors.int16)
    candidate_right = raptors.array(right.tolist(), dtype=raptors.int16)
    assert_array_matches(
        np.where(condition, left, right),
        raptors.where(candidate_condition, candidate_left, candidate_right),
    )
    for expected, actual in zip(np.nonzero(left), raptors.where(candidate_left)):
        assert_array_matches(expected, actual)
    independent = raptors.copy(candidate_left)
    independent[0, 0] = 100
    assert candidate_left[0, 0] == 0


def test_copyto_broadcast_mask_and_casting():
    expected = np.zeros((2, 3), dtype=np.float32)
    candidate = raptors.zeros((2, 3), dtype=raptors.float32)
    source = np.array([1, 2, 3], dtype=np.int16)
    mask = np.array([[True, False, True], [False, True, False]])
    np.copyto(expected, source, where=mask)
    raptors.copyto(
        candidate,
        raptors.array(source.tolist(), dtype=raptors.int16),
        where=raptors.array(mask.tolist(), dtype=raptors.bool_),
    )
    assert_array_matches(expected, candidate)
    with pytest.raises(TypeError, match="cannot cast"):
        raptors.copyto(raptors.zeros(2, dtype=raptors.int8),
                       raptors.array([1, 2], dtype=raptors.float32),
                       casting="safe")


@pytest.mark.parametrize(
    "axis,mode",
    [(None, "raise"), (0, "raise"), (-1, "raise"), (1, "wrap"), (1, "clip")],
)
def test_take_matches_numpy(axis, mode):
    values = np.arange(12, dtype=np.int16).reshape(3, 4)
    indices = np.array([0, -1, 4, 5], dtype=np.int64) if mode != "raise" else np.array([0, -1])
    if axis == 0 and mode == "raise":
        indices = np.array([0, -1])
    candidate = raptors.array(values.tolist(), dtype=raptors.int16)
    expected = np.take(values, indices, axis=axis, mode=mode)
    actual = raptors.take(candidate, raptors.array(indices.tolist(), dtype=raptors.int64), axis=axis, mode=mode)
    assert_array_matches(expected, actual)


def test_take_out_and_take_along_axis():
    values = np.arange(12, dtype=np.int32).reshape(3, 4)
    indices = np.array([[3, 0], [2, 1], [0, 3]], dtype=np.int64)
    expected = np.take_along_axis(values, indices, axis=1)
    candidate = raptors.array(values.tolist(), dtype=raptors.int32)
    actual = raptors.take_along_axis(
        candidate,
        raptors.array(indices.tolist(), dtype=raptors.int64),
        axis=1,
    )
    assert_array_matches(expected, actual)
    out = raptors.zeros((2, 4), dtype=raptors.int32)
    expected_out = np.zeros((2, 4), dtype=np.int32)
    expected = np.take(values, [2, 0], axis=0, out=expected_out)
    actual = raptors.take(
        candidate,
        raptors.array([2, 0], dtype=raptors.int64),
        axis=0,
        out=out,
    )
    assert actual is out
    assert_array_matches(expected, out)


@pytest.mark.parametrize(
    "axis,right",
    [(None, [6, 7, 8]), (0, [[6, 7, 8]]), (1, [[6], [7]]), (-1, [[6], [7]])],
)
def test_append_matches_numpy(axis, right):
    left = np.arange(6, dtype=np.int16).reshape(2, 3)
    right = np.asarray(right, dtype=np.int16)
    candidate = raptors.array(left.tolist(), dtype=raptors.int16)
    append_values = raptors.array(right.tolist(), dtype=raptors.int16)
    assert_array_matches(np.append(left, right, axis=axis),
                         raptors.append(candidate, append_values, axis=axis))


@pytest.mark.parametrize("name,reference", [("tril", np.tril), ("triu", np.triu)])
@pytest.mark.parametrize("k", [-1, 0, 1])
def test_triangular_helpers_match_numpy(name, reference, k):
    values = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    candidate = raptors.array(values.tolist(), dtype=raptors.int16)
    assert_array_matches(reference(values, k=k), getattr(raptors, name)(candidate, k=k))


def test_fill_diagonal_matches_numpy_and_wrap():
    expected = np.zeros((3, 3), dtype=np.int16)
    candidate = raptors.zeros((3, 3), dtype=raptors.int16)
    np.fill_diagonal(expected, [1, 2, 3])
    raptors.fill_diagonal(candidate, [1, 2, 3])
    assert_array_matches(expected, candidate)

    expected_wrap = np.zeros((5, 3), dtype=np.int16)
    candidate_wrap = raptors.zeros((5, 3), dtype=raptors.int16)
    np.fill_diagonal(expected_wrap, 7, wrap=True)
    raptors.fill_diagonal(candidate_wrap, 7, wrap=True)
    assert_array_matches(expected_wrap, candidate_wrap)


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_sort_and_argsort_match_numpy(axis):
    values = np.array([[9, 2, 7], [1, 8, 3]], dtype=np.int16)
    candidate = raptors.array(values.tolist(), dtype=raptors.int16)
    assert_array_matches(np.sort(values, axis=axis), raptors.sort(candidate, axis=axis))
    assert_array_matches(np.argsort(values, axis=axis), raptors.argsort(candidate, axis=axis))


@pytest.mark.parametrize("name,np_dtype", [("int32", np.int32), ("int16", np.int16)])
@pytest.mark.parametrize("axis", [0, 1])
def test_sort_preserves_fortran_layout(name, np_dtype, axis):
    values = np.asfortranarray(np.array([[9, 2, 7], [1, 8, 3]], dtype=np_dtype))
    candidate = raptors.transpose(
        raptors.array(values.T.tolist(), dtype=getattr(raptors, name))
    )
    expected = np.sort(values, axis=axis)
    actual = raptors.sort(candidate, axis=axis)
    assert_array_matches(expected, actual)
    assert actual.flags.c_contiguous == expected.flags.c_contiguous
    assert actual.flags.f_contiguous == expected.flags.f_contiguous


def test_int32_sort_matches_numpy_for_negative_stride_input():
    values = np.array([[9, 2, 7, 4], [1, 8, 3, 6], [5, 0, 11, 10]], dtype=np.int32)
    reversed_values = np.flip(values, axis=1)
    candidate = raptors.flip(
        raptors.array(values.tolist(), dtype=raptors.int32), axis=1
    )
    assert_array_matches(
        np.sort(reversed_values, axis=0),
        raptors.sort(candidate, axis=0),
    )


def test_sort_and_argsort_support_descending_and_default_last_axis():
    values = np.array([[7, 1, 5], [4, 9, 2]], dtype=np.int16)
    candidate = raptors.array(values.tolist(), dtype=raptors.int16)
    assert_array_matches(np.sort(values), raptors.sort(candidate))
    assert_array_matches(np.sort(values, descending=True), raptors.sort(candidate, descending=True))
    assert_array_matches(np.argsort(values, descending=True), raptors.argsort(candidate, descending=True))
    expected_method = values.copy()
    expected_method.sort(descending=True)
    method_array = raptors.array(values.tolist(), dtype=raptors.int16)
    method_array.sort(descending=True)
    assert_array_matches(expected_method, method_array)


def test_partition_and_argpartition_match_partition_invariants():
    values = np.array([8, 1, 4, 3, 9, 2], dtype=np.int32)
    candidate = raptors.array(values.tolist(), dtype=raptors.int32)
    expected = np.partition(values, 2)
    actual = raptors.partition(candidate, 2)
    assert int(actual[2]) == int(expected[2])
    assert all(int(value) <= int(actual[2]) for value in actual[:2])
    assert all(int(value) >= int(actual[2]) for value in actual[3:])
    actual_indices = raptors.argpartition(candidate, 2)
    permuted = [int(values[int(index)]) for index in actual_indices]
    assert sorted(permuted) == sorted(int(value) for value in values)
    assert permuted[2] == int(expected[2])
    assert all(value <= permuted[2] for value in permuted[:2])
    assert all(value >= permuted[2] for value in permuted[3:])


@pytest.mark.parametrize("side", ["left", "right"])
def test_searchsorted_and_digitize_match_numpy(side):
    values = np.array([1, 3, 5, 8], dtype=np.int32)
    needles = np.array([[0, 3], [6, 9]], dtype=np.int64)
    candidate = raptors.array(values.tolist(), dtype=raptors.int32)
    actual = raptors.searchsorted(
        candidate, raptors.array(needles.tolist(), dtype=raptors.int64), side=side
    )
    assert_array_matches(np.searchsorted(values, needles, side=side), actual)
    bins = np.array([1, 3, 5, 8], dtype=np.int32)
    data = np.array([0, 1, 2, 3, 5, 9], dtype=np.int32)
    assert_array_matches(
        np.digitize(data, bins, right=(side == "left")),
        raptors.digitize(
            raptors.array(data.tolist(), dtype=raptors.int32),
            raptors.array(bins.tolist(), dtype=raptors.int32),
            right=(side == "left"),
        ),
    )


def test_searchsorted_sorter_and_descending_digitize():
    values = np.array([30, 10, 20], dtype=np.int32)
    sorter = np.argsort(values)
    candidate = raptors.array(values.tolist(), dtype=raptors.int32)
    assert int(raptors.searchsorted(candidate, 20, sorter=raptors.array(sorter.tolist(), dtype=raptors.int64))) == int(
        np.searchsorted(values, 20, sorter=sorter)
    )
    descending = np.array([8, 5, 3, 1], dtype=np.int32)
    data = np.array([9, 8, 6, 5, 2, 1, 0], dtype=np.int32)
    for right in (False, True):
        actual = raptors.digitize(
            raptors.array(data.tolist(), dtype=raptors.int32),
            raptors.array(descending.tolist(), dtype=raptors.int32),
            right=right,
        )
        assert_array_matches(np.digitize(data, descending, right=right), actual)


def test_bincount_unique_and_numeric_set_routines():
    values = np.array([0, 1, 1, 3, 2, 1], dtype=np.int64)
    candidate = raptors.array(values.tolist(), dtype=raptors.int64)
    assert_array_matches(np.bincount(values, minlength=5), raptors.bincount(candidate, minlength=5))
    weights = np.arange(values.size, dtype=np.float32)
    assert_array_matches(
        np.bincount(values, weights=weights),
        raptors.bincount(candidate, weights=raptors.array(weights.tolist(), dtype=raptors.float32)),
    )

    values = np.array([3, 1, 3, 2, 1], dtype=np.int16)
    candidate = raptors.array(values.tolist(), dtype=raptors.int16)
    expected_unique = np.unique(values, return_index=True, return_inverse=True, return_counts=True)
    actual_unique = raptors.unique(
        candidate, return_index=True, return_inverse=True, return_counts=True
    )
    for expected, actual in zip(expected_unique, actual_unique):
        assert_array_matches(expected, actual)
    left = np.array([1, 2, 2, 5], dtype=np.int16)
    right = np.array([2, 3, 5], dtype=np.int16)
    left_candidate = raptors.array(left.tolist(), dtype=raptors.int16)
    right_candidate = raptors.array(right.tolist(), dtype=raptors.int16)
    assert_array_matches(
        np.intersect1d(left, right),
        raptors.intersect1d(left_candidate, right_candidate),
    )
    assert_array_matches(
        np.union1d(left, right), raptors.union1d(left_candidate, right_candidate)
    )
    assert_array_matches(
        np.setdiff1d(left, right), raptors.setdiff1d(left_candidate, right_candidate)
    )
    assert_array_matches(
        np.setxor1d(left, right), raptors.setxor1d(left_candidate, right_candidate)
    )
    assert_array_matches(np.isin(left, right), raptors.isin(left_candidate, right_candidate))


@pytest.mark.parametrize("density", [False, True])
def test_histogram_matches_numpy_with_integer_edges_and_weights(density):
    values = np.array([0.2, 0.8, 1.1, 1.9, 2.4, 4.0], dtype=np.float64)
    bins = np.array([0, 1, 2, 4], dtype=np.int64)
    weights = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    expected = np.histogram(values, bins=bins, weights=weights, density=density)
    actual = raptors.histogram(
        raptors.array(values.tolist(), dtype=raptors.float64),
        raptors.array(bins.tolist(), dtype=raptors.int64),
        weights=raptors.array(weights.tolist(), dtype=raptors.float64),
        density=density,
    )
    assert_array_matches(expected[0], actual[0])
    assert_array_matches(expected[1], actual[1])


def test_put_and_put_along_axis_match_numpy_mutation():
    expected = np.arange(6, dtype=np.int16).reshape(2, 3)
    candidate = raptors.array(expected.tolist(), dtype=raptors.int16)
    np.put(expected, [0, 0, -1], [7, 8], mode="wrap")
    raptors.put(candidate, raptors.array([0, 0, -1], dtype=raptors.int64), [7, 8], mode="wrap")
    assert_array_matches(expected, candidate)

    expected = np.arange(12, dtype=np.int32).reshape(3, 4)
    indices = np.array([[3, 0], [2, 1], [0, 3]], dtype=np.int64)
    values = np.array([[10, 11], [12, 13], [14, 15]], dtype=np.int32)
    candidate = raptors.array(expected.tolist(), dtype=raptors.int32)
    np.put_along_axis(expected, indices, values, axis=1)
    raptors.put_along_axis(
        candidate,
        raptors.array(indices.tolist(), dtype=raptors.int64),
        raptors.array(values.tolist(), dtype=raptors.int32),
        axis=1,
    )
    assert_array_matches(expected, candidate)


def test_compress_extract_and_select_match_numpy():
    values = np.arange(12, dtype=np.int16).reshape(3, 4)
    condition = np.array([True, False, True])
    candidate = raptors.array(values.tolist(), dtype=raptors.int16)
    r_condition = raptors.array(condition.tolist(), dtype=raptors.bool_)
    assert_array_matches(np.compress(condition, values, axis=0), raptors.compress(r_condition, candidate, axis=0))
    flat_condition = np.array([True, False] * 6)
    assert_array_matches(
        np.extract(flat_condition, values),
        raptors.extract(raptors.array(flat_condition.tolist(), dtype=raptors.bool_), candidate),
    )

    condition_1 = np.array([[True], [False]])
    condition_2 = np.array([[False, True, False]])
    choice_1 = np.array([[1], [2]], dtype=np.int16)
    choice_2 = np.array([[3, 4, 5]], dtype=np.int16)
    expected = np.select([condition_1, condition_2], [choice_1, choice_2], default=-1)
    actual = raptors.select(
        [raptors.array(condition_1.tolist(), dtype=raptors.bool_),
         raptors.array(condition_2.tolist(), dtype=raptors.bool_)],
        [raptors.array(choice_1.tolist(), dtype=raptors.int16),
         raptors.array(choice_2.tolist(), dtype=raptors.int16)],
        default=-1,
    )
    assert_array_matches(expected, actual)


def test_putmask_and_place_keep_distinct_value_repetition_rules():
    expected_masked = np.arange(6, dtype=np.int16)
    expected_placed = np.arange(6, dtype=np.int16)
    mask = np.array([True, False, True, False, True, False])
    np.putmask(expected_masked, mask, [10, 20])
    np.place(expected_placed, mask, [10, 20])
    masked = raptors.array(np.arange(6).tolist(), dtype=raptors.int16)
    placed = raptors.array(np.arange(6).tolist(), dtype=raptors.int16)
    r_mask = raptors.array(mask.tolist(), dtype=raptors.bool_)
    raptors.putmask(masked, r_mask, [10, 20])
    raptors.place(placed, r_mask, [10, 20])
    assert_array_matches(expected_masked, masked)
    assert_array_matches(expected_placed, placed)


def test_roll_permute_dims_and_broadcast_shapes():
    values = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    candidate = raptors.array(values.tolist(), dtype=raptors.int32)
    for shift, axis in [(3, None), ((1, -1), (0, 2)), (2, -1), (100, 1)]:
        assert_array_matches(np.roll(values, shift, axis=axis), raptors.roll(candidate, shift, axis=axis))
    assert_array_matches(
        np.permute_dims(values, (2, 0, 1)),
        raptors.permute_dims(candidate, (2, 0, 1)),
    )
    assert raptors.broadcast_shapes((0, 3), (1, 3)) == (0, 3)
    assert raptors.broadcast_shapes((2, 1, 4), (3, 4)) == (2, 3, 4)
    with pytest.raises(ValueError):
        raptors.broadcast_shapes((2, 3), (4, 3))
    assert_array_matches(np.permute_dims(values), raptors.permute_dims(candidate))
    assert_array_matches(np.reshape(values, (4, 6)), raptors.reshape(candidate, (4, 6), copy=True))


def test_average_and_quantile_family_match_numpy():
    values = np.array([[1, 8, 3, 4], [7, 2, 9, 6]], dtype=np.float32)
    candidate = raptors.array(values.tolist(), dtype=raptors.float32)
    weights = np.array([1, 2, 3, 4], dtype=np.float32)
    r_weights = raptors.array(weights.tolist(), dtype=raptors.float32)
    expected_scalar = np.average(values)
    actual_scalar = raptors.average(candidate)
    assert type(actual_scalar).__name__ == "Float32Scalar"
    assert float(actual_scalar) == float(expected_scalar)
    assert actual_scalar.dtype.name == expected_scalar.dtype.name
    expected_average = np.average(values, axis=1, weights=weights, returned=True)
    actual_average = raptors.average(candidate, axis=1, weights=r_weights, returned=True)
    assert_array_matches(expected_average[0], actual_average[0])
    assert_array_matches(expected_average[1], actual_average[1])
    assert_array_matches(np.median(values, axis=1), raptors.median(candidate, axis=1))

    for method in [
        "inverted_cdf", "averaged_inverted_cdf", "closest_observation",
        "interpolated_inverted_cdf", "hazen", "weibull", "linear",
        "median_unbiased", "normal_unbiased", "lower", "higher", "midpoint", "nearest",
    ]:
        expected = np.quantile(values, [0.15, 0.5, 0.82], axis=1, method=method)
        actual = raptors.quantile(
            candidate,
            [0.15, 0.5, 0.82],
            axis=1,
            method=method,
        )
        assert_array_matches(expected, actual, values=False)
        actual_values = np.array(
            [[float(actual[row, column]) for column in range(actual.shape[1])]
             for row in range(actual.shape[0])],
            dtype=expected.dtype,
        )
        np.testing.assert_allclose(expected, actual_values, rtol=1e-12, atol=1e-12)
    assert_array_matches(
        np.percentile(values, [25, 75], axis=0),
        raptors.percentile(candidate, [25, 75], axis=0),
    )


def test_weighted_quantile_and_average_errors():
    values = np.array([1, 10, 20, 40], dtype=np.int16)
    weights = np.array([1, 1, 6, 2], dtype=np.int64)
    expected = np.quantile(values, [0.25, 0.5, 0.9], weights=weights, method="inverted_cdf")
    actual = raptors.quantile(
        raptors.array(values.tolist(), dtype=raptors.int16),
        [0.25, 0.5, 0.9],
        weights=raptors.array(weights.tolist(), dtype=raptors.int64),
        method="inverted_cdf",
    )
    assert_array_matches(expected, actual)
    with pytest.raises(ZeroDivisionError):
        raptors.average(
            raptors.array([1, 2], dtype=raptors.int16),
            weights=raptors.array([0, 0], dtype=raptors.float64),
        )


@pytest.mark.parametrize(
    "name,reference",
    [
        ("nanmean", np.nanmean),
        ("nanvar", np.nanvar),
        ("nanstd", np.nanstd),
        ("nanmin", np.nanmin),
        ("nanmax", np.nanmax),
        ("nanmedian", np.nanmedian),
        ("nanargmin", np.nanargmin),
        ("nanargmax", np.nanargmax),
    ],
)
@pytest.mark.parametrize("axis,keepdims", [(None, False), (0, True), (1, False)])
def test_nan_reductions_match_numpy(name, reference, axis, keepdims):
    values = np.array([[np.nan, 4.0, 8.0], [3.0, np.nan, 2.0]])
    candidate = raptors.array(values.tolist(), dtype=raptors.float64)
    with np.errstate(all="ignore"):
        expected = reference(values, axis=axis, keepdims=keepdims)
    with np.errstate(all="ignore"):
        actual = getattr(raptors, name)(candidate, axis=axis, keepdims=keepdims)
    if not hasattr(actual, "shape"):
        assert actual.dtype.name == expected.dtype.name
        assert actual == expected
    else:
        assert_array_matches(expected, actual)


@pytest.mark.parametrize("name,reference", [("nanmin", np.nanmin), ("nanmax", np.nanmax)])
def test_nan_extrema_support_initial_where_and_all_nan_warning(name, reference):
    values = np.array([[np.nan, 2.0], [3.0, 4.0]], dtype=np.float64)
    mask = np.array([[True, False], [False, True]])
    candidate = raptors.array(values.tolist(), dtype=raptors.float64)
    candidate_mask = raptors.array(mask.tolist(), dtype=raptors.bool_)

    expected = reference(values, axis=1, where=mask, initial=0.0)
    actual = getattr(raptors, name)(
        candidate,
        axis=1,
        where=candidate_mask,
        initial=0.0,
    )
    assert_array_matches(expected, actual)

    all_nan = np.array([np.nan, np.nan], dtype=np.float64)
    candidate_all_nan = raptors.array(all_nan.tolist(), dtype=raptors.float64)
    with pytest.warns(RuntimeWarning, match="All-NaN slice encountered"):
        expected_nan = reference(all_nan)
    with pytest.warns(RuntimeWarning, match="All-NaN slice encountered"):
        actual_nan = getattr(raptors, name)(candidate_all_nan)
    assert actual_nan.dtype.name == expected_nan.dtype.name
    assert np.isnan(float(actual_nan))

    with pytest.raises(ValueError):
        getattr(raptors, name)(
            candidate,
            axis=1,
            where=raptors.zeros((2, 2), dtype=raptors.bool_),
        )

    expected_initial = reference(np.array([1.0, 2.0]), initial=0.0)
    actual_initial = getattr(raptors, name)(
        raptors.array([1.0, 2.0], dtype=raptors.float64),
        initial=0.0,
    )
    assert actual_initial.dtype.name == expected_initial.dtype.name
    assert float(actual_initial) == float(expected_initial)


def test_nan_quantile_family_skips_nan_and_preserves_quantile_shape():
    values = np.array([[np.nan, 1.0, 9.0], [2.0, np.nan, 8.0]], dtype=np.float32)
    candidate = raptors.array(values.tolist(), dtype=raptors.float32)
    for name, reference, quantiles in [
        ("nanquantile", np.nanquantile, [0.25, 0.75]),
        ("nanpercentile", np.nanpercentile, [25, 75]),
    ]:
        expected = reference(values, quantiles, axis=1, keepdims=True)
        actual = getattr(raptors, name)(candidate, quantiles, axis=1, keepdims=True)
        assert_array_matches(expected, actual)
    expected = np.nanmedian(values)
    actual = raptors.nanmedian(candidate)
    assert actual.dtype.name == expected.dtype.name
    assert float(actual) == float(expected)


def test_broadcast_and_diagonal_views_match_numpy_alias_and_writeability():
    values = np.arange(12, dtype=np.int16).reshape(3, 4)
    candidate = raptors.array(values.tolist(), dtype=raptors.int16)
    broadcast_expected = np.broadcast_to(values[:1, :], (3, 4))
    broadcast_actual = raptors.broadcast_to(candidate[:1, :], (3, 4))
    assert_array_matches(broadcast_expected, broadcast_actual)
    assert not broadcast_actual.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        broadcast_actual[0, 0] = 10

    diagonals = [
        (0, 0, 1),
        (1, 0, 1),
        (-1, 1, 0),
    ]
    for offset, axis1, axis2 in diagonals:
        expected = np.diagonal(values, offset=offset, axis1=axis1, axis2=axis2)
        actual = raptors.diagonal(candidate, offset=offset, axis1=axis1, axis2=axis2)
        assert_array_matches(expected, actual)
        assert not actual.flags.writeable

    assert_array_matches(np.diag(np.arange(4), k=-1), raptors.diag(raptors.arange(4), k=-1))
    assert_array_matches(np.diagflat(values, k=1), raptors.diagflat(candidate, k=1))
    expected_arrays = np.broadcast_arrays(np.arange(3).reshape(3, 1), np.arange(4))
    actual_arrays = raptors.broadcast_arrays(
        raptors.array([[0], [1], [2]], dtype=raptors.int64),
        raptors.arange(4),
    )
    assert len(expected_arrays) == len(actual_arrays) == 2
    for expected, actual in zip(expected_arrays, actual_arrays):
        assert_array_matches(expected, actual)


def test_phase_04_views_keep_storage_alive_and_preserve_mutation_contracts():
    values = np.arange(12, dtype=np.int32).reshape(3, 4)
    source = raptors.array(values.tolist(), dtype=raptors.int32)
    reshaped = raptors.reshape(source, (4, 3))
    flattened_view = raptors.ravel(source)
    flattened_copy = raptors.flatten(source)

    del source
    gc.collect()

    reshaped[0, 0] = 91
    flattened_view[1] = 92
    flattened_copy[2] = 93

    expected = values.copy()
    expected[0, 0] = 91
    expected[0, 1] = 92
    assert_array_matches(expected.reshape((4, 3)), reshaped)
    assert flattened_copy[2] == 93
    assert reshaped[0, 2] == values[0, 2]


def test_phase_04_broadcast_and_diagonal_views_retain_their_owner():
    values = np.arange(9, dtype=np.int32).reshape(3, 3)
    source = raptors.array(values.tolist(), dtype=raptors.int32)
    diagonal = raptors.diagonal(source)
    broadcast = raptors.broadcast_to(source[0, :], (3, 3))

    del source
    gc.collect()

    assert_array_matches(np.diagonal(values), diagonal)
    assert_array_matches(np.broadcast_to(values[0, :], (3, 3)), broadcast)
    assert not diagonal.flags.writeable
    assert not broadcast.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        diagonal[0] = 100
    with pytest.raises(ValueError, match="read-only"):
        broadcast[0, 0] = 100


def test_phase_04_overlap_assignment_snapshots_source_and_copy_results_do_not_alias():
    expected = np.array([1, 2, 3, 4], dtype=np.int32)
    candidate = raptors.array(expected.tolist(), dtype=raptors.int32)
    expected[1:] = expected[:-1]
    candidate[1:] = candidate[:-1]
    assert_array_matches(expected, candidate)

    original = raptors.array([4, 1, 3, 2], dtype=raptors.int32)
    joined = raptors.concatenate((original, original))
    ordered = raptors.sort(original)
    joined[0] = 99
    ordered[0] = 98
    assert_array_matches(np.array([4, 1, 3, 2], dtype=np.int32), original)


@pytest.mark.parametrize("mode", ["raise", "wrap", "clip"])
def test_choose_modes_and_broadcasting_match_numpy(mode):
    indices = np.array([[0], [1], [-1]], dtype=np.int64)
    choices = [
        np.array([10, 20, 30, 40], dtype=np.int16),
        np.array([1, 2, 3, 4], dtype=np.int16),
    ]
    reference_indices = indices.copy()
    if mode == "raise":
        with pytest.raises(ValueError):
            np.choose(reference_indices, choices, mode=mode)
        with pytest.raises(ValueError):
            raptors.choose(
                raptors.array(indices.tolist(), dtype=raptors.int64),
                [raptors.array(choice.tolist(), dtype=raptors.int16) for choice in choices],
                mode=mode,
            )
        return
    expected = np.choose(reference_indices, choices, mode=mode)
    actual = raptors.choose(
        raptors.array(indices.tolist(), dtype=raptors.int64),
        [raptors.array(choice.tolist(), dtype=raptors.int16) for choice in choices],
        mode=mode,
    )
    assert_array_matches(expected, actual)


@pytest.mark.parametrize(
    "axis,obj",
    [(None, [1, 1, 5]), (0, 1), (1, slice(1, None, 2)), (-1, [-1, 0])],
)
def test_delete_matches_numpy_for_flat_axis_slice_and_repeated_indices(axis, obj):
    values = np.arange(12, dtype=np.int16).reshape(3, 4)
    candidate = raptors.array(values.tolist(), dtype=raptors.int16)
    expected = np.delete(values, obj, axis=axis)
    actual = raptors.delete(candidate, obj, axis=axis)
    assert_array_matches(expected, actual)


def test_lexsort_and_named_unique_results_match_numpy():
    keys = [
        np.array([1, 0, 1, 0, 1]),
        np.array([2, 2, 1, 1, 1]),
    ]
    actual_keys = [
        raptors.array(key.tolist(), dtype=raptors.int64)
        for key in keys
    ]
    assert_array_matches(np.lexsort(keys), raptors.lexsort(actual_keys))

    values = np.array([3.0, np.nan, 1.0, 3.0, np.nan, 2.0])
    candidate = raptors.array(values.tolist(), dtype=raptors.float64)
    for name in ("unique_all", "unique_counts", "unique_inverse"):
        expected = getattr(np, name)(values)
        actual = getattr(raptors, name)(candidate)
        assert type(actual).__name__ == type(expected).__name__
        for field in expected._fields:
            assert_array_matches(getattr(expected, field), getattr(actual, field))
    assert_array_matches(np.unique_values(values), raptors.unique_values(candidate))


def test_broadcast_arrays_keeps_numpy_2_5_writeable_compatibility():
    source = raptors.array([[1], [2]], dtype=raptors.int16)
    broadcast, = raptors.broadcast_arrays(source)
    assert broadcast.flags.writeable


def test_ndarray_reduction_methods_dispatch_to_phase_04_routines():
    values = np.array([[1.0, 2.0, 3.0], [6.0, 5.0, 4.0]], dtype=np.float32)
    candidate = raptors.array(values.tolist(), dtype=raptors.float32)
    for name in ("sum", "prod", "min", "max", "mean", "var", "std", "any", "all"):
        expected = getattr(values, name)(axis=1, keepdims=True)
        actual = getattr(candidate, name)(axis=1, keepdims=True)
        assert_array_matches(expected, actual)
    expected = values.argmin(axis=1)
    assert_array_matches(expected, candidate.argmin(axis=1))
    expected = values.argmax(axis=1)
    assert_array_matches(expected, candidate.argmax(axis=1))
    assert float(candidate.mean()) == float(values.mean())


@pytest.mark.parametrize("density", [False, True])
def test_multidimensional_histograms_match_numpy(density):
    x = np.array([0.1, 0.3, 0.8, 1.2, 1.7])
    y = np.array([0.8, 0.1, 0.4, 1.5, 1.8])
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bins = [np.array([0.0, 0.5, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])]
    expected_2d = np.histogram2d(x, y, bins=bins, weights=weights, density=density)
    actual_2d = raptors.histogram2d(
        raptors.array(x.tolist(), dtype=raptors.float64),
        raptors.array(y.tolist(), dtype=raptors.float64),
        bins=[
            raptors.array(bins[0].tolist(), dtype=raptors.float64),
            raptors.array(bins[1].tolist(), dtype=raptors.float64),
        ],
        weights=raptors.array(weights.tolist(), dtype=raptors.float64),
        density=density,
    )
    for expected, actual in zip(expected_2d, actual_2d):
        assert_array_matches(expected, actual)

    samples = np.column_stack((x, y))
    expected_dd = np.histogramdd(samples, bins=bins, weights=weights, density=density)
    actual_dd = raptors.histogramdd(
        raptors.array(samples.tolist(), dtype=raptors.float64),
        bins=[
            raptors.array(bins[0].tolist(), dtype=raptors.float64),
            raptors.array(bins[1].tolist(), dtype=raptors.float64),
        ],
        weights=raptors.array(weights.tolist(), dtype=raptors.float64),
        density=density,
    )
    assert_array_matches(expected_dd[0], actual_dd[0])
    for expected, actual in zip(expected_dd[1], actual_dd[1]):
        assert_array_matches(expected, actual)


@pytest.mark.parametrize(
    "values,obj,axis",
    [([8, 9], 1, None), ([8, 9], [1, 3], None), ([8, 9], 1, 0), ([7, 8, 9], 1, 1)],
)
def test_insert_matches_numpy_for_flat_and_axis_insertions(values, obj, axis):
    source = np.arange(6, dtype=np.int16).reshape(3, 2) if axis is not None else np.arange(5, dtype=np.int16)
    candidate = raptors.array(source.tolist(), dtype=raptors.int16)
    reference = np.insert(source, obj, values, axis=axis)
    actual = raptors.insert(candidate, obj, values, axis=axis)
    assert_array_matches(reference, actual)


def test_block_builds_nested_numeric_arrays_and_rejects_bad_layouts():
    left = np.arange(6, dtype=np.int16).reshape(2, 3)
    right = np.arange(6, 12, dtype=np.float32).reshape(2, 3)
    bottom_left = np.arange(12, 18, dtype=np.int16).reshape(2, 3)
    bottom_right = np.arange(18, 24, dtype=np.float32).reshape(2, 3)
    expected = np.block([[left, right], [bottom_left, bottom_right]])
    actual = raptors.block(
        [
            [raptors.array(left.tolist(), dtype=raptors.int16), raptors.array(right.tolist(), dtype=raptors.float32)],
            [raptors.array(bottom_left.tolist(), dtype=raptors.int16), raptors.array(bottom_right.tolist(), dtype=raptors.float32)],
        ]
    )
    assert_array_matches(expected, actual)
    assert_array_matches(np.block([1, 2, 3]), raptors.block([1, 2, 3]))
    line = raptors.array([0, 1, 2], dtype=raptors.int64)
    assert_array_matches(np.block([np.arange(3), np.arange(3)]), raptors.block([line, line]))
    with pytest.raises(ValueError):
        raptors.block([])
    with pytest.raises(ValueError):
        raptors.block(
            [[raptors.ones((2, 2)), raptors.ones((3, 2))]]
        )


def test_numeric_equality_and_closeness_routines_match_numpy():
    left = np.array([[1.0], [np.nan], [np.inf]], dtype=np.float64)
    right = np.array([[1.0, 1.000001], [np.nan, np.nan], [np.inf, -np.inf]])
    r_left = raptors.array(left.tolist(), dtype=raptors.float64)
    r_right = raptors.array(right.tolist(), dtype=raptors.float64)
    assert raptors.array_equal(r_left, raptors.array(left.tolist(), dtype=raptors.float64), equal_nan=True)
    assert raptors.array_equal([np.nan], [np.nan], equal_nan=True)
    assert not raptors.array_equal([np.nan], [np.nan])
    assert raptors.array_equiv([1, 2, 3], [[1, 2, 3]])
    assert not raptors.array_equiv([1, 2], [[1, 2, 3]])
    for equal_nan in (False, True):
        expected = np.isclose(left, right, equal_nan=equal_nan)
        actual = raptors.isclose(r_left, r_right, equal_nan=equal_nan)
        assert_array_matches(expected, actual)
        assert raptors.allclose(r_left, r_right, equal_nan=equal_nan) == np.allclose(
            left, right, equal_nan=equal_nan
        )
    assert bool(raptors.isclose(1 + 1j, 1 + 1.00000001j)[()]) == bool(
        np.isclose(1 + 1j, 1 + 1.00000001j)
    )
    assert raptors.array_equal(
        [1, 2], raptors.array([1, 2], dtype=raptors.int16)
    )


@pytest.mark.parametrize("dtype_name", ["longdouble", "clongdouble"])
def test_closeness_preserves_extended_precision(dtype_name):
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("platform long double has no precision beyond float64")
    dtype = np.dtype(dtype_name)
    candidate_dtype = raptors.DType(dtype_name)
    left_values = np.array([2**53 + 1], dtype=dtype)
    right_values = np.array([2**53 + 2], dtype=dtype)
    left = raptors.array([2**53 + 1], dtype=candidate_dtype)
    right = raptors.array([2**53 + 2], dtype=candidate_dtype)
    expected = np.isclose(left_values, right_values, rtol=0, atol=0)
    actual = raptors.isclose(left, right, rtol=0, atol=0)
    assert_array_matches(expected, actual)
    assert raptors.allclose(left, right, rtol=0, atol=0) == np.allclose(
        left_values, right_values, rtol=0, atol=0
    )


def test_unique_axis_sorted_option_and_public_alias_identity():
    values = np.array([[1, 2, 1, 3], [4, 5, 4, 6]], dtype=np.int16)
    candidate = raptors.array(values.tolist(), dtype=raptors.int16)
    expected = np.unique(values, axis=1, return_index=True, return_inverse=True, return_counts=True)
    actual = raptors.unique(
        candidate,
        axis=1,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    for expected_value, actual_value in zip(expected, actual):
        assert_array_matches(expected_value, actual_value)
    expected_unsorted = np.unique([3, 1, 2, 1], sorted=False)
    actual_unsorted = raptors.unique([3, 1, 2, 1], sorted=False)
    assert sorted(int(value) for value in expected_unsorted) == sorted(
        int(actual_unsorted[index]) for index in range(actual_unsorted.shape[0])
    )
    axis_values = np.array([[4, 2, 4, 1], [8, 5, 8, 6]], dtype=np.int16)
    axis_candidate = raptors.array(axis_values.tolist(), dtype=raptors.int16)
    axis_unique, axis_inverse, axis_counts = raptors.unique(
        axis_candidate,
        axis=1,
        return_inverse=True,
        return_counts=True,
        sorted=False,
    )
    assert_array_matches(np.asfortranarray([[4, 2, 1], [8, 5, 6]], dtype=np.int16), axis_unique)
    assert_array_matches(np.array([0, 1, 0, 2]), axis_inverse)
    assert_array_matches(np.array([2, 1, 1]), axis_counts)
    assert raptors.amin is raptors.min
    assert raptors.amax is raptors.max
    assert raptors.concat is raptors.concatenate


def test_ndarray_routine_counterparts_preserve_numpy_return_and_mutation_behavior():
    values = np.arange(12, dtype=np.int16).reshape(3, 4)
    candidate = raptors.array(values.tolist(), dtype=raptors.int16)
    assert_array_matches(values.ravel(), candidate.ravel())
    assert_array_matches(values.flatten(), candidate.flatten())
    assert_array_matches(values[None, :, :].squeeze(), candidate[None, :, :].squeeze())
    assert_array_matches(values.swapaxes(0, 1), candidate.swapaxes(0, 1))
    assert_array_matches(values.diagonal(), candidate.diagonal())
    assert_array_matches(values.argsort(axis=0), candidate.argsort(axis=0))
    assert_array_matches(values.argpartition(2, axis=1), candidate.argpartition(2, axis=1))
    assert_array_matches(values.take([2, 0], axis=1), candidate.take([2, 0], axis=1))
    assert_array_matches(values.repeat(2, axis=0), candidate.repeat(2, axis=0))
    assert_array_matches(values.ravel().searchsorted([2, 7]), candidate.ravel().searchsorted([2, 7]))

    expected_sorted = values.copy()
    expected_sorted.sort(axis=1)
    candidate.sort(axis=1)
    assert_array_matches(expected_sorted, candidate)
    expected_partition = expected_sorted.copy()
    expected_partition.partition(1, axis=1)
    candidate.partition(1, axis=1)
    actual_partition = np.array(
        [[int(candidate[row, column]) for column in range(candidate.shape[1])] for row in range(candidate.shape[0])],
        dtype=np.int16,
    )
    np.testing.assert_array_equal(
        np.sort(actual_partition, axis=1), np.sort(expected_partition, axis=1)
    )
    expected_put = expected_partition.copy()
    expected_put.put([0, 3], [90, 91])
    candidate.put([0, 3], [90, 91])
    assert_array_matches(expected_put, candidate)


def test_ndarray_reductions_forward_where_masks():
    values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    mask = np.array([[True, False, True], [False, True, False]])
    candidate = raptors.array(values.tolist(), dtype=raptors.float64)
    r_mask = raptors.array(mask.tolist(), dtype=raptors.bool_)
    for method in ("sum", "prod", "min", "max", "mean", "std", "var", "any", "all"):
        kwargs = {"initial": 100.0 if method == "min" else -100.0} if method in ("min", "max") else {}
        expected = getattr(values, method)(axis=1, where=mask, **kwargs)
        actual = getattr(candidate, method)(axis=1, where=r_mask, **kwargs)
        assert_array_matches(expected, actual)
