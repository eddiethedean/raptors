"""Pinned NumPy 2.5.3 differential coverage for the 0.3 ufunc preview."""

import json
from pathlib import Path

import numpy as np
import pytest
import raptors

from harness import assert_array_matches


CONTRACT_PATH = Path(__file__).parents[3] / "compat" / "raptors-0.3.json"
CONTRACT = json.loads(CONTRACT_PATH.read_text())
UFUNC_CONTRACTS = CONTRACT["scope"]["ufunc_objects"]
NUMERIC_CODES = set("?bBhHiIlLqQefd gFDG".replace(" ", ""))
NUMERIC_PAIR_DTYPES = tuple(
    np.dtype(dtype)
    for dtype in (
        np.bool_,
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
        np.longdouble,
        np.clongdouble,
    )
)


def _numpy_numeric_signatures(ufunc):
    """Keep only the bool, integer, floating, and complex oracle loops."""
    result = []
    for signature in ufunc.types:
        inputs, outputs = signature.split("->")
        if set(inputs + outputs) <= NUMERIC_CODES:
            result.append(signature)
    return result


def _inputs_for_signature(name, signature):
    input_codes = signature.split("->", 1)[0]
    inputs = []
    for position, code in enumerate(input_codes):
        dtype = np.dtype(code)
        if dtype.kind == "b":
            values = [True, False]
        elif dtype.kind in "iu":
            if name in {"left_shift", "right_shift", "bitwise_left_shift", "bitwise_right_shift"} and position:
                values = [1, 2]
            elif name in {"power", "pow"} and position:
                values = [2, 3]
            elif name in {"ldexp"} and position:
                values = [1, 2]
            elif name in {"remainder", "mod", "fmod", "floor_divide", "divmod", "gcd", "lcm"}:
                values = [7, 5] if position == 0 else [3, 2]
            else:
                values = [2, 5]
        elif dtype.kind == "f":
            if name in {"arccos", "arcsin", "arctanh"}:
                values = [0.25, 0.5]
            elif name == "arccosh":
                values = [1.25, 2.0]
            elif name in {"floor_divide", "remainder", "mod", "fmod", "divmod"}:
                values = [7.5, 5.25] if position == 0 else [3.0, 2.0]
            elif name in {"power", "pow"} and position:
                values = [2.0, 3.0]
            else:
                values = [0.5, 1.25]
        elif dtype.kind == "c":
            values = [0.5 + 0.25j, 1.25 - 0.5j]
        else:  # The signature filter should make this unreachable.
            raise AssertionError((name, signature, code))
        inputs.append((np.array(values, dtype=dtype), raptors.array(values, dtype=raptors.DType(code))))
    return inputs


def _binary_dtype_pair_cases():
    cases = []
    for contract in UFUNC_CONTRACTS:
        reference = getattr(np, contract["name"])
        if reference.nin != 2:
            continue
        for left_dtype in NUMERIC_PAIR_DTYPES:
            for right_dtype in NUMERIC_PAIR_DTYPES:
                try:
                    reference.resolve_dtypes(
                        (left_dtype, right_dtype) + (None,) * reference.nout,
                        casting="same_kind",
                    )
                except (TypeError, ValueError):
                    continue
                cases.append(
                    pytest.param(
                        contract["name"],
                        left_dtype,
                        right_dtype,
                        id=f"{contract['name']}-{left_dtype.name}-{right_dtype.name}",
                    )
                )
    return cases


def _pair_input_values(name, position, dtype):
    if dtype.kind == "b":
        return [True, False] if position == 0 else [True, True]
    if dtype.kind in "iu":
        if name in {"left_shift", "right_shift"} and position == 1:
            return [1, 2]
        if name in {"remainder", "mod", "fmod", "floor_divide", "divmod", "gcd", "lcm"}:
            return [7, 5] if position == 0 else [3, 2]
        if name in {"power", "pow", "float_power"} and position == 1:
            return [2, 3]
        if name == "ldexp" and position == 1:
            return [1, 2]
        return [2, 5]
    if dtype.kind == "f":
        if name in {"remainder", "mod", "fmod", "floor_divide", "divmod"}:
            return [7.5, 5.25] if position == 0 else [3.0, 2.0]
        if name in {"power", "pow", "float_power"} and position == 1:
            return [2.0, 3.0]
        if name == "ldexp" and position == 1:
            return [1, 2]
        return [0.5, 1.25] if position == 0 else [1.5, 2.0]
    if dtype.kind == "c":
        return [0.5 + 0.25j, 1.25 - 0.5j]
    raise AssertionError((name, position, dtype))


def _raptors_dtype_for_numpy(dtype):
    if dtype.kind == "f" and dtype.char == "g":
        return raptors.DType("longdouble")
    if dtype.kind == "c" and dtype.char == "G":
        return raptors.DType("clongdouble")
    return raptors.DType(dtype.name)


def _as_comparable(value):
    name = type(value).__name__
    if name == "LongDoubleScalar":
        return np.longdouble(str(value))
    if name == "ComplexLongDoubleScalar":
        return complex(value)
    return value.item() if hasattr(value, "item") else value


def _assert_ufunc_result_matches(reference, candidate, *, exact=False):
    if isinstance(reference, tuple):
        assert isinstance(candidate, tuple)
        assert len(reference) == len(candidate)
        for expected, actual in zip(reference, candidate):
            _assert_ufunc_result_matches(expected, actual, exact=exact)
        return

    if np.isscalar(reference):
        assert candidate.dtype.name == np.asarray(reference).dtype.name
        expected = _as_comparable(reference)
        actual = _as_comparable(candidate)
        if exact or np.asarray(reference).dtype.kind not in "fc":
            assert actual == expected
        else:
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
            if expected == 0:
                assert bool(np.signbit(actual)) == bool(np.signbit(expected))
        return

    assert_array_matches(reference, candidate, values=False)
    kind = reference.dtype.kind
    if exact or kind not in "fc":
        assert_array_matches(reference, candidate)
        return

    itemsize = reference.dtype.itemsize
    if kind == "f":
        rtol, atol = (2e-3, 2e-5) if itemsize <= 2 else (2e-6, 2e-7)
        if itemsize > 4:
            rtol, atol = 2e-14, 2e-15
    else:
        rtol, atol = (2e-6, 2e-7) if itemsize <= 8 else (2e-14, 2e-15)

    for index in np.ndindex(reference.shape) if reference.shape else [()]:
        expected = _as_comparable(reference[index])
        actual = _as_comparable(candidate[index])
        if np.isnan(expected):
            assert np.isnan(actual), f"expected NaN at {index}"
        elif np.isinf(expected):
            assert actual == expected, f"infinity mismatch at {index}"
        else:
            np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
            if expected == 0:
                if kind == "c":
                    assert bool(np.signbit(actual.real)) == bool(np.signbit(expected.real))
                    assert bool(np.signbit(actual.imag)) == bool(np.signbit(expected.imag))
                else:
                    assert bool(np.signbit(actual)) == bool(np.signbit(expected)), (
                        f"signed zero mismatch at {index}"
                    )


@pytest.mark.parametrize("contract", UFUNC_CONTRACTS, ids=lambda item: item["name"])
def test_registered_ufunc_aliases_and_metadata_match_numpy(contract):
    reference = getattr(np, contract["name"])
    canonical = getattr(raptors, contract["name"])

    assert type(canonical).__name__ == "ufunc"
    assert canonical.__name__ == reference.__name__ == contract["name"]
    assert canonical.nin == reference.nin
    assert canonical.nout == reference.nout
    assert canonical.nargs == reference.nargs
    expected_types = _numpy_numeric_signatures(reference)
    assert canonical.ntypes == len(expected_types)
    assert canonical.types == expected_types, (canonical.types, expected_types)
    assert canonical.identity == reference.identity
    assert canonical.signature == reference.signature

    for public_name in contract["public_names"]:
        actual_alias = getattr(raptors, public_name)
        expected_alias = getattr(np, public_name)
        assert actual_alias is canonical
        assert expected_alias is reference


def test_ufunc_inventory_has_all_contract_names_and_no_numpy_runtime_fallback():
    public_names = [name for item in UFUNC_CONTRACTS for name in item["public_names"]]
    assert len(public_names) == len(set(public_names)) == 101
    assert all(hasattr(raptors, name) for name in public_names)
    assert raptors.ufunc.__module__ == "raptors"


@pytest.mark.parametrize(
    "contract,signature",
    [
        (contract, signature)
        for contract in UFUNC_CONTRACTS
        for signature in dict.fromkeys(
            _numpy_numeric_signatures(getattr(np, contract["name"]))
        )
    ],
    ids=lambda item: item["name"] if isinstance(item, dict) else item,
)
def test_every_numeric_ufunc_loop_matches_numpy(contract, signature):
    name = contract["name"]
    oracle = getattr(np, name)
    candidate = getattr(raptors, name)
    inputs = _inputs_for_signature(name, signature)
    expected_inputs = [pair[0] for pair in inputs]
    actual_inputs = [pair[1] for pair in inputs]

    with np.errstate(all="ignore"), raptors.errstate(all="ignore"):
        expected = oracle(*expected_inputs)
        actual = candidate(*actual_inputs)
    _assert_ufunc_result_matches(expected, actual)


@pytest.mark.parametrize(
    "name,left_dtype,right_dtype",
    _binary_dtype_pair_cases(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_every_valid_binary_numeric_dtype_pair_matches_numpy(name, left_dtype, right_dtype):
    input_values = [
        _pair_input_values(name, position, dtype)
        for position, dtype in enumerate((left_dtype, right_dtype))
    ]
    expected_inputs = [
        np.array(values, dtype=dtype)
        for values, dtype in zip(input_values, (left_dtype, right_dtype))
    ]
    actual_inputs = [
        raptors.array(
            values,
            dtype=_raptors_dtype_for_numpy(dtype),
        )
        for values, dtype in zip(input_values, (left_dtype, right_dtype))
    ]

    with np.errstate(all="ignore"), raptors.errstate(all="ignore"):
        expected = getattr(np, name)(*expected_inputs)
        actual = getattr(raptors, name)(*actual_inputs)
    _assert_ufunc_result_matches(expected, actual)


def test_ufunc_call_controls_broadcast_where_and_out_match_numpy():
    expected_left = np.array([[1, 2, 3]], dtype=np.int16)
    expected_right = np.array([[10], [20]], dtype=np.int16)
    expected_mask = np.array([[True, False, True], [False, True, True]])
    expected_out = np.full((2, 3), -7, dtype=np.int32)
    actual_left = raptors.array(expected_left.tolist(), dtype=raptors.int16)
    actual_right = raptors.array(expected_right.tolist(), dtype=raptors.int16)
    actual_mask = raptors.array(expected_mask.tolist(), dtype=raptors.bool_)
    actual_out = raptors.array([[-7, -7, -7], [-7, -7, -7]], dtype=raptors.int32)

    expected_result = np.add(
        expected_left,
        expected_right,
        out=expected_out,
        where=expected_mask,
        dtype=np.int32,
    )
    actual_result = raptors.add(
        actual_left,
        actual_right,
        out=actual_out,
        where=actual_mask,
        dtype=raptors.int32,
    )
    assert actual_result is actual_out
    _assert_ufunc_result_matches(expected_result, actual_result, exact=True)


@pytest.mark.parametrize("keyword", ["signature", "sig"])
def test_signature_alias_selects_the_same_loop_as_numpy(keyword):
    expected_left = np.array([1.5, 2.5], dtype=np.float32)
    expected_right = np.array([2.0, 3.0], dtype=np.float32)
    actual_left = raptors.array([1.5, 2.5], dtype=raptors.float32)
    actual_right = raptors.array([2.0, 3.0], dtype=raptors.float32)
    expected = np.add(expected_left, expected_right, **{keyword: "dd->d"})
    actual = raptors.add(actual_left, actual_right, **{keyword: "dd->d"})
    _assert_ufunc_result_matches(expected, actual)


def test_multi_output_ufunc_out_tuple_and_positional_outputs_match_numpy():
    expected_values = np.array([-2.75, 3.5], dtype=np.float64)
    actual_values = raptors.array([-2.75, 3.5], dtype=raptors.float64)
    expected_fraction = np.empty(2, dtype=np.float64)
    expected_integral = np.empty(2, dtype=np.float64)
    actual_fraction = raptors.empty(2, dtype=raptors.float64)
    actual_integral = raptors.empty(2, dtype=raptors.float64)

    expected = np.modf(expected_values, out=(expected_fraction, expected_integral))
    actual = raptors.modf(actual_values, out=(actual_fraction, actual_integral))
    assert actual[0] is actual_fraction
    assert actual[1] is actual_integral
    _assert_ufunc_result_matches(expected, actual, exact=True)

    expected_first = np.empty(2, dtype=np.float64)
    expected_second = np.empty(2, dtype=np.int32)
    actual_first = raptors.empty(2, dtype=raptors.float64)
    actual_second = raptors.empty(2, dtype=raptors.int32)
    expected = np.frexp(expected_values, out=(expected_first, expected_second))
    actual = raptors.frexp(actual_values, out=(actual_first, actual_second))
    _assert_ufunc_result_matches(expected, actual, exact=True)


def test_multi_output_where_preserves_each_output_and_preflights_all_outputs():
    expected_values = np.array([-2.75, 3.5, 4.25], dtype=np.float64)
    actual_values = raptors.array([-2.75, 3.5, 4.25], dtype=raptors.float64)
    expected_mask = np.array([True, False, True])
    actual_mask = raptors.array([True, False, True], dtype=raptors.bool_)
    expected_fraction = np.full(3, -7.0, dtype=np.float64)
    expected_integral = np.full(3, -9.0, dtype=np.float64)
    actual_fraction = raptors.array([-7.0, -7.0, -7.0], dtype=raptors.float64)
    actual_integral = raptors.array([-9.0, -9.0, -9.0], dtype=raptors.float64)
    expected = np.modf(
        expected_values,
        out=(expected_fraction, expected_integral),
        where=expected_mask,
    )
    actual = raptors.modf(
        actual_values,
        out=(actual_fraction, actual_integral),
        where=actual_mask,
    )
    assert actual[0] is actual_fraction
    assert actual[1] is actual_integral
    _assert_ufunc_result_matches(expected, actual, exact=True)

    unchanged = raptors.array([31.0, 32.0, 33.0], dtype=raptors.float64)
    invalid_second = raptors.array([1, 2, 3], dtype=raptors.int8)
    with pytest.raises(TypeError):
        raptors.modf(actual_values, out=(unchanged, invalid_second))
    _assert_ufunc_result_matches(np.array([31.0, 32.0, 33.0]), unchanged, exact=True)


def test_multi_output_signature_dtype_and_casting_controls_match_numpy():
    expected_values = np.array([-2.75, 3.5], dtype=np.float32)
    actual_values = raptors.array([-2.75, 3.5], dtype=raptors.float32)
    expected_first = np.zeros(2, dtype=np.float64)
    expected_second = np.zeros(2, dtype=np.float64)
    actual_first = raptors.zeros(2, dtype=raptors.float64)
    actual_second = raptors.zeros(2, dtype=raptors.float64)
    expected = np.modf(
        expected_values,
        signature="d->dd",
        out=(expected_first, expected_second),
        casting="safe",
    )
    actual = raptors.modf(
        actual_values,
        signature="d->dd",
        out=(actual_first, actual_second),
        casting="safe",
    )
    _assert_ufunc_result_matches(expected, actual, exact=True)

    with pytest.raises(TypeError):
        np.modf(expected_values, dtype=np.float64, signature="d->dd")
    with pytest.raises(TypeError):
        raptors.modf(actual_values, dtype=np.float64, signature="d->dd")


def test_numeric_ufunc_layouts_and_broadcasting_match_numpy():
    values = np.arange(1, 13, dtype=np.float64).reshape(3, 4)
    expected_inputs = [values, values.T, values[::-1, ::2], values[:, :1]]
    actual_base = raptors.array(values.tolist(), dtype=raptors.float64)
    actual_inputs = [actual_base, actual_base.T, actual_base[::-1, ::2], actual_base[:, :1]]

    for expected_input, actual_input in zip(expected_inputs, actual_inputs):
        _assert_ufunc_result_matches(np.square(expected_input), raptors.square(actual_input))

    expected = np.add(expected_inputs[3], np.arange(4, dtype=np.float64))
    actual = raptors.add(actual_inputs[3], raptors.array([0.0, 1.0, 2.0, 3.0]))
    _assert_ufunc_result_matches(expected, actual)


def test_ufunc_methods_reduce_accumulate_reduceat_outer_and_at_match_numpy():
    values = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    expected = np.add.reduce(values, axis=1, initial=10, keepdims=True)
    actual_values = raptors.array(values.tolist(), dtype=raptors.int64)
    _assert_ufunc_result_matches(expected, raptors.add.reduce(actual_values, axis=1, initial=10, keepdims=True), exact=True)

    expected = np.multiply.accumulate(values, axis=1)
    actual = raptors.multiply.accumulate(actual_values, axis=1)
    _assert_ufunc_result_matches(expected, actual, exact=True)

    cuts = np.array([0, 2], dtype=np.int64)
    expected = np.add.reduceat(values, cuts, axis=1)
    actual = raptors.add.reduceat(actual_values, raptors.array([0, 2], dtype=raptors.int64), axis=1)
    _assert_ufunc_result_matches(expected, actual, exact=True)

    left = np.array([1, 2], dtype=np.int64)
    right = np.array([[10, 20], [30, 40]], dtype=np.int64)
    expected = np.add.outer(left, right)
    actual = raptors.add.outer(
        raptors.array([1, 2], dtype=raptors.int64),
        raptors.array(right.tolist(), dtype=raptors.int64),
    )
    _assert_ufunc_result_matches(expected, actual, exact=True)

    expected_target = np.zeros(4, dtype=np.int64)
    actual_target = raptors.zeros(4, dtype=raptors.int64)
    indices = np.array([1, 1, 3], dtype=np.int64)
    updates = np.array([2, 5, 7], dtype=np.int64)
    np.add.at(expected_target, indices, updates)
    raptors.add.at(
        actual_target,
        raptors.array([1, 1, 3], dtype=raptors.int64),
        raptors.array([2, 5, 7], dtype=raptors.int64),
    )
    _assert_ufunc_result_matches(expected_target, actual_target, exact=True)


def test_ufunc_methods_support_scalar_and_broadcast_advanced_at_indices():
    expected = np.zeros((2, 3), dtype=np.int64)
    actual = raptors.zeros((2, 3), dtype=raptors.int64)
    rows = np.array([[0], [1]], dtype=np.int64)
    columns = np.array([[1, 1, 2]], dtype=np.int64)
    np.add.at(expected, (rows, columns), 1)
    raptors.add.at(
        actual,
        (
            raptors.array([[0], [1]], dtype=raptors.int64),
            raptors.array([[1, 1, 2]], dtype=raptors.int64),
        ),
        1,
    )
    _assert_ufunc_result_matches(expected, actual, exact=True)


@pytest.mark.parametrize(
    "name,values",
    [
        ("fmax", [float("nan"), 2.0]),
        ("fmin", [float("nan"), 2.0]),
        ("maximum", [float("nan"), 2.0]),
        ("minimum", [float("nan"), 2.0]),
    ],
)
def test_extrema_nan_and_signed_zero_semantics_match_numpy(name, values):
    expected_left = np.array(values, dtype=np.float64)
    expected_right = np.array([1.0, -0.0], dtype=np.float64)
    actual_left = raptors.array(values, dtype=raptors.float64)
    actual_right = raptors.array([1.0, -0.0], dtype=raptors.float64)
    with np.errstate(all="ignore"), raptors.errstate(all="ignore"):
        expected = getattr(np, name)(expected_left, expected_right)
        actual = getattr(raptors, name)(actual_left, actual_right)
    _assert_ufunc_result_matches(expected, actual, exact=True)


@pytest.mark.parametrize("name", ["remainder", "fmod", "floor_divide"])
def test_division_family_negative_and_zero_edges_match_numpy(name):
    if name == "floor_divide":
        left = np.array([-7, 7, 0], dtype=np.int64)
        right = np.array([3, -3, 2], dtype=np.int64)
        actual_left = raptors.array([-7, 7, 0], dtype=raptors.int64)
        actual_right = raptors.array([3, -3, 2], dtype=raptors.int64)
    else:
        left = np.array([-7.5, 7.5, -0.0], dtype=np.float64)
        right = np.array([3.0, -3.0, 2.0], dtype=np.float64)
        actual_left = raptors.array([-7.5, 7.5, -0.0], dtype=raptors.float64)
        actual_right = raptors.array([3.0, -3.0, 2.0], dtype=raptors.float64)
    with np.errstate(all="ignore"), raptors.errstate(all="ignore"):
        expected = getattr(np, name)(left, right)
        actual = getattr(raptors, name)(actual_left, actual_right)
    _assert_ufunc_result_matches(expected, actual, exact=True)


@pytest.mark.parametrize("name", ["logaddexp", "logaddexp2"])
def test_logaddexp_large_values_are_stable_and_match_numpy(name):
    left = np.array([1000.0, -1000.0, -np.inf], dtype=np.float64)
    right = np.array([999.0, -1001.0, -np.inf], dtype=np.float64)
    actual_left = raptors.array(left.tolist(), dtype=raptors.float64)
    actual_right = raptors.array(right.tolist(), dtype=raptors.float64)
    with np.errstate(all="ignore"), raptors.errstate(all="ignore"):
        expected = getattr(np, name)(left, right)
        actual = getattr(raptors, name)(actual_left, actual_right)
    _assert_ufunc_result_matches(expected, actual)


@pytest.mark.parametrize("name", ["log1p", "expm1"])
def test_complex_log1p_and_expm1_near_zero_match_numpy(name):
    values = np.array([1e-20 - 1e-20j, -0.25 + 0.5j], dtype=np.complex128)
    actual_values = raptors.array(values.tolist(), dtype=raptors.complex128)
    with np.errstate(all="ignore"), raptors.errstate(all="ignore"):
        expected = getattr(np, name)(values)
        actual = getattr(raptors, name)(actual_values)
    _assert_ufunc_result_matches(expected, actual)


@pytest.mark.parametrize(
    "dtype,raptors_dtype,large",
    [
        (np.complex64, raptors.complex64, 1e20),
        (np.complex128, raptors.complex128, 1e200),
    ],
)
def test_complex_inverse_trig_preserves_branch_cuts_and_large_finite_values(
    dtype, raptors_dtype, large
):
    values = np.array(
        [
            complex(-4.0, 0.0),
            complex(-4.0, -0.0),
            complex(-1.0, 0.0),
            complex(-1.0, -0.0),
            complex(-0.0, 0.0),
            complex(-0.0, -0.0),
            complex(0.25, 1e-20),
            complex(0.25, -1e-20),
            complex(large, large),
        ],
        dtype=dtype,
    )
    actual_values = raptors.array(values.tolist(), dtype=raptors_dtype)

    with np.errstate(all="ignore"), raptors.errstate(all="ignore"):
        for name in ("arcsin", "arccos"):
            expected = getattr(np, name)(values)
            actual = getattr(raptors, name)(actual_values)
            _assert_ufunc_result_matches(expected, actual)


@pytest.mark.parametrize(
    "loop_keyword,loop_value,actual_value",
    [
        ("dtype", np.float64, raptors.float64),
        ("signature", "dd->d", "dd->d"),
        ("signature", (None, None, np.float64), (None, None, raptors.float64)),
    ],
    ids=("dtype", "signature-string", "signature-tuple"),
)
def test_broadcast_where_out_combined_with_loop_casting_and_order(
    loop_keyword, loop_value, actual_value
):
    expected_left = np.array([[1.5], [-2.25]], dtype=np.float32)
    expected_right = np.array([[2, -1, 4]], dtype=np.int16)
    expected_mask = np.array([[True, False, True]])
    actual_left = raptors.array(expected_left.tolist(), dtype=raptors.float32)
    actual_right = raptors.array(expected_right.tolist(), dtype=raptors.int16)
    actual_mask = raptors.array(expected_mask.tolist(), dtype=raptors.bool_)

    expected_out = np.full((2, 3), -9.0, dtype=np.float64, order="F")
    actual_out = raptors.array([[-9.0, -9.0]] * 3, dtype=raptors.float64).T
    expected = np.add(
        expected_left,
        expected_right,
        out=expected_out,
        where=expected_mask,
        casting="safe",
        order="F",
        **{loop_keyword: loop_value},
    )
    actual = raptors.add(
        actual_left,
        actual_right,
        out=actual_out,
        where=actual_mask,
        casting="safe",
        order="F",
        **{loop_keyword: actual_value},
    )

    assert actual is actual_out
    assert actual.strides == expected.strides
    _assert_ufunc_result_matches(expected, actual, exact=True)


def test_logical_ufuncs_keep_numpy_truthiness_and_input_casting_rules():
    expected_left = np.array([0, -2, 5], dtype=np.int16)
    expected_right = np.array([0, 1, 3], dtype=np.int64)
    actual_left = raptors.array([0, -2, 5], dtype=raptors.int16)
    actual_right = raptors.array([0, 1, 3], dtype=raptors.int64)

    for keyword in (
        {"dtype": np.bool_, "casting": "same_kind"},
        {"dtype": np.bool_, "casting": "safe"},
        {"signature": "bb->?", "casting": "same_kind"},
    ):
        actual_keyword = dict(keyword)
        if "dtype" in actual_keyword:
            actual_keyword["dtype"] = raptors.bool_
        expected_result = np.logical_and(expected_left, expected_right, **keyword)
        actual_result = raptors.logical_and(actual_left, actual_right, **actual_keyword)
        _assert_ufunc_result_matches(expected_result, actual_result, exact=True)

    with pytest.raises(TypeError):
        np.logical_and(
            expected_left,
            expected_right,
            signature="bb->?",
            casting="safe",
        )
    with pytest.raises(TypeError):
        raptors.logical_and(
            actual_left,
            actual_right,
            signature="bb->?",
            casting="safe",
        )
    with pytest.raises(TypeError):
        np.logical_and(expected_left, expected_right, dtype=np.int8)
    with pytest.raises(TypeError):
        raptors.logical_and(actual_left, actual_right, dtype=raptors.int8)


def test_multi_output_combines_signature_casting_where_and_fortran_outputs():
    expected_values = np.array([[-2.75, 3.5, 4.25], [5.5, -6.25, 7.0]], dtype=np.float32)
    expected_mask = np.array([[True, False, True]])
    actual_values = raptors.array(expected_values.tolist(), dtype=raptors.float32)
    actual_mask = raptors.array(expected_mask.tolist(), dtype=raptors.bool_)

    expected_first = np.full((2, 3), -7.0, dtype=np.float64, order="F")
    expected_second = np.full((2, 3), -9.0, dtype=np.float64, order="F")
    actual_first = raptors.array([[-7.0, -7.0]] * 3, dtype=raptors.float64).T
    actual_second = raptors.array([[-9.0, -9.0]] * 3, dtype=raptors.float64).T
    expected = np.modf(
        expected_values,
        signature="d->dd",
        out=(expected_first, expected_second),
        where=expected_mask,
        casting="safe",
        order="F",
    )
    actual = raptors.modf(
        actual_values,
        signature="d->dd",
        out=(actual_first, actual_second),
        where=actual_mask,
        casting="safe",
        order="F",
    )

    assert actual[0] is actual_first
    assert actual[1] is actual_second
    for expected_output, actual_output in zip(expected, actual):
        assert expected_output.strides == actual_output.strides
    _assert_ufunc_result_matches(expected, actual, exact=True)


@pytest.mark.parametrize(
    "dtype,minimum,expected_count",
    [
        (np.int8, -(2**7), 1),
        (np.int16, -(2**15), 1),
        (np.int32, -(2**31), 1),
        (np.int64, -(2**63), 1),
    ],
)
def test_bitwise_count_handles_signed_minimum_values(dtype, minimum, expected_count):
    expected = np.bitwise_count(np.array([minimum, -1, 0, 7], dtype=dtype))
    actual = raptors.bitwise_count(
        raptors.array([minimum, -1, 0, 7], dtype=np.dtype(dtype).name)
    )
    _assert_ufunc_result_matches(expected, actual, exact=True)
    assert actual[0] == expected_count


def test_numeric_operators_reflected_comparison_divmod_and_inplace_match_numpy():
    expected = np.array([2, 5, 8], dtype=np.int64)
    actual = raptors.array([2, 5, 8], dtype=raptors.int64)
    _assert_ufunc_result_matches(20 - expected, 20 - actual, exact=True)
    _assert_ufunc_result_matches(20 / expected, 20 / actual)
    _assert_ufunc_result_matches(expected < 6, actual < 6, exact=True)
    expected_divmod = divmod(20, expected)
    actual_divmod = divmod(20, actual)
    _assert_ufunc_result_matches(expected_divmod, actual_divmod, exact=True)

    expected += 3
    actual += 3
    _assert_ufunc_result_matches(expected, actual, exact=True)


def test_floating_error_state_restores_and_raise_mode_matches_numpy():
    assert raptors.geterr() == np.geterr()
    old = raptors.seterr(divide="raise", invalid="ignore")
    try:
        assert old == np.geterr()
        assert raptors.geterr()["divide"] == "raise"
        with pytest.raises(FloatingPointError):
            raptors.divide(
                raptors.array([1.0], dtype=raptors.float64),
                raptors.array([0.0], dtype=raptors.float64),
            )
        with raptors.errstate(divide="ignore"):
            assert raptors.geterr()["divide"] == "ignore"
        assert raptors.geterr()["divide"] == "raise"
    finally:
        raptors.seterr(**old)


def test_floating_error_warning_and_callback_modes_match_numpy_contract():
    messages = []
    previous_callback = raptors.seterrcall(lambda name, flag: messages.append((name, flag)))
    previous = raptors.seterr(divide="call", invalid="ignore", over="ignore", under="ignore")
    try:
        result = raptors.divide(
            raptors.array([1.0], dtype=raptors.float64),
            raptors.array([0.0], dtype=raptors.float64),
        )
        assert np.isinf(float(result[0]))
        assert len(messages) == 1
        assert messages[0][0] == "divide by zero"
    finally:
        raptors.seterr(**previous)
        raptors.seterrcall(previous_callback)

    with pytest.warns(RuntimeWarning, match="divide by zero encountered in divide"):
        raptors.divide(
            raptors.array([1.0], dtype=raptors.float64),
            raptors.array([0.0], dtype=raptors.float64),
        )


@pytest.mark.parametrize("value", ["not-a-mode", "warns"])
def test_floating_error_api_rejects_unknown_modes(value):
    with pytest.raises(ValueError):
        raptors.seterr(divide=value)


@pytest.mark.parametrize(
    "casting,source,target,should_succeed",
    [
        ("no", np.int16, np.int16, True),
        ("equiv", np.int16, np.int16, True),
        ("safe", np.int8, np.int16, True),
        ("safe", np.int16, np.int8, False),
        ("same_kind", np.int16, np.float32, True),
        ("same_kind", np.float32, np.int16, False),
        ("unsafe", np.float32, np.int16, True),
    ],
)
def test_dtype_and_casting_controls_match_numpy(casting, source, target, should_succeed):
    expected_source = np.array([1, 2], dtype=source)
    actual_source = raptors.array([1, 2], dtype=raptors.DType(source.__name__))
    dtype = np.dtype(target)
    expected_call = lambda: np.add(expected_source, expected_source, dtype=dtype, casting=casting)
    actual_call = lambda: raptors.add(
        actual_source,
        actual_source,
        dtype=raptors.DType(dtype.name),
        casting=casting,
    )
    if should_succeed:
        expected = expected_call()
        actual = actual_call()
        _assert_ufunc_result_matches(expected, actual, exact=True)
    else:
        with pytest.raises(TypeError):
            expected_call()
        with pytest.raises(TypeError):
            actual_call()


def test_signature_dtype_constraints_and_output_casting_match_numpy():
    expected_left = np.array([1.25, 2.5], dtype=np.float32)
    expected_right = np.array([2.0, 3.0], dtype=np.float32)
    actual_left = raptors.array([1.25, 2.5], dtype=raptors.float32)
    actual_right = raptors.array([2.0, 3.0], dtype=raptors.float32)

    for signature in (
        (np.float64, np.float64, np.float64),
        (None, None, np.float64),
    ):
        expected = np.add(expected_left, expected_right, signature=signature)
        actual = raptors.add(actual_left, actual_right, signature=signature)
        _assert_ufunc_result_matches(expected, actual)

    expected_out = np.zeros(2, dtype=np.float64)
    actual_out = raptors.zeros(2, dtype=raptors.float64)
    expected = np.add(
        expected_left,
        expected_right,
        signature="dd->d",
        out=expected_out,
        casting="safe",
    )
    actual = raptors.add(
        actual_left,
        actual_right,
        signature="dd->d",
        out=actual_out,
        casting="safe",
    )
    assert actual is actual_out
    _assert_ufunc_result_matches(expected, actual, exact=True)


def test_where_false_preserves_outputs_and_rejects_invalid_masks_and_outputs():
    expected_out = np.array([17, 19, 23], dtype=np.int16)
    actual_out = raptors.array([17, 19, 23], dtype=raptors.int16)
    expected_result = np.add(
        np.array([1, 2, 3], dtype=np.int16),
        10,
        out=expected_out,
        where=False,
    )
    actual_result = raptors.add(
        raptors.array([1, 2, 3], dtype=raptors.int16),
        10,
        out=actual_out,
        where=False,
    )
    assert actual_result is actual_out
    _assert_ufunc_result_matches(expected_result, actual_result, exact=True)

    with pytest.warns(UserWarning, match="'where' used without 'out'"):
        with pytest.raises(TypeError):
            np.add(np.array([1]), 1, where=np.array([1]))
    with pytest.warns(UserWarning, match="'where' used without 'out'"):
        with pytest.raises(TypeError):
            raptors.add(raptors.array([1]), 1, where=raptors.array([1]))
    with pytest.raises(ValueError):
        np.add(np.arange(3), 1, out=np.empty(2, dtype=np.int64))
    with pytest.raises(ValueError):
        raptors.add(raptors.array([0, 1, 2]), 1, out=raptors.empty(2, dtype=raptors.int64))


def test_overlapping_ufunc_inputs_and_outputs_use_input_snapshots():
    expected_base = np.arange(1, 7, dtype=np.int64)
    actual_base = raptors.array([1, 2, 3, 4, 5, 6], dtype=raptors.int64)
    np.add(expected_base[:-1], expected_base[1:], out=expected_base[1:])
    raptors.add(actual_base[:-1], actual_base[1:], out=actual_base[1:])
    _assert_ufunc_result_matches(expected_base, actual_base, exact=True)


@pytest.mark.parametrize("order", ["C", "F", "A", "K"])
def test_output_order_controls_match_numpy(order):
    expected_left = np.arange(1, 7, dtype=np.float64).reshape(2, 3).T
    expected_right = np.full((2, 3), 2.0).T
    actual_left = raptors.array(expected_left.T.tolist(), dtype=raptors.float64).T
    actual_right = raptors.array(expected_right.T.tolist(), dtype=raptors.float64).T
    expected = np.add(expected_left, expected_right, order=order)
    actual = raptors.add(actual_left, actual_right, order=order)
    _assert_ufunc_result_matches(expected, actual)
    assert tuple(actual.strides) == expected.strides


def test_non_native_byteorder_inputs_and_outputs_follow_casting_rules():
    native_dtype = np.dtype("int32")
    swapped_dtype = native_dtype.newbyteorder("S")
    expected_values = np.array([1, 20, -300], dtype=swapped_dtype)
    actual_values = raptors.array(
        [1, 20, -300],
        dtype=raptors.DType(swapped_dtype.str),
    )
    with pytest.raises(TypeError):
        np.add(expected_values, expected_values, casting="no")
    with pytest.raises(TypeError):
        raptors.add(actual_values, actual_values, casting="no")

    for casting in ("equiv", "safe", "same_kind"):
        expected = np.add(expected_values, expected_values, casting=casting)
        actual = raptors.add(actual_values, actual_values, casting=casting)
        _assert_ufunc_result_matches(expected, actual, exact=True)

    expected_out = np.full(3, -1, dtype=swapped_dtype)
    actual_out = raptors.array([-1, -1, -1], dtype=raptors.DType(swapped_dtype.str))
    expected = np.add(expected_values, 1, out=expected_out, casting="same_kind")
    actual = raptors.add(actual_values, 1, out=actual_out, casting="same_kind")
    assert actual is actual_out
    _assert_ufunc_result_matches(expected, actual, exact=True)

    expected_native_out = np.full(3, -1, dtype=native_dtype)
    actual_native_out = raptors.array([-1, -1, -1], dtype=raptors.int32)
    with pytest.raises(TypeError):
        np.add(expected_values, 1, out=expected_native_out, casting="no")
    with pytest.raises(TypeError):
        raptors.add(actual_values, 1, out=actual_native_out, casting="no")

    expected_native = np.array([1, 20, -300], dtype=native_dtype)
    actual_native = raptors.array([1, 20, -300], dtype=raptors.int32)
    expected = np.add(expected_native, 1, out=expected_native_out, casting="no")
    actual = raptors.add(actual_native, 1, out=actual_native_out, casting="no")
    _assert_ufunc_result_matches(expected, actual, exact=True)


@pytest.mark.parametrize(
    "values,axis,kwargs",
    [
        ([[1, 2, 3], [4, 5, 6]], (0, 1), {}),
        ([[1, 2, 3], [4, 5, 6]], (-1,), {"keepdims": True}),
        ([[1, 2, 3], [4, 5, 6]], 1, {"where": [[True, False, True]]}),
        ([[1, 2, 3], [4, 5, 6]], 1, {"where": False, "initial": 10}),
    ],
)
def test_reduce_axis_tuple_keepdims_where_and_initial_match_numpy(values, axis, kwargs):
    expected_values = np.array(values, dtype=np.int64)
    actual_values = raptors.array(values, dtype=raptors.int64)
    expected_kwargs = dict(kwargs)
    actual_kwargs = dict(kwargs)
    if "where" in expected_kwargs and isinstance(expected_kwargs["where"], list):
        actual_kwargs["where"] = raptors.array(expected_kwargs["where"], dtype=raptors.bool_)
    expected = np.add.reduce(expected_values, axis=axis, **expected_kwargs)
    actual = raptors.add.reduce(actual_values, axis=axis, **actual_kwargs)
    _assert_ufunc_result_matches(expected, actual, exact=True)


def test_reduce_empty_identity_initial_and_output_match_numpy():
    expected_empty = np.empty((2, 0, 3), dtype=np.int64)
    actual_empty = raptors.empty((2, 0, 3), dtype=raptors.int64)
    expected = np.add.reduce(expected_empty, axis=1)
    actual = raptors.add.reduce(actual_empty, axis=1)
    _assert_ufunc_result_matches(expected, actual, exact=True)

    expected = np.multiply.reduce(expected_empty, axis=1, initial=3)
    actual = raptors.multiply.reduce(actual_empty, axis=1, initial=3)
    _assert_ufunc_result_matches(expected, actual, exact=True)

    expected_output = np.full(2, -1, dtype=np.int64)
    actual_output = raptors.array([-1, -1], dtype=raptors.int64)
    expected = np.add.reduce(
        np.arange(6, dtype=np.int64).reshape(2, 3),
        axis=1,
        out=expected_output,
    )
    actual = raptors.add.reduce(
        raptors.array([[0, 1, 2], [3, 4, 5]], dtype=raptors.int64),
        axis=1,
        out=actual_output,
    )
    assert actual is actual_output
    _assert_ufunc_result_matches(expected, actual, exact=True)


def test_accumulate_reduceat_negative_axis_boundaries_and_empty_indices_match_numpy():
    expected_values = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    actual_values = raptors.array([[1, 2, 3], [4, 5, 6]], dtype=raptors.int64)
    expected = np.multiply.accumulate(expected_values, axis=-1, dtype=np.int64)
    actual = raptors.multiply.accumulate(actual_values, axis=-1, dtype=raptors.int64)
    _assert_ufunc_result_matches(expected, actual, exact=True)

    for indices in ([0, 0, 2], [2, 1, 1], []):
        expected = np.add.reduceat(expected_values, indices, axis=-1)
        actual = raptors.add.reduceat(
            actual_values,
            raptors.array(indices, dtype=raptors.int64),
            axis=-1,
        )
        _assert_ufunc_result_matches(expected, actual, exact=True)


@pytest.mark.parametrize(
    "operation,left,right,mode,message",
    [
        ("divide", [1.0], [0.0], "divide", "divide by zero"),
        ("multiply", [1e308], [1e308], "over", "overflow"),
        ("multiply", [1e-300], [1e-300], "under", "underflow"),
        ("sqrt", [-1.0], None, "invalid", "invalid value"),
    ],
)
def test_floating_error_modes_cover_divide_overflow_underflow_and_invalid(
    operation, left, right, mode, message
):
    expected_state = np.seterr(all="ignore", **{mode: "raise"})
    actual_state = raptors.seterr(all="ignore", **{mode: "raise"})
    try:
        with pytest.raises(FloatingPointError, match=message):
            if right is None:
                getattr(np, operation)(np.array(left, dtype=np.float64))
            else:
                getattr(np, operation)(np.array(left, dtype=np.float64), np.array(right, dtype=np.float64))
        with pytest.raises(FloatingPointError, match=message):
            if right is None:
                getattr(raptors, operation)(raptors.array(left, dtype=raptors.float64))
            else:
                getattr(raptors, operation)(
                    raptors.array(left, dtype=raptors.float64),
                    raptors.array(right, dtype=raptors.float64),
                )
    finally:
        np.seterr(**expected_state)
        raptors.seterr(**actual_state)


def test_exact_subnormal_flags_match_numpy():
    tiny = np.nextafter(np.float64(0), np.float64(1))
    cases = [
        ("add", [tiny], [0.0]),
        ("multiply", [tiny], [1.0]),
        ("multiply", [tiny], [2.0]),
        ("divide", [tiny], [1.0]),
        ("ldexp", [1.0], [-1074]),
    ]

    for name, left, right in cases:
        expected_left = np.array(left, dtype=np.float64)
        actual_left = raptors.array(left, dtype=raptors.float64)
        if right is not None:
            expected_right = np.array(right, dtype=np.int64 if name == "ldexp" else np.float64)
            actual_right = raptors.array(
                right,
                dtype=raptors.int64 if name == "ldexp" else raptors.float64,
            )

        try:
            with np.errstate(under="raise"):
                expected = (
                    getattr(np, name)(expected_left)
                    if right is None
                    else getattr(np, name)(expected_left, expected_right)
                )
            expected_underflow = False
        except FloatingPointError:
            expected_underflow = True

        try:
            with raptors.errstate(under="raise"):
                actual = (
                    getattr(raptors, name)(actual_left)
                    if right is None
                    else getattr(raptors, name)(actual_left, actual_right)
                )
            actual_underflow = False
        except FloatingPointError:
            actual_underflow = True

        assert actual_underflow == expected_underflow, name
        if expected_underflow:
            continue
        _assert_ufunc_result_matches(expected, actual)

    # NumPy's exp2 exception flag for exact subnormal powers varies across
    # platform wheels. Compare the numerical result here and keep the flag
    # variation documented until it is stable across the supported targets.
    with np.errstate(under="ignore"):
        expected = np.exp2(np.array([-1074.0], dtype=np.float64))
    with raptors.errstate(under="ignore"):
        actual = raptors.exp2(raptors.array([-1074.0], dtype=raptors.float64))
    _assert_ufunc_result_matches(expected, actual)


@pytest.mark.parametrize("dtype", [np.dtype("longdouble"), np.dtype("clongdouble")])
def test_wide_longdouble_addition_preserves_precision(dtype):
    candidate_dtype = _raptors_dtype_for_numpy(dtype)
    left_real = np.array([2**53 + 1], dtype=np.longdouble)
    right_real = np.array([1], dtype=np.longdouble)
    left = left_real if dtype.kind == "f" else left_real.astype(dtype)
    right = right_real if dtype.kind == "f" else right_real.astype(dtype)
    expected = np.add(left, right)

    actual = raptors.add(
        raptors.array([2**53 + 1], dtype=candidate_dtype),
        raptors.array([1], dtype=candidate_dtype),
    )
    expected_candidate = raptors.array([int(expected[0].real)], dtype=candidate_dtype)

    assert actual.dtype == candidate_dtype
    assert bool(raptors.equal(actual, expected_candidate)[0]), (
        actual[0],
        expected_candidate[0],
    )


def test_wide_longdouble_exp_uses_extended_exponent_range():
    dtype = np.dtype("longdouble")
    candidate_dtype = _raptors_dtype_for_numpy(dtype)
    expected_input = np.array([1000.0], dtype=dtype)
    actual_input = raptors.array([1000.0], dtype=candidate_dtype)

    try:
        with np.errstate(over="raise"):
            expected = np.exp(expected_input)
        expected_overflow = False
    except FloatingPointError:
        expected_overflow = True

    try:
        with raptors.errstate(over="raise"):
            actual = raptors.exp(actual_input)
        actual_overflow = False
    except FloatingPointError:
        actual_overflow = True

    assert actual_overflow == expected_overflow
    if not expected_overflow:
        assert bool(raptors.isfinite(actual)[0]) == bool(np.isfinite(expected[0]))


@pytest.mark.parametrize(
    "name,left,right",
    [
        ("multiply", [np.nextafter(0.0, 1.0)], [1.5]),
        ("divide", [np.nextafter(0.0, 1.0)], [2.0]),
    ],
)
def test_inexact_subnormal_results_report_underflow(name, left, right):
    expected_left = np.array(left, dtype=np.float64)
    expected_right = np.array(right, dtype=np.float64)
    actual_left = raptors.array(left, dtype=raptors.float64)
    actual_right = raptors.array(right, dtype=raptors.float64)

    with np.errstate(under="raise"), pytest.raises(FloatingPointError, match="underflow"):
        getattr(np, name)(expected_left, expected_right)
    with raptors.errstate(under="raise"), pytest.raises(
        FloatingPointError, match="underflow"
    ):
        getattr(raptors, name)(actual_left, actual_right)


def test_floating_error_print_log_and_errstate_decorator_match_numpy(capfd):
    import io

    expected_state = np.seterr(divide="print")
    actual_state = raptors.seterr(divide="print")
    try:
        np.divide(np.array([1.0]), np.array([0.0]))
        expected_output = capfd.readouterr()
        raptors.divide(raptors.array([1.0]), raptors.array([0.0]))
        actual_output = capfd.readouterr()
        assert actual_output == expected_output
    finally:
        np.seterr(**expected_state)
        raptors.seterr(**actual_state)

    expected_log = io.StringIO()
    actual_log = io.StringIO()
    expected_call = np.seterrcall(expected_log)
    actual_call = raptors.seterrcall(actual_log)
    expected_state = np.seterr(divide="log")
    actual_state = raptors.seterr(divide="log")
    try:
        np.divide(np.array([1.0]), np.array([0.0]))
        raptors.divide(raptors.array([1.0]), raptors.array([0.0]))
        assert actual_log.getvalue() == expected_log.getvalue()
    finally:
        np.seterr(**expected_state)
        raptors.seterr(**actual_state)
        np.seterrcall(expected_call)
        raptors.seterrcall(actual_call)

    @raptors.errstate(divide="ignore")
    def decorated():
        assert raptors.geterr()["divide"] == "ignore"
        return raptors.divide(raptors.array([1.0]), raptors.array([0.0]))

    decorated()
    assert raptors.geterr()["divide"] == np.geterr()["divide"]
