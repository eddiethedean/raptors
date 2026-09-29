import gc

import numpy as np
from hypothesis import given, settings, strategies as st
import raptors

from harness import assert_array_matches

SMALL_I64 = st.integers(min_value=-(2**63), max_value=2**63 - 1)


@settings(max_examples=100, deadline=None, derandomize=True)
@given(
    values=st.lists(SMALL_I64, min_size=0, max_size=32),
    start=st.one_of(st.none(), st.integers(-40, 40)),
    stop=st.one_of(st.none(), st.integers(-40, 40)),
    step=st.integers(-8, 8).filter(lambda value: value != 0),
)
def test_generated_slice_layout_and_values(values, start, stop, step):
    expected = np.array(values, dtype=np.int64)[slice(start, stop, step)]
    actual = raptors.array(values, dtype=raptors.int64)[slice(start, stop, step)]
    assert_array_matches(expected, actual)


@settings(max_examples=100, deadline=None, derandomize=True)
@given(values=st.lists(SMALL_I64, min_size=0, max_size=32))
def test_generated_overlapping_assignment_matches_numpy(values):
    expected = np.array(values, dtype=np.int64)
    actual = raptors.array(values, dtype=raptors.int64)
    expected[1:] = expected[:-1]
    actual[1:] = actual[:-1]
    assert_array_matches(expected, actual)


@settings(max_examples=60, deadline=None, derandomize=True)
@given(rows=st.integers(1, 8), columns=st.integers(0, 8), data=st.lists(SMALL_I64, min_size=0, max_size=64))
def test_generated_rectangular_construction(rows, columns, data):
    count = rows * columns
    flat = (data + [0] * count)[:count]
    nested = [flat[row * columns:(row + 1) * columns] for row in range(rows)]
    expected = np.array(nested, dtype=np.int64)
    actual = raptors.array(nested, dtype=raptors.int64)
    assert_array_matches(expected, actual)


@settings(max_examples=60, deadline=None, derandomize=True)
@given(values=st.lists(st.integers(0, 2**64 - 1), min_size=1, max_size=20), cut=st.integers(0, 20))
def test_generated_nested_views_survive_owner_destruction(values, cut):
    expected_owner = np.array(values, dtype=np.uint64)
    actual_owner = raptors.array(values, dtype=raptors.uint64)
    cut = min(cut, len(values))
    expected_view = expected_owner[::-1][cut:]
    actual_view = actual_owner[::-1][cut:]
    del expected_owner, actual_owner
    gc.collect()
    assert_array_matches(expected_view, actual_view)


@settings(max_examples=80, deadline=None, derandomize=True)
@given(
    values=st.lists(SMALL_I64, min_size=0, max_size=32),
    step=st.integers(min_value=1, max_value=5),
    replacement=SMALL_I64,
)
def test_generated_composed_numeric_operation_sequence(values, step, replacement):
    int64_values = [((replacement + i + 2**63) % 2**64) - 2**63 for i in range(32)]
    expected_owner = np.array(values, dtype=np.int64)
    actual_owner = raptors.array(values, dtype=raptors.int64)

    expected_view = expected_owner[::-1][::step]
    actual_view = actual_owner[::-1][::step]
    expected_reshape = expected_view.reshape((1, len(expected_view)), copy=False)
    actual_reshape = actual_view.reshape((1, len(actual_view)), copy=False)
    assert_array_matches(expected_reshape, actual_reshape)

    index_values = list(range(len(expected_view) - 1, -1, -2))
    expected_index = np.array(index_values, dtype=np.int64)
    actual_index = raptors.array(index_values, dtype=raptors.int64)
    expected_selected = expected_view[expected_index]
    actual_selected = actual_view[actual_index]
    assert_array_matches(expected_selected, actual_selected)

    expected_cast = expected_selected.astype(np.float32)
    actual_cast = actual_selected.astype(raptors.float32)
    assert_array_matches(expected_cast, actual_cast)
    expected_copy = expected_selected.copy()
    actual_copy = actual_selected.copy()
    if expected_copy.size:
        expected_copy[0] = -1
        actual_copy[0] = -1
    assert_array_matches(expected_selected, actual_selected)

    mask_values = [int(expected_view[i]) % 2 == 0 for i in range(len(expected_view))]
    expected_mask = np.array(mask_values, dtype=np.bool_)
    actual_mask = raptors.array(mask_values, dtype=raptors.bool_)
    expected_view[expected_mask] = replacement
    actual_view[actual_mask] = replacement
    if index_values:
        updates = int64_values[: len(index_values)]
        expected_view[expected_index] = updates
        actual_view[actual_index] = updates
    assert_array_matches(expected_view, actual_view)

    del expected_owner, actual_owner
    gc.collect()
    if len(expected_view):
        expected_view[0] = int64_values[-1]
        actual_view[0] = int64_values[-1]
    assert_array_matches(expected_view, actual_view)
