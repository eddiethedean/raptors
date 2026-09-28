from types import SimpleNamespace

import numpy as np
import pytest
import raptors

from harness import assert_array_matches, assert_candidate_observable


class Int64Scalar(int):
    def item(self):
        return int(self)


class FaultyArray:
    def __init__(self, *, dtype="int64", shape=(2,), strides=(8,), values=(1, 2), fail=False):
        self.dtype = SimpleNamespace(name=dtype)
        self.shape = shape
        self.strides = strides
        self._values = values
        self._fail = fail

    def __getitem__(self, index):
        if self._fail:
            raise ReferenceError("owner expired")
        if isinstance(index, tuple):
            index = index[0] if index else 0
        return Int64Scalar(self._values[index])


def test_harness_detects_wrong_dtype_and_shape():
    reference = np.array([1, 2], dtype=np.int64)
    with pytest.raises(AssertionError, match="dtype mismatch"):
        assert_array_matches(reference, FaultyArray(dtype="float64"))
    with pytest.raises(AssertionError, match="shape mismatch"):
        assert_array_matches(reference, FaultyArray(shape=(1,), strides=(8,), values=(1,)))


def test_harness_detects_broadcast_shape_errors():
    reference = np.broadcast_to(np.array([1, 2, 3], dtype=np.int64), (2, 3))
    wrong_backend = FaultyArray(shape=(3,), strides=(8,), values=(1, 2, 3))
    with pytest.raises(AssertionError, match="shape mismatch"):
        assert_array_matches(reference, wrong_backend)


def test_harness_detects_alias_and_owner_lifetime_faults():
    reference = np.array([0, 0, 1, 2], dtype=np.int64)
    wrong_after_overlap = FaultyArray(shape=(4,), strides=(8,), values=(0, 1, 2, 3))
    with pytest.raises(AssertionError, match="value mismatch"):
        assert_array_matches(reference, wrong_after_overlap)
    expired_owner = FaultyArray(fail=True)
    with pytest.raises(AssertionError, match="owner expired"):
        assert_candidate_observable(expired_owner)


def test_correct_preview_passes_the_same_harness():
    reference = np.array([[1, 2], [3, 4]], dtype=np.int64)
    candidate = raptors.array([[1, 2], [3, 4]], dtype=raptors.int64)
    assert_array_matches(reference, candidate)
