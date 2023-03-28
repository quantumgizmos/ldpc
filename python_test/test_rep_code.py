import pytest
import numpy as np
import scipy.sparse as sp
from typing import List

from ldpc2.codes import rep_code, ring_code


@pytest.mark.parametrize("distance, expected_output", [
    (2, np.array([[1, 1]])),
    (3, np.array([[1, 1, 0], [0, 1, 1]])),
    (4, np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])),
    (5, np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]]))
])
def test_rep_code(distance: int, expected_output: List[List[int]]):
    parity_check_matrix = rep_code(distance).toarray()
    print(parity_check_matrix)
    np.testing.assert_array_equal(parity_check_matrix, expected_output)


@pytest.mark.parametrize("distance, expected_output", [
    (2, np.array([[1, 1], [1, 1]])),
    (3, np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])),
    (4, np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]])),
    (5, np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1], [1, 0, 0, 0, 1]]))
])
def test_ring_code(distance: int, expected_output: List[List[int]]):
    parity_check_matrix = ring_code(distance).toarray()
    np.testing.assert_array_equal(parity_check_matrix, expected_output)


def test_rep_code_distance_lt2():
    with pytest.raises(ValueError):
        rep_code(1)

def test_ring_code_distance_lt2():
    with pytest.raises(ValueError):
        ring_code(1)