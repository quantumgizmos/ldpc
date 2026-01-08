import pytest
import numpy as np
import scipy.sparse as sp
from typing import List

from ldpc.codes import rep_code, ring_code, hamming_code
from ldpc.codes.generate_ldpc_peg import generate_ldpc_peg


@pytest.mark.parametrize(
    "distance, expected_output",
    [
        (2, np.array([[1, 1]])),
        (3, np.array([[1, 1, 0], [0, 1, 1]])),
        (4, np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])),
        (
            5,
            np.array(
                [[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]]
            ),
        ),
    ],
)
def test_rep_code(distance: int, expected_output: List[List[int]]):
    parity_check_matrix = rep_code(distance).toarray()
    print(parity_check_matrix)
    np.testing.assert_array_equal(parity_check_matrix, expected_output)


@pytest.mark.parametrize(
    "distance, expected_output",
    [
        (2, np.array([[1, 1], [1, 1]])),
        (3, np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])),
        (4, np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]])),
        (
            5,
            np.array(
                [
                    [1, 1, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 0, 1, 1],
                    [1, 0, 0, 0, 1],
                ]
            ),
        ),
    ],
)
def test_ring_code(distance: int, expected_output: List[List[int]]):
    parity_check_matrix = ring_code(distance).toarray()
    np.testing.assert_array_equal(parity_check_matrix, expected_output)


def test_rep_code_distance_lt2():
    with pytest.raises(ValueError):
        rep_code(1)


def test_ring_code_distance_lt2():
    with pytest.raises(ValueError):
        ring_code(1)


def test_hamming_code_output_type():
    assert isinstance(hamming_code(3), sp.csr_matrix)


def test_hamming_code_output_shape():
    assert hamming_code(3).shape == (3, 7)


def test_hamming_code_output_dtype():
    assert hamming_code(3).dtype == np.uint8


def test_hamming_code_rank_type():
    with pytest.raises(TypeError):
        hamming_code(3.0)


def test_hamming_code_rank_value():
    with pytest.raises(ValueError):
        hamming_code(-1)

@pytest.mark.parametrize("m,n,dv,dc", [(10, 20, 3, 6),])
# Test that generate_ldpc_peg returns a CSR matrix of correct shape
# and type.
def test_generate_ldpc_peg_output_type_and_shape(m: int, n: int, dv: int, dc: int):
    H = generate_ldpc_peg(m, n, dv, dc)
    assert isinstance(H, sp.csr_matrix)
    assert H.shape == (m, n)

@pytest.mark.parametrize("m,n,dv,dc", [(12, 24, 3, 6),])

# Test that all entries of the generated matrix are binary (0 or 1).
def test_generate_ldpc_peg_binary_entries(m: int, n: int, dv: int, dc: int):
    H_arr = generate_ldpc_peg(m, n, dv, dc).toarray()
    assert set(np.unique(H_arr)).issubset({0, 1})

@pytest.mark.parametrize("m,n,dv,dc", [(15, 30, 2, 4),])

# Test that each variable node has exactly dv edges and each check node
# has degree at most dc.
def test_generate_ldpc_peg_degrees(m: int, n: int, dv: int, dc: int):
    H = generate_ldpc_peg(m, n, dv, dc)
    col_sums = np.array(H.sum(axis=0)).flatten()
    np.testing.assert_array_equal(col_sums, np.full(n, dv))
    row_sums = np.array(H.sum(axis=1)).flatten()
    assert np.all(row_sums <= dc)

@pytest.mark.parametrize("m,n,dv,dc", [(20, 40, 2, 6),])

# Test that the generated matrix contains no 4-cycles, i.e.
# any two variable nodes share at most one common check neighbor.
def test_generate_ldpc_peg_no_four_cycles(m: int, n: int, dv: int, dc: int):
    H = generate_ldpc_peg(m, n, dv, dc)
    overlap = (H.T @ H).toarray()
    diag = np.diag(overlap)
    np.testing.assert_array_equal(diag, np.full(n, dv))
    off_diag = overlap - np.diag(diag)
    assert np.all(off_diag <= 1)


if __name__ == "__main__":
    pytest.main([__file__])
