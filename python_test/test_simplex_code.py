import pytest
import numpy as np
from ldpc.codes.hamming_code import hamming_code
from ldpc.codes.simplex_code import simplex_code

def test_simplex_code_m3():
    m = 3
    H_simplex_actual = simplex_code(m)
    expected_H_simplex_m3 = np.array([
        [1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1]
    ], dtype=np.uint8) # Use uint8 for binary matrices

    # Check if matrices are equal
    assert np.array_equal(H_simplex_actual.toarray(), expected_H_simplex_m3)
    # Check dimensions
    assert H_simplex_actual.shape == (4, 7) # Based on the example given

def test_simplex_code_duality():
    """Tests the duality property: H_hamming @ H_simplex.T = 0"""
    for m_val in [3, 4, 5]: # Test for a few small values of m
        H_hamming = hamming_code(m_val)
        H_simplex = simplex_code(m_val)

        # Perform matrix multiplication over GF(2)
        # scipy.sparse matrices support matrix multiplication
        result = (H_hamming @ H_simplex.transpose()).toarray() % 2

        # All elements should be zero
        assert np.all(result == 0), f"Duality check failed for m={m_val}"

def test_simplex_code_dimensions():
    """Tests if the output matrix has the correct dimensions."""
    for m_val in [2, 3, 4, 5]:
        H_simplex = simplex_code(m_val)
        n = 2**m_val - 1 # Length of the code
        #The dimension of the parity check matrix of the simplex code
        # should be ( (2^m - 1) - m ) x (2^m - 1).
        expected_rows = (2**m_val - 1) - m_val
        expected_cols = 2**m_val - 1
        assert H_simplex.shape == (expected_rows, expected_cols), \
            f"Dimension mismatch for m={m_val}: Expected ({expected_rows}, {expected_cols}), got {H_simplex.shape}"
