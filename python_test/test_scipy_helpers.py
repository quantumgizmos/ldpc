import numpy as np
import scipy.sparse
import pytest
from ldpc.helpers.scipy_helpers import convert_to_binary_sparse  # Replace 'your_module' with the actual module name

def test_with_valid_numpy_array():
    matrix = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    result = convert_to_binary_sparse(matrix)
    assert isinstance(result, scipy.sparse.spmatrix)
    assert np.array_equal(result.toarray(), matrix)
    assert result.nnz == 2

def test_with_valid_scipy_sparse_matrix():
    matrix = scipy.sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.int8))
    result = convert_to_binary_sparse(matrix)
    assert isinstance(result, scipy.sparse.spmatrix)
    # print(result.dtype)
    assert result.dtype == np.dtype(np.int8)
    assert np.array_equal(result.toarray(), matrix.toarray())
    assert result.nnz == 2


def test_with_invalid_data_type():
    with pytest.raises(TypeError):
        convert_to_binary_sparse("not a matrix")

def test_with_float_dtype():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])  # float type
    result = convert_to_binary_sparse(matrix)
    assert result.dtype == np.dtype(np.uint8)
    assert result.nnz == 2

def test_with_non_binary_matrix():
    matrix = np.array([[2, 0], [0, 3]], dtype=np.int8)  # Non-binary elements
    with pytest.raises(ValueError):
        convert_to_binary_sparse(matrix)

def test_non_binary():
    matrix = scipy.sparse.csr_matrix(np.array([[0, 2], [1, 0]], dtype=np.int8))
    with pytest.raises(ValueError):
        convert_to_binary_sparse(matrix)

def test_eliminate_zeros():
    matrix = scipy.sparse.csr_matrix(np.array([[1, 1], [1, 1]], dtype=np.uint8))
    matrix.data[0] = 0
    result = convert_to_binary_sparse(matrix)
    assert result.nnz == 3
    