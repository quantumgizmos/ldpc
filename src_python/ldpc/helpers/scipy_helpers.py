import numpy as np
import scipy.sparse
from typing import Union


def convert_to_binary_sparse(
    matrix: Union[np.ndarray, scipy.sparse.spmatrix],
) -> scipy.sparse.spmatrix:
    """
    Convert a numpy array or a scipy sparse matrix to a binary scipy sparse matrix.

    This function checks if the input matrix is either a numpy array or a scipy sparse matrix with a suitable data type.
    It converts a numpy array to a scipy sparse matrix (CSR format), eliminates zero elements, and ensures that all
    elements are either 0 or 1. Raises appropriate errors for invalid inputs or non-binary matrices.

    Parameters
    ----------
    matrix : Union[np.ndarray, scipy.sparse.spmatrix]
        The input matrix to be converted. This should be either a numpy array or a scipy sparse matrix.
        Acceptable data types for the matrix are uint8, int8, or int.

    Returns
    -------
    scipy.sparse.spmatrix
        A binary scipy sparse matrix in CSR format. The matrix will only contain elements that are either 0 or 1.

    Raises
    ------
    TypeError
        If the input is not a numpy array or a scipy sparse matrix.
        If the data type of the matrix is not uint8, int8, or int.

    ValueError
        If the input matrix contains elements other than 0 or 1.
    """

    # Check input type
    if not isinstance(matrix, (np.ndarray, scipy.sparse.spmatrix)):
        raise TypeError(
            f"Input must be a binary numpy array or scipy sparse matrix, not {type(matrix)}"
        )

    # Check dtype
    if matrix.dtype not in [np.uint8, np.int8, int, float]:
        raise TypeError(
            f"Input matrix must have dtype uint8, int8, or int, not {matrix.dtype}"
        )
    else:
        pass

    # Convert numpy array to sparse matrix
    matrix = (
        scipy.sparse.csr_matrix(matrix, dtype=np.uint8)
        if isinstance(matrix, np.ndarray)
        else matrix
    )

    # Check if the matrix is binary
    if not np.all(np.isin(matrix.data, [1, 0, 1.0, 0.0])):
        raise ValueError("Input matrix must be a binary matrix.")

    if matrix.dtype == float:
        matrix = matrix.astype(np.uint8)

    # Eliminate any zero elements
    matrix.eliminate_zeros()

    return matrix
