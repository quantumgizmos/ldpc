from scipy import sparse
import numpy as np
from typing import Optional


def random_binary_code(
    rows: int,
    cols: int,
    row_weight: int,
    seed: Optional[int] = None,
    variance: float = 0,
) -> sparse.spmatrix:
    """
    Generate a random binary code matrix with given dimensions and row weight,
    optionally with variance on the row weight.

    Parameters
    ----------
    rows : int
        The number of rows in the matrix, representing the code words.
    cols : int
        The number of columns in the matrix, representing the code length.
    row_weight : int
        The mean number of non-zero elements in each row.
    seed : Optional[int], optional
        The seed for the random number generator (default is None, which does not seed the generator).
    variance : float, optional
        The variance in the number of non-zero elements per row (default is 0, which means no variance).

    Returns
    -------
    scipy.sparse.spmatrix
        A scipy sparse matrix in CSR format with the specified row weight and dimensions, representing a binary code.
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize lists to store the coordinates of non-zero elements.
    row_indices = []
    col_indices = []

    for row in range(rows):
        # Determine the actual row weight with the specified variance.
        actual_row_weight = max(1, int(np.random.normal(row_weight, np.sqrt(variance))))

        # Ensure the row weight does not exceed the number of columns.
        actual_row_weight = min(actual_row_weight, cols)

        # Ensure unique column indices for the current row.
        cols_selected = np.random.choice(cols, actual_row_weight, replace=False)
        row_indices.extend([row] * actual_row_weight)
        col_indices.extend(cols_selected)

    # Create a binary array with ones at the specified indices.
    data = np.ones(len(row_indices), dtype=int)

    # Create a sparse matrix in COO format and convert to CSR.
    binary_code_matrix = sparse.coo_matrix(
        (data, (row_indices, col_indices)), shape=(rows, cols), dtype=np.uint8
    ).tocsr()

    return binary_code_matrix
