import numpy as np
import scipy.sparse as sp


def hamming_code(rank: int) -> sp.csr_matrix:
    """
    Outputs a Hamming code parity check matrix given its rank.

    The Hamming code parity check matrix is a binary matrix whose columns are all nonzero binary permutations of
    a fixed set of rank binary vectors. The Hamming code is defined over the binary field GF(2), and the rank
    specifies the length of the binary vectors. Each column of the parity check matrix corresponds to a binary
    codeword of the Hamming code.

    Parameters
    ----------
    rank: int
        The rank of the Hamming code parity check matrix.

    Returns
    -------
    sp.csr_matrix
        The Hamming code parity check matrix in sparse CSR format.

    Raises
    ------
    TypeError
        If the input variable 'rank' is not of type 'int'.

    Example
    -------
    >>> print(hamming_code(3).toarray())
    [[0 0 0 1 1 1 1]
     [0 1 1 0 0 1 1]
     [1 0 1 0 1 0 1]]

    """
    if not isinstance(rank, int):
        raise TypeError("The input variable 'rank' must be of type 'int'.")

    # The number of columns in the parity check matrix is (2^rank) - 1
    num_cols = int((2**rank) - 1)

    # Initialize lists to store the row indices, column indices, and data values of the nonzero elements in the matrix
    row_indices = []
    col_indices = []
    data = []

    # Generate all possible binary vectors of length 'rank' and use them as columns in the parity check matrix
    for i in range(num_cols):
        # Convert integer i to binary representation
        mod2_col = np.array(list(np.binary_repr(i + 1, width=rank)), dtype=int)
        # Loop over the elements in the binary vector
        for j, value in enumerate(mod2_col):
            if value == 1:
                # Store the indices and value of each non-zero element in the column
                row_indices.append(j)
                col_indices.append(i)
                data.append(np.uint8(value))

    # Create a sparse matrix in CSR format from the data, row indices, and column indices
    return sp.csr_matrix(
        (data, (row_indices, col_indices)), shape=(rank, num_cols), dtype=np.uint8
    )
