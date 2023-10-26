import pytest
import numpy as np
from scipy.sparse import csr_matrix
from ldpc.gf2sparse import io_test, rank, kernel, PluDecomposition


def hamming_code(rank: int) -> csr_matrix:
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
    csr_matrix
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
    num_cols = int((2 ** rank) - 1)

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
    return csr_matrix((data, (row_indices, col_indices)), shape=(rank, num_cols))



def rep_code(distance: int) -> csr_matrix:
    """
    Outputs repetition code parity check matrix for specified distance.
    Parameters
    ----------
    distance: int
        The distance of the repetition code. Must be greater than or equal to 2.
    Returns
    -------
    csr_matrix
        The repetition code parity check matrix in sparse CSR matrix format.
    Examples
    --------
    >>> print(rep_code(5).toarray())
    [[1 1 0 0 0]
     [0 1 1 0 0]
     [0 0 1 1 0]
     [0 0 0 1 1]]
    """

    if distance < 2:
        raise ValueError("Distance should be greater than or equal to 2.")

    rows = []
    cols = []
    data = []

    for i in range(distance - 1):
        rows += [i, i]
        cols += [i, i+1]
        data += [1, 1]

    return csr_matrix((data, (rows, cols)), shape=(distance-1, distance), dtype=np.uint8)

def ring_code(distance: int) -> csr_matrix:
    """
    Outputs ring code (closed-loop repetion code) parity check matrix
    for a specified distance. 
    Parameters
    ----------
    distance: int
        The distance of the repetition code. Must be greater than or equal to 2.
    Returns
    -------
    csr_matrix
        The repetition code parity check matrix in sparse CSR matrix format.
    Examples
    --------
    >>> print(ring_code(5).toarray())
    [[1 1 0 0 0]
     [0 1 1 0 0]
     [0 0 1 1 0]
     [0 0 0 1 1]
     [1 0 0 0 1]]
    """

    if distance < 2:
        raise ValueError("Distance should be greater than or equal to 2.")

    rows = []
    cols = []
    data = []

    for i in range(distance - 1):
        rows += [i, i]
        cols += [i, i+1]
        data += [1, 1]

    # close the loop
    rows += [distance - 1, distance - 1]
    cols += [0, distance - 1]
    data += [1, 1]

    return csr_matrix((data, (rows, cols)), shape=(distance, distance), dtype=np.uint8)


def test_constructor_rep_code():

    for i in range(2,10):

        H = rep_code(i)
        out = io_test(H)
        assert np.array_equal(out.toarray(),H.toarray()) == True

    for i in range(2,10):
        H = rep_code(i).toarray()
        out = io_test(H)
        assert np.array_equal(out.toarray(),H) == True

    for i in range(2,100):

        H = ring_code(i)
        out = io_test(H)
        assert np.array_equal(out.toarray(),H.toarray()) == True

    for i in range(2,100):
        H = ring_code(i).toarray()
        out = io_test(H)
        assert np.array_equal(out.toarray(),H) == True


def test_rank_rep_code():
    
    for i in range(2,10):
        H = rep_code(i)
        assert rank(H) == i-1

    for i in range(2,10):
        H = ring_code(i)
        assert rank(H) == i-1

    for i in range(2,10):
        H = rep_code(i).T
        assert rank(H) == i-1

    assert rank(rep_code(1000).T) == 999

def test_kernel_rep_code():

    for i in range(2,10):
        H = rep_code(i)
        assert kernel(H).shape == (1, i)
        assert not ((H@kernel(H).T).data % 2).any()

    for i in range(2,10):
        H = ring_code(i)
        assert kernel(H).shape == (1, i)
        assert not ((H@kernel(H).T).data % 2).any()

    for i in range(2,10):
        H = rep_code(i).T
        assert kernel(H).nnz == 0

    for i in range(2,10):
        H = ring_code(i).T
        assert kernel(H).shape == (1, i)
        assert not ((H@kernel(H).T).data % 2).any()

    assert kernel(rep_code(1000)).shape == (1, 1000)

def test_kernel_hamming_code():
    for i in range(3,10):
        H = hamming_code(i)
        assert kernel(H).shape == (2**i - 1 - i, 2**i - 1)
        assert not ((H@kernel(H).T).data % 2).any()

def test_plu_decomposition():
    
    CODES = [rep_code,ring_code,hamming_code]
    
    for code in CODES:
        for d in range(2,10):
            H = code(d)
            plu = PluDecomposition(H)
            P = plu.P
            L = plu.L
            U = plu.U
            assert np.array_equal( (P@L@U).toarray() %2, H.toarray() )

def test_lu_solve():

    CODES = [rep_code,ring_code,hamming_code]
    
    for code in CODES:
        for d in range(2,10):
            H = code(d)
            plu = PluDecomposition(H)
            x = np.random.randint(2, size=H.shape[1])
            y = H@x % 2
            x_solution = plu.lu_solve(y)
            assert np.array_equal(H@x_solution % 2, y)


if __name__ == "__main__":
    pytest.main([__file__])