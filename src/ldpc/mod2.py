"""
Assorted functions to work with binary vectors and matrices
"""

import numpy as np
from scipy import sparse

def mod10_to_mod2(dec, length=0):
 
    """Converts a decimal number to a binary number, with optional padding
    to produce a binary vector of a given length.

    Parameters
    ----------
    dec : int
        Decimal number (base 10).
    length : int, optional
        The length of the binary string. If the specified `length' is greater
        than the bit-length of the binary number the output is left-padded
        with zeros. The default `length' is set to zero.

    Returns
    -------
    list
        A binary integer list encoding the binary represenation of the
        inputted decimal number 
    
    Examples
    --------
    >>> mod10_to_mod2(2,length=5)
    [0, 0, 0, 1, 0]

    """

    # Convert dec to a binary string, with <length> leading zeros
    bin_str = format(dec, '0{}b'.format(length))

    # Split that string into array, converting chars to ints
    return [int(b) for b in bin_str]


def mod2_to_mod10(binary_arr):
    
    """
    Converts binary number represented as a list to a decimal number.

    Parameters
    ----------
    binary_arr : list
        A binary number represented as the entries of a list
    
    Returns
    -------
    int
        The decimal representation of the inputted binary array 
    
    Examples
    --------
    >>> mod2_to_mod10([0,0,0,1,0])
    2

    """

    bases = 2 ** np.arange(len(binary_arr))[::-1]
    return binary_arr @ bases


def row_echelon(matrix, full=False):
    
    """
    Converts a binary matrix to row echelon form via Gaussian Elimination

    Parameters
    ----------
    matrix : numpy.ndarry or scipy.sparse
        A binary matrix in either numpy.ndarray format or scipy.sparse
    full: bool, optional
        If set to `True', Gaussian elimination is only performed on the rows below
        the pivot. If set to `False' Gaussian eliminatin is performed on rows above
        and below the pivot. 
    
    Returns
    -------
        row_ech_form: numpy.ndarray
            The row echelon form of input matrix
        rank: int
            The rank of the matrix
        transform_matrix: numpy.ndarray
            The transformation matrix such that (transform_matrix@matrix)=row_ech_form
        pivot_cols: list
            List of the indices of pivot num_cols found during Gaussian elimination
    """

    num_rows, num_cols = np.shape(matrix)

    # Take copy of matrix if numpy (why?) and initialise transform matrix to identity
    if isinstance(matrix, np.ndarray):
        the_matrix = np.copy(matrix)
        transform_matrix = np.identity(num_rows).astype(int)
    elif isinstance(matrix, sparse.csr.csr_matrix):
        the_matrix = matrix
        transform_matrix = sparse.eye(num_rows, dtype="int", format="csr")
    else:
        raise ValueError('Unrecognised matrix type')

    pivot_row = 0
    pivot_cols = []

    # Iterate over cols, for each col find a pivot (if it exists)
    for col in range(num_cols):

        # Select the pivot - if not in this row, swap rows to bring a 1 to this row, if possible
        if the_matrix[pivot_row, col] != 1:

            # Find a row with a 1 in this col
            swap_row_index = pivot_row + np.argmax(the_matrix[pivot_row:num_rows, col])

            # If an appropriate row is found, swap it with the pivot. Otherwise, all zeroes - will loop to next col
            if the_matrix[swap_row_index, col] == 1:

                # Swap rows
                the_matrix[[swap_row_index, pivot_row]] = the_matrix[[pivot_row, swap_row_index]]

                # Transformation matrix update to reflect this row swap
                transform_matrix[[swap_row_index, pivot_row]] = transform_matrix[[pivot_row, swap_row_index]]

        # If we have got a pivot, now let's ensure values below that pivot are zeros
        if the_matrix[pivot_row, col]:

            if not full:  
                elimination_range = [k for k in range(pivot_row + 1, num_rows)]
            else:
                elimination_range = [k for k in range(num_rows) if k != pivot_row]

            # Let's zero those values below the pivot by adding our current row to their row
            for j in elimination_range:

                if the_matrix[j, col] != 0 and pivot_row != j:    ### Do we need second condition?

                    the_matrix[j] = (the_matrix[j] + the_matrix[pivot_row]) % 2

                    # Update transformation matrix to reflect this op
                    transform_matrix[j] = (transform_matrix[j] + transform_matrix[pivot_row]) % 2

            pivot_row += 1
            pivot_cols.append(col)

        # Exit loop once there are no more rows to search
        if pivot_row >= num_rows:
            break

    # The rank is equal to the maximum pivot index
    matrix_rank = pivot_row
    row_esch_matrix = the_matrix

    return [row_esch_matrix, matrix_rank, transform_matrix, pivot_cols]


def rank(matrix):
    """
    Returns the rank of a binary matrix

    Parameters
    ----------

    matrix: numpy.ndarray
        A binary matrix in numpy.ndarray format

    Returns
    -------
    int
        The rank of the matrix
    
    """
    return row_echelon(matrix)[1]


def reduced_row_echelon(matrix):
    """
    Converts matrix to reduced row echelon form such that the output has
    the form rre=[I,A]

    Parameters
    ----------
    matrix: numpy.ndarray
        A binary matrix in numpy.ndarray format

    Returns
    -------
    reduced_row_echelon_from: numpy.ndarray
        The reduced row echelon form of the inputted matrix in the form rre=[I,A]
    matrix_rank: int
        The binary rank of the matrix
    transform_matrix_rows: numpy.ndarray
        The transformation matrix for row permutations
    transform_matrix_cols: numpy.ndarray
        The transformation matrix for the columns
    """

    num_rows, num_cols = matrix.shape

    # Row reduce matrix to calculate rank and find the pivot cols
    _, matrix_rank, _, pivot_columns = row_echelon(matrix)

    # Rearrange matrix so that the pivot columns are first
    info_set_order = pivot_columns + [j for j in range(num_cols) if j not in pivot_columns]
    transform_matrix_columns = (np.identity(num_cols)[info_set_order].astype(int)).T
    m_q = matrix @ transform_matrix_columns  # Rearranged M

    # Row reduce m_q
    reduced_row_echelon_form, _, transform_matrix_rows, _ = row_echelon(m_q, full=True)

    return [reduced_row_echelon_form, matrix_rank, transform_matrix_rows, transform_matrix_columns]

def nullspace(matrix):
    """
    Computes the nullspace of the matrix M. Also sometimes referred to as the kernel.

    All vectors x in the nullspace of M satisfy the following condition:

    Mx=0 \forall x \in nullspace(M)

    Notes
    -----
    Why does this work?

    The transformation matrix, P, transforms the matrix M into row echelon form, ReM:

    P@M=ReM=[A,0]^T,

    where the width of A is equal to the rank. This means the bottom n-k rows of P
    must produce a zero vector when applied to M. For a more formal definition see
    the Rank-Nullity theorem.

    Parameters
    ----------
    matrix: numpy.ndarray
        A binary matrix in numpy.ndarray format
    
    Returns
    -------
    numpy.ndarray
        A binary matrix where each row is a nullspace vector of the inputted binary
        matrix
    """

    transpose = matrix.T
    m, n = transpose.shape
    _, matrix_rank, transform, _ = row_echelon(transpose)
    nspace = transform[matrix_rank:m]
    return nspace

def row_span(matrix):
    """
    Outputs the span of the row space of the matrix i.e. all linear combinations of the rows


    Parameters
    ----------
    matrix: numpy.ndarray
        The input matrix

    Returns
    -------
    numpy.ndarray
        A numpy.ndarray matrix with rows reperesenting all linear combinations of the rows of
        the inputted matrix.
    """

    span = []
    for row in matrix:
        temp = [row]
        for element in span:
            temp.append((row + element) % 2)
        span = list(np.unique(temp + span, axis=0))
    if span:
        return np.vstack(span)
    else:
        return np.array([])


def inverse(matrix):
    """
    Computes the left inverse of a full-rank matrix.

    Notes
    -----

    The `left inverse' is computed when the number of rows in the matrix
    exceeds the matrix rank. The left inverse is defined as follows:

    Inverse(M.T@M)@M.T

    We can make a further simplification by noting that the row echelon form matrix
    with full column rank has the form:

    row_echelon_form=P@M=vstack[I,A]

    In this case the left inverse simplifies to:

    Inverse(M^T@P^T@P@M)@M^T@P^T@P=M^T@P^T@P=row_echelon_form.T@P

    Parameters
    ----------
    matrix: numpy.ndarray
        The binary matrix to be inverted in numpy.ndarray format. This matrix must either be
        square full-rank or rectangular with full-column rank.

    Returns
    -------
    numpy.ndarray
        The inverted binary matrix


    Examples
    --------

    >>> # full-rank square matrix
    >>> mat=np.array([[1,1,0],[0,1,0],[0,0,1]])
    >>> i_mat=inverse(mat)
    >>> print(i_mat@mat%2)
    [[1 0 0]
     [0 1 0]
     [0 0 1]]

    >>> # full-column rank matrix
    >>> mat=np.array([[1,1,0],[0,1,0],[0,0,1],[0,1,1]])
    >>> i_mat=inverse(mat)
    >>> print(i_mat@mat%2)
    [[1 0 0]
     [0 1 0]
     [0 0 1]]

    """
    m, n = matrix.shape
    row_echelon_form, matrix_rank, transform, _ = row_echelon(matrix, True)
    if m == n and matrix_rank == m:
        return transform

    # compute the left-inverse
    elif m > matrix_rank and n == matrix_rank:  # left inverse
        return row_echelon_form.T @ transform % 2

    else:
        raise ValueError("This matrix is not invertible. Please provide either a full-rank square\
        matrix or a rectangular matrix with full column rank.")

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)   

