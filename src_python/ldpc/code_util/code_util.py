import numpy as np
from itertools import combinations
import ldpc.mod2
import scipy.sparse
from scipy.special import comb as nCr
from typing import Union, Tuple, List


def construct_generator_matrix(pcm: Union[np.ndarray, scipy.sparse.spmatrix]) -> scipy.sparse.spmatrix:
    '''
    Constructs a generator matrix G from a given parity check matrix H. 
    The generator matrix G is formed such that it satisfies the condition H * G.T = 0 (mod 2), 
    where G.T represents the transpose of G and the multiplication is carried out in GF(2).

    Each row of the generator matrix G is a vector in the null space of H, 
    meaning that when matrix H is multiplied by any of these vectors, 
    the result is a zero vector, satisfying the parity check condition.

    Parameters
    ----------
    pcm : Union[np.ndarray, scipy.sparse.spmatrix]
        A binary matrix representing the parity check matrix H, 
        which can be either a dense numpy array or a sparse matrix.

    Returns
    -------
    scipy.sparse.spmatrix
        The generator matrix G as a sparse matrix, which provides efficient storage 
        and performance for matrices that are large or have a high sparsity.

    Examples
    --------
    >>> H = np.array([[0, 0, 0, 1, 1, 1, 1],
                      [0, 1, 1, 0, 0, 1, 1],
                      [1, 0, 1, 0, 1, 0, 1]])
    >>> G = construct_generator_matrix(H)
    >>> assert (H @ G.T % 2).nnz == 0  # Verifying that H * G.T is a zero matrix in GF(2)
    >>> print(G.toarray())  # Convert the sparse matrix G to a dense array for printing
    [[1 1 1 0 0 0 0]
     [0 1 1 1 1 0 0]
     [0 1 0 1 0 1 0]
     [0 0 1 1 0 0 1]]

    Note
    ----
    The function assumes that the input matrix H is a valid parity check matrix 
    and does not perform any checks for this condition. It is the user's responsibility 
    to ensure that H is properly formed.

    The function uses the `mod2.nullspace` method from the `ldpc` package to find the null space.

    '''
    return ldpc.mod2.nullspace(pcm)


def systematic_form(H):
    '''
    Converts H into systematic form so that::
        
        H=[I,A]


    Parameters
    ----------
    H: numpy.ndarray
        A parity check matrix

    Returns
    -------
    numpy.ndarray
        The parity check matrix in systematic form
    '''

    if(isinstance(H, scipy.sparse.spmatrix)):
        H = H.toarray()

    return reduced_row_echelon(H)[0]


# def codewords(H):
#     '''
#     Computes all of the the codewords of the code corresponding to the parity check matrix H. The codewords are given by the span of the nullspace.

#     Parameters
#     ----------

#     H: numpy.ndarray
#         A parity check matrix.

#     Returns
#     -------
#     numpy.ndarray
#         A matrix where each row corresponds to a codeword

#     Note
#     ----
#     If you want to calculate a basis of the codewords use `ldpc.mod2.nullspace`.

#     Examples
#     --------
#     >>> H=np.array([[0, 0, 0, 1, 1, 1, 1],[0, 1, 1, 0, 0, 1, 1],[1, 0, 1, 0, 1, 0, 1]])
#     >>> print(codewords(H))
#     [[0 0 0 0 0 0 0]
#      [0 0 0 1 1 1 1]
#      [0 0 1 0 1 1 0]
#      [0 0 1 1 0 0 1]
#      [0 1 0 0 1 0 1]
#      [0 1 0 1 0 1 0]
#      [0 1 1 0 0 1 1]
#      [0 1 1 1 1 0 0]
#      [1 0 0 0 0 1 1]
#      [1 0 0 1 1 0 0]
#      [1 0 1 0 1 0 1]
#      [1 0 1 1 0 1 0]
#      [1 1 0 0 1 1 0]
#      [1 1 0 1 0 0 1]
#      [1 1 1 0 0 0 0]
#      [1 1 1 1 1 1 1]]
#     '''

#     if(isinstance(H, scipy.sparse.spmatrix)):
#         H = H.toarray()

#     _, n = H.shape
#     zero_cw = np.zeros(n).astype(int)  # zero codewords
#     cw = row_span(nullspace(H))  # nonzero codewords
#     cw = np.vstack([zero_cw, cw])

#     return cw


# def compute_code_distance(H):
#     '''
#     Computes the distance of the code given by parity check matrix H. The code distance is given by the minimum weight of a nonzero codeword.

#     Note
#     ----
#     The runtime of this function scales exponentially with the block size. In practice, computing the code distance of codes with block lengths greater than ~10 will be very slow.

#     Parameters
#     ----------
#     H: numpy.ndarray
#         The parity check matrix
    
#     Returns
#     -------
#     int
#         The code distance
#     '''

#     if(isinstance(H, scipy.sparse.spmatrix)):
#         H = H.toarray()

#     ker=nullspace(H)

#     if len(ker)==0: return np.inf #return np.inf if the kernel is empty (eg. infinite code distance)

#     cw=row_span(ker) #nonzero codewords

#     return np.min(np.sum(cw, axis=1))


# def get_code_parameters(H):
#     """
#     Returns the code parameters in [n,k,d] notation where n is the block length, k is the number of encoed bits and d is the code distance.

#     Parameters
#     ----------
#     H: numpy.ndarray
#         The parity check matrix

#     Returns
#     -------
#     n: int
#         The block length
#     k: int
#         The number of encoded bits
#     d: int
#         The code distance
#     r: int
#         The rank of the parity check matrix
#     m: int
#         The number of checks (rows in the parity check matrix)
#     """
#     if(isinstance(H, scipy.sparse.spmatrix)):
#         H = H.toarray()

#     m, n = H.shape
#     r = rank(H)
#     k = n - r
#     d = compute_code_distance(H)

#     return n, k, d, r, m



# def search_cycles(H, girth,terminate=False):

#     m, n = np.shape(H)

#     cycle_count = 0

#     for i, combination in enumerate(combinations([k for k in range(m)], girth // 2)):
#         row_sum = np.zeros(n).astype(int)
#         for j, element in enumerate(combination):
#             row_sum += H[element]

#         two_count = np.count_nonzero(row_sum == 2)
#         if two_count >= girth // 2:
#             if terminate: return True
#             cycle_count += nCr(two_count, girth // 2)

#     if terminate: return False #terminates if the code is not cycle free
#     return cycle_count


def search_cycles(H, girth,row=None,terminate=True,exclude_rows=[]):

    """
    Searches (and counts) cycles of a specified girth.

    Parameters
    ----------

    H: numpy.ndarray
        The parity check matrix
    girth: int
        The girth (length) of the code cycles to search for
    row: int, optional
        Default value is None. If a row is specified, the function returns the local girth for that row. If row=None, then the global girth is calculated.
    terminate: int, optional
        Default value is True. If set to True, the search function will terminate as soon as the first cycle of the specefied girth is found
    exclude_rows: list, optional
        The rows of the parity check to ignore. This is useful when you are filling an empty matrix.
    
    Returns
    -------
    bool, if Terminate=True
        True if a cycle of specified girth is found. False if no cycles are found.
    int, if Terminate=False
        If terminate is set to true, the function will count the number of cycles of the specified girth
    """

    if(isinstance(H, scipy.sparse.spmatrix)):
        H = H.toarray()

    m, n = np.shape(H)
    cycle_count = 0

    if row is None:
        print(girth)
        print(list(combinations([k for k in range(m)], girth // 2)))
        for i, combination in enumerate(combinations([k for k in range(m)], girth // 2)):
            row_sum = np.zeros(n).astype(int)
            for _, element in enumerate(combination):
                row_sum += H[element]
            two_count = np.count_nonzero(row_sum == 2)
            if two_count >= girth // 2:
                if terminate: return True
                cycle_count += nCr(two_count, girth // 2) 
    else:
        rows=[row]+exclude_rows
        for i, combination in enumerate(combinations([k for k in range(m) if k not in rows], (girth // 2)-1)):
            row_sum = np.zeros(n).astype(int)
            temp=(row,)+combination
            for _, element in enumerate(temp):
                row_sum += H[element]

            two_count = np.count_nonzero(row_sum == 2)
            if two_count >= girth // 2:
                if terminate: return True
                cycle_count += nCr(two_count, girth // 2)     

    if terminate: return False #terminates if the code is not cycle free
    return cycle_count



def compute_column_row_weights(H):

    """
    Returns the upper bounds on the row and column weights of parity check matrix

    Parameters
    ----------

    H: numpy.ndarray
        The parity check matrix

    Returns
    -------
    int
        The maximum column-weight
    int
        The maximum row-weight
    """

    if(isinstance(H, scipy.sparse.spmatrix)):
        H = H.toarray()

    return np.sum(H,axis=0), np.sum(H,axis=1)

def get_ldpc_params(H):

    """
    Returns the upper bounds on the row and column weights of parity check matrix

    Parameters
    ----------

    H: numpy.ndarray
        The parity check matrix

    Returns
    -------
    int
        The maximum column-weight
    int
        The maximum row-weight
    """

    if(isinstance(H, scipy.sparse.spmatrix)):
        H = H.toarray()

    cols, rows = compute_column_row_weights(H)

    return np.max(cols), np.max(rows)

