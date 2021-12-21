import numpy as np
from itertools import combinations
from ldpc.mod2 import reduced_row_echelon, nullspace, row_span, rank
from scipy.special import comb as nCr


def construct_generator_matrix(H):
    '''
    Constructs a generator matrix from a parity check H. The generator matrix G satisfies the condition::
                    
        H@G.T = 0.
    
    Each of the columns of the generator matrix is a nullspace vector of
    the matrix H.
    
    
    Parameters
    ----------
    H: numpy.ndarray
        A binary matrix in numpy.ndarray format.
    
    Returns
    -------
    numpy.ndarray
        The generator matrix in numpy.ndarray format
    
    Examples
    --------
    >>> H=np.array([[0, 0, 0, 1, 1, 1, 1],[0, 1, 1, 0, 0, 1, 1],[1, 0, 1, 0, 1, 0, 1]])
    >>> G=construct_generator_matrix(H)
    >>> assert (H@G.T%2).any()==False
    >>> print(G)
    [[1 1 1 0 0 0 0]
     [0 1 1 1 1 0 0]
     [0 1 0 1 0 1 0]
     [0 0 1 1 0 0 1]]

    '''
    return nullspace(H)


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
    return reduced_row_echelon(H)[0]


def codewords(H):
    '''
    Computes all of the the codewords of the code corresponding to the parity check matrix H. The codewords are given by the span of the nullspace.

    Parameters
    ----------

    H: numpy.ndarray
        A parity check matrix.

    Returns
    -------
    numpy.ndarray
        A matrix where each row corresponds to a codeword

    Note
    ----
    If you want to calculate a basis of the codewords use `ldpc.mod2.nullspace`.

    Examples
    --------
    >>> H=np.array([[0, 0, 0, 1, 1, 1, 1],[0, 1, 1, 0, 0, 1, 1],[1, 0, 1, 0, 1, 0, 1]])
    >>> print(codewords(H))
    [[0 0 0 0 0 0 0]
     [0 0 0 1 1 1 1]
     [0 0 1 0 1 1 0]
     [0 0 1 1 0 0 1]
     [0 1 0 0 1 0 1]
     [0 1 0 1 0 1 0]
     [0 1 1 0 0 1 1]
     [0 1 1 1 1 0 0]
     [1 0 0 0 0 1 1]
     [1 0 0 1 1 0 0]
     [1 0 1 0 1 0 1]
     [1 0 1 1 0 1 0]
     [1 1 0 0 1 1 0]
     [1 1 0 1 0 0 1]
     [1 1 1 0 0 0 0]
     [1 1 1 1 1 1 1]]
    '''
    _, n = H.shape
    zero_cw = np.zeros(n).astype(int)  # zero codewords
    cw = row_span(nullspace(H))  # nonzero codewords
    cw = np.vstack([zero_cw, cw])

    return cw


def compute_code_distance(H):
    '''
    Computes the distance of the code given by parity check matrix H. The code distance is given by the minimum weight of a nonzero codeword.

    Note
    ----
    The runtime of this function scales exponentially with the block size. In practice, computing the code distance of codes with block lengths greater than ~10 will be very slow.

    Parameters
    ----------
    H: numpy.ndarray
        The parity check matrix
    
    Returns
    -------
    int
        The code distance
    '''

    ker=nullspace(H)

    if len(ker)==0: return np.inf #return np.inf if the kernel is empty (eg. infinite code distance)

    cw=row_span(ker) #nonzero codewords

    return np.min(np.sum(cw, axis=1))


def get_code_parameters(H):
    """
    Returns the code parameters in [n,k,d] notation where n is the block length, k is the number of encoed bits and d is the code distance.

    Parameters
    ----------
    H: numpy.ndarray
        The parity check matrix

    Returns
    -------
    n: int
        The block length
    k: int
        The number of encoded bits
    d: int
        The code distance
    r: int
        The rank of the parity check matrix
    m: int
        The number of checks (rows in the parity check matrix)
    """

    m, n = H.shape
    r = rank(H)
    k = n - r
    d = compute_code_distance(H)

    return n, k, d, r, m



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

    cols, rows = compute_column_row_weights(H)

    return np.max(cols), np.max(rows)

if __name__ == "__main__":
    import doctest
    # doctest.testmod(verbose=True)
    from ldpc.mod2 import nullspace
    from ldpc.codes import hamming_code

    H=np.array([[0, 0, 0, 1, 1, 1, 1],[0, 1, 1, 0, 0, 1, 1],[1, 0, 1, 0, 1, 0, 1]])

    construct_generator_matrix(H)
    H=hamming_code(5)

    cycles=search_cycles(H,6,terminate=False)
    print(cycles)
    cycles=0
    m,n=H.shape
    for i in range(m):
        cycles+=search_cycles(H,6,row=i,terminate=False)
    print(cycles)

    # girth=4
    # print(list(combinations([k for k in range(m)], girth // 2)))

