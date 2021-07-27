import numpy as np
from itertools import combinations
from math import factorial as fact
from .mod2 import reduced_row_echelon, nullspace, row_span, rank

#nCr combinations function
def nCr(n,r): return int(fact(n) // fact(r) // fact(n-r))

def construct_generator_matrix(H):
    '''
    Constructs a generator matrix from a parity check H.
    
    The generator matrix G satisfies the condition:
                    $$H@G.T = 0$$.
    
    Each of the columns of the generator matrix is a nullspace vector of
    the matrix H
    
    
    Parameters
    ----------

    H: numpy.ndarray
        A binary matrix in numpy.ndarray format.
    
    Returns
    -------

    numpy.ndarray
        The generator matrix in numpy.ndarray format
    
    '''
    return nullspace(H)


def systematic_form(H):
    '''
    Converts H into systematic form so that H=[I,A]
    '''
    return reduced_row_echelon(H)[0]


def codewords(H):
    '''
    Computes the codewords of the code corresponding to the parity check matrix H
    -- The codewords are given by the span of the nullspace
    '''
    _, n = H.shape
    zero_cw = np.zeros(n).astype(int)  # zero codewords
    cw = row_span(nullspace(H))  # nonzero codewords
    cw = np.vstack([zero_cw, cw])

    return cw


def compute_code_distance(H):
    '''
    Computes the distance of the code given by parity check matrix H.
    -- The code distance is given by the mininum weight of a nonzero codeword
    '''

    ker=nullspace(H)

    if len(ker)==0: return np.inf #return np.inf if the kernel is empty (eg. infinite code distance)

    cw=row_span(ker) #nonzero codewords

    return np.min(np.sum(cw, axis=1))


def get_code_parameters(H):
    m, n = H.shape
    r = rank(H)
    k = n - r
    d = compute_code_distance(H)

    return n, k, d, r, m



def search_cycles(H, girth,terminate=False):

    m, n = np.shape(H)

    cycle_count = 0

    for i, combination in enumerate(combinations([k for k in range(m)], girth // 2)):
        row_sum = np.zeros(n).astype(int)
        for j, element in enumerate(combination):
            row_sum += H[element]

        two_count = np.count_nonzero(row_sum == 2)
        if two_count >= girth // 2:
            if terminate: return True
            cycle_count += nCr(two_count, girth // 2)

    if terminate: return False #terminates if the code is not cycle free
    return cycle_count


def compute_column_row_weights(H):

    return np.sum(H,axis=0), np.sum(H,axis=1)

def get_ldpc_params(H):

    cols, rows = compute_column_row_weights(H)

    return np.max(cols), np.max(rows)

