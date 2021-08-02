import numpy as np
from ldpc.mod2 import mod10_to_mod2


def hamming_code(rank):
    """
    Outputs a Hamming code parity check matrix given its rank.
    
    Parameters
    ----------
    rank: int
        The rank of of the Hamming code parity check matrix.

    Returns
    -------
    numpy.ndarray
        The Hamming code parity check matrix in numpy.ndarray format. 

    
    Example
    -------
    >>> print(hamming_code(3))
    [[0 0 0 1 1 1 1]
     [0 1 1 0 0 1 1]
     [1 0 1 0 1 0 1]]

    """
    rank = int(rank)
    num_rows = (2 ** rank) - 1

    pc_matrix = np.zeros((num_rows, rank), dtype=int)

    for i in range(0, num_rows):
        pc_matrix[i] = mod10_to_mod2(i + 1, rank)

    return pc_matrix.T


def rep_code(distance):
    """
    Outputs repetition code parity check matrix for specified distance.

    Parameters
    ----------
    distance: int
        The distance of the repetition code.


    Returns
    -------
    numpy.ndarray
        The repetition code parity check matrix in numpy.ndarray format.

    Examples
    --------
    >>> print(rep_code(5))
    [[1 1 0 0 0]
     [0 1 1 0 0]
     [0 0 1 1 0]
     [0 0 0 1 1]]
    """

    pcm = np.zeros((distance - 1, distance), dtype=int)

    for i in range(distance - 1):
        pcm[i, i] = 1
        pcm[i, i + 1] = 1

    return pcm


def ring_code(distance):
    """
    Outputs ring code (closed-loop repetion code) parity check matrix
    for a specified distance. 

    Parameters
    ----------
    distance: int
        The distance of the repetition code.

    Returns
    -------
    numpy.ndarray
        The repetition code parity check matrix in numpy.ndarray format.

    Examples
    --------
    >>> print(ring_code(5))
    [[1 1 0 0 0]
     [0 1 1 0 0]
     [0 0 1 1 0]
     [0 0 0 1 1]
     [1 0 0 0 1]]
    """

    pcm = np.zeros((distance, distance), dtype=int)

    for i in range(distance - 1):
        pcm[i, i] = 1
        pcm[i, i + 1] = 1

    # close the loop
    i = distance - 1
    pcm[i, 0] = 1
    pcm[i, i] = 1

    return pcm


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)