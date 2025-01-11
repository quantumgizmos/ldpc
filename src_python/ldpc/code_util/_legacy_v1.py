from ldpc.code_util import compute_exact_code_distance


def compute_code_distance(H):
    """
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
    """

    return compute_exact_code_distance(H)
