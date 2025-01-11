import numpy as np


def generate_bsc_error(n: int, error_rate: float) -> np.ndarray:
    """
    Generate a numpy array of binary symmetric channel (BSC) errors.

    Parameters:
        n (int): The length of the array to generate.
        error_rate (float): The probability of a bit being flipped.

    Returns:
        np.ndarray: An array of length `n` containing 0s and 1s with a probability
        of `error_rate` of each bit being flipped.

    Example:
        >>> generate_bsc_error(5, 0.1)
        array([0, 0, 1, 0, 0], dtype=uint8)

        >>> generate_bsc_error(10, 0.3)
        array([1, 0, 0, 1, 1, 1, 0, 0, 0, 1], dtype=uint8)
    """
    return np.random.binomial(1, error_rate, n).astype(np.uint8)
