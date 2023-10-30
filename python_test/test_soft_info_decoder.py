import pytest
from ldpc.bp_decoder import SoftInfoBpDecoder
import numpy as np


def test_errored_close_to_zero():
    """
    Test decoding a 3-qubit ring code with an errored zero syndrome
    """
    # Setup repetition code
    n = 3
    pcm = np.eye(n, dtype=int)
    pcm += np.roll(pcm, 1, axis=1)

    cutoff = 10
    sbpd = SoftInfoBpDecoder(pcm, error_rate=0.1, max_iter=n, ms_scaling_factor=1.0, cutoff=10.0)

    soft_syndrome = np.full(n, 2)
    soft_syndrome[0] = -1
    soft_syndrome[1] = 1  # syndrome is incorrect, but only just
    soft_decoding = sbpd.decode(soft_syndrome)

    assert np.array_equal(soft_decoding, np.zeros(n, dtype=int))


def test_one_errored_syndrome_bit():
    """
    Test decoding a 3-qubit ring code with an errored syndrome
    """
    # Setup repetition code
    n = 3
    pcm = np.eye(n, dtype=int)
    pcm += np.roll(pcm, 1, axis=1)

    sbpd = SoftInfoBpDecoder(pcm, error_rate=0.1, max_iter=n, ms_scaling_factor=1.0, cutoff=10.0)

    soft_syndrome = np.array([-20, 1, 20])
    expected_decoding = np.array([0, 1, 0])
    soft_decoding = sbpd.decode(soft_syndrome)

    assert np.array_equal(soft_decoding, expected_decoding)


def test_long_rep_code():
    n = 20
    pcm = np.eye(n, dtype=int)
    pcm += np.roll(pcm, 1, axis=1)
    sbpd = SoftInfoBpDecoder(pcm, error_rate=0.1, max_iter=n, ms_scaling_factor=1.0, cutoff=10.0)

    soft_syndrome = np.full(n, 10)
    soft_syndrome[0] = -20
    soft_syndrome[1] = 1
    expected_decoding = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    soft_decoding = sbpd.decode(soft_syndrome)

    assert np.array_equal(soft_decoding, expected_decoding)


def test_hamming_code():
    """
   Test decoding a Hamming code with errored syndrome
   """
    # Setup repetition code
    n = 20
    pcm = np.array([
        [1, 0, 0, 1, 1, 0, 1],
        [0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 0, 1, 1]
    ])

    sbpd = SoftInfoBpDecoder(pcm, error_rate=0.1, max_iter=n, ms_scaling_factor=1.0, cutoff=10.0)

    soft_syndrome = np.array([20, -20, -11])
    expected_decoding = np.array([0, 0, 0, 0, 0, 1, 0])
    soft_decoding = sbpd.decode(soft_syndrome)

    assert np.array_equal(soft_decoding, expected_decoding)


if __name__ == "__main__":
    pytest.main([__file__])
