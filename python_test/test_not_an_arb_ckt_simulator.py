from scipy.sparse import csr_matrix
import numpy as np

from ldpc.ckt_noise.not_an_arb_ckt_simulator import get_stabilizer_time_steps


def test_get_stabilizer_time_steps():
    M = csr_matrix([[1, 1, 0], [1, 1, 0], [1, 1, 1]], dtype=np.uint8)

    time_steps, measured_bits = get_stabilizer_time_steps(M)
    time_steps_expected = [[None, 1, 0], [1, 0, 2], [0, None, 1]]
    measured_bits_expected = [[None, 1, 0], [1, 0, None], [0, 2, 1]]
    assert time_steps == time_steps_expected
    assert measured_bits == measured_bits_expected
