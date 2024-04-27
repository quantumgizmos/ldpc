import numpy as np
from ldpc.bp_decoder import BpDecoder
import ldpc.codes

def test_bp_gd():

    H = ldpc.codes.ring_code(6)
    error = np.array([1,1,1,0,0,0])
    syndrome = H@error % 2

    decoder = BpDecoder(H, error_rate = 0.01, bp_method="ms", ms_scaling_factor = 0.625, max_iter=100, input_vector_type="syndrome")

    # Test decoding
    decoded = decoder.gd_decode(syndrome,10)
    assert decoder.converge == True
    assert np.array_equal(syndrome, H@decoded % 2)

if __name__ == '__main__':
    test_bp_gd()
