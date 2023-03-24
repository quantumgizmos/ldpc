import pytest
import numpy as np
from ldpc2.bp_decoder import bp_decoder

def test_constructor():
    # test valid input parameters
    pcm = np.array([[1,0,1,0],[0,1,1,0],[0,0,1,1]])
    decoder = bp_decoder(pcm, error_rate=0.1, max_iter=10, bp_method='product_sum')
    assert decoder.m == 3
    assert decoder.n == 4
    assert decoder.bp_method == 0
    assert decoder.max_iter == 10
    assert decoder.ms_scaling_factor == 1.0
    assert decoder.schedule == 0
    assert decoder.omp_thread_count == 1
    assert decoder.random_serial_schedule == 0
    assert decoder.serial_schedule_order is None

    # test invalid input parameters
    with pytest.raises(TypeError):
        decoder = bp_decoder(5)  # invalid pcm input
    with pytest.raises(ValueError):
        decoder = bp_decoder(pcm, error_rate='0.1')  # invalid error_rate input
        decoder = bp_decoder(pcm, max_iter=-1)  # invalid max_iter input
        decoder = bp_decoder(pcm, bp_method='invalid')  # invalid bp_method input
        decoder = bp_decoder(pcm, ms_scaling_factor='1.0')  # invalid ms_scaling_factor input
        decoder = bp_decoder(pcm, schedule='invalid')  # invalid schedule input
        decoder = bp_decoder(pcm, omp_thread_count='1')  # invalid omp_thread_count input
        decoder = bp_decoder(pcm, serial_schedule_order=[0,1])  # invalid serial_schedule_order input

# if __name__ == "__main__":
#     test_constructor()