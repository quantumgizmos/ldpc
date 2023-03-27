import pytest
import numpy as np
from scipy.sparse import csr_matrix
from ldpc2.bp_decoder import bp_decoder

def test_constructor():
    # test valid input parameters
    pcm = np.array([[1,0,1,0],[0,1,1,0],[0,0,1,1]])
    decoder = bp_decoder(pcm, error_rate=0.1, max_iter=10, bp_method='product_sum')
    assert decoder.m == 3
    assert decoder.n == 4
    assert decoder.bp_method == "product_sum"
    assert decoder.max_iter == 10
    assert decoder.ms_scaling_factor == 1.0
    assert decoder.schedule == "parallel"
    assert decoder.omp_thread_count == 1
    assert decoder.random_serial_schedule == 0
    assert np.array_equal(decoder.serial_schedule_order, np.array([0, 1, 2, 3]))

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

def test_bp_decoder_init():

    # test with numpy ndarray as pcm
    pcm = np.array([[1, 0, 1], [0, 1, 1]])
    decoder = bp_decoder(pcm, error_rate=0.1, max_iter=10, bp_method='prod_sum', ms_scaling_factor=0.5, schedule='parallel', omp_thread_count=4, random_serial_schedule=1, serial_schedule_order=[1,2,0])
    assert decoder is not None
    assert decoder.m == 2
    assert decoder.n == 3
    assert decoder.max_iter == 10
    assert decoder.bp_method == "product_sum"
    assert decoder.ms_scaling_factor == 0.5
    assert decoder.schedule == "parallel"
    assert decoder.omp_thread_count == 4
    assert decoder.random_serial_schedule == 1
    assert np.array_equal(decoder.serial_schedule_order, np.array([1, 2, 0]))

    # test with scipy.sparse csr_matrix as pcm
    pcm = csr_matrix([[1, 0, 1], [0, 1, 1]])
    decoder = bp_decoder(pcm, error_channel=[0.1, 0.2, 0.3])
    assert decoder is not None
    assert decoder.m == 2
    assert decoder.n == 3
    assert decoder.max_iter == 3
    assert decoder.bp_method == "product_sum"
    assert decoder.ms_scaling_factor == 1.0
    assert decoder.schedule == "parallel"
    assert decoder.omp_thread_count == 1
    assert decoder.random_serial_schedule == 0
    assert np.array_equal(decoder.serial_schedule_order, np.array([0, 1, 2]))


    # test with invalid pcm type
    with pytest.raises(TypeError):
        decoder = bp_decoder('invalid', error_rate=0.1)

    # test with invalid max_iter type
    with pytest.raises(ValueError):
        decoder = bp_decoder(pcm, error_rate=0.1,max_iter='invalid')

    # test with invalid max_iter value
    with pytest.raises(ValueError):
        decoder = bp_decoder(pcm, error_rate =0.1, max_iter=-1)

    # test with invalid bp_method value
    with pytest.raises(ValueError):
        decoder = bp_decoder(pcm,error_rate=0.1, bp_method='invalid')

    # test with invalid schedule value
    with pytest.raises(ValueError):
        decoder = bp_decoder(pcm,error_rate=0.1, schedule='invalid')

    # test with invalid ms_scaling_factor value
    with pytest.raises(TypeError):
        decoder = bp_decoder(pcm,error_rate=0.1, ms_scaling_factor='invalid')

    # test with invalid omp_thread_count value
    with pytest.raises(TypeError):
        decoder = bp_decoder(pcm, error_rate=0.1,omp_thread_count='invalid')

    # test with invalid random_serial_schedule value
    with pytest.raises(TypeError):
        decoder = bp_decoder(pcm, error_rate=0.1, random_serial_schedule='invalid')

    # test with invalid serial_schedule_order value
    with pytest.raises(ValueError):
        decoder = bp_decoder(pcm, error_rate=0.1, serial_schedule_order=[1, 2])


# if __name__ == "__main__":
#     test_constructor()