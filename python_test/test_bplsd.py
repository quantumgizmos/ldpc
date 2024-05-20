import pytest
import numpy as np
import scipy.sparse
from ldpc.codes import rep_code, ring_code, hamming_code
from ldpc.bplsd_decoder import BpLsdDecoder

# Define valid inputs for testing
valid_pcms = [
    np.array([[1, 0, 1], [0, 1, 1]]),
    scipy.sparse.csr_matrix([[1, 0, 1], [0, 1, 1]]),
]

valid_error_rates = [
    0.1,
]

valid_max_iters = [
    0,
    10,
]

valid_bp_methods = [
    "product_sum",
    "minimum_sum",
]

valid_ms_scaling_factors = [
    1.0,
    0.5,
]

valid_schedules = [
    "parallel",
    "serial",
]

valid_omp_thread_counts = [
    1,
    4,
]

valid_random_schedule_seeds = [
    42,
]

valid_serial_schedule_orders = [
    None,
    [1, 0, 2],
]

# Combine valid inputs for parameterized testing
valid_input_permutations = pytest.mark.parametrize(
    "pcm, error_rate, max_iter, bp_method, ms_scaling_factor, schedule, omp_thread_count, random_schedule_seed, serial_schedule_order",
    [
        (pcm, error, max_iter, bp_method, ms_factor, schedule, omp_count, seed, order)
        for pcm in valid_pcms
        for error in valid_error_rates
        for max_iter in valid_max_iters
        for bp_method in valid_bp_methods
        for ms_factor in valid_ms_scaling_factors
        for schedule in valid_schedules
        for omp_count in valid_omp_thread_counts
        for seed in valid_random_schedule_seeds
        for order in valid_serial_schedule_orders
    ],
)

def test_BpLsdDecoder_init():

    # test with numpy ndarray as pcm
    pcm = np.array([[1, 0, 1], [0, 1, 1]])
    decoder = BpLsdDecoder(pcm, error_rate=0.1, max_iter=10, bp_method='prod_sum', ms_scaling_factor=0.5, schedule='parallel', omp_thread_count=4, random_schedule_seed=1, serial_schedule_order=[1,2,0],input_vector_type = "syndrome")
    assert decoder is not None
    assert decoder.check_count == 2
    assert decoder.bit_count == 3
    assert decoder.max_iter == 10
    assert decoder.bp_method == "product_sum"
    assert decoder.ms_scaling_factor == 0.5
    assert decoder.schedule == "parallel"
    assert decoder.omp_thread_count == 4
    assert decoder.random_schedule_seed == 1
    assert np.array_equal(decoder.serial_schedule_order, np.array([1, 2, 0]))
    assert np.array_equal(decoder.input_vector_type, "syndrome")

    # test with scipy.sparse scipy.sparse.csr_matrix as pcm
    pcm = scipy.sparse.csr_matrix([[1, 0, 1], [0, 1, 1]])
    decoder = BpLsdDecoder(pcm, error_channel=[0.1, 0.2, 0.3],input_vector_type = "syndrome")
    assert decoder is not None
    assert decoder.check_count == 2
    assert decoder.bit_count == 3
    assert decoder.max_iter == 3
    assert decoder.bp_method == "product_sum"
    assert decoder.ms_scaling_factor == 1.0
    assert decoder.schedule == "parallel"
    assert decoder.omp_thread_count == 1
    assert decoder.random_schedule_seed == 0
    assert np.array_equal(decoder.serial_schedule_order, np.array([0, 1, 2]))
    assert np.array_equal(decoder.input_vector_type, "syndrome")


    # test with invalid pcm type
    with pytest.raises(TypeError):
        decoder = BpLsdDecoder('invalid', error_rate=0.1)

    # test with invalid max_iter type
    with pytest.raises(TypeError):
        decoder = BpLsdDecoder(pcm, error_rate=0.1,max_iter='invalid')

    # test with invalid max_iter value
    with pytest.raises(ValueError):
        decoder = BpLsdDecoder(pcm, error_rate =0.1, max_iter=-1)

    # test with invalid bp_method value
    with pytest.raises(ValueError):
        decoder = BpLsdDecoder(pcm,error_rate=0.1, bp_method='invalid')

    # test with invalid schedule value
    with pytest.raises(ValueError):
        decoder = BpLsdDecoder(pcm,error_rate=0.1, schedule='invalid')

    # test with invalid ms_scaling_factor value
    with pytest.raises(TypeError):
        decoder = BpLsdDecoder(pcm,error_rate=0.1, ms_scaling_factor='invalid')

    # test with invalid omp_thread_count value
    with pytest.raises(TypeError):
        decoder = BpLsdDecoder(pcm, error_rate=0.1,omp_thread_count='invalid')

    # test with invalid random_schedule_seed value
    with pytest.raises(TypeError):
        decoder = BpLsdDecoder(pcm, error_rate=0.1, random_schedule_seed='invalid')

    # test with invalid serial_schedule_order value
    with pytest.raises(Exception):
        decoder = BpLsdDecoder(pcm, error_rate=0.1, serial_schedule_order=[1, 2])

def test_rep_code_ms():

    H = rep_code(3)

    lsd = BpLsdDecoder(H,error_rate=0.1, bp_method='min_sum', ms_scaling_factor=1.0)
    assert lsd is not None
    assert lsd.bp_method == "minimum_sum"
    assert lsd.schedule == "parallel"
    assert np.array_equal(lsd.error_channel,np.array([0.1, 0.1, 0.1]))


    decoding = lsd.decode(np.array([1, 1]))
    assert(np.array_equal(decoding, np.array([0, 1,0])))

    lsd.error_channel = np.array([0.1, 0, 0.1])
    assert np.array_equal(lsd.error_channel,np.array([0.1, 0, 0.1]))

    decoding=lsd.decode(np.array([1, 1]))
    assert(np.array_equal(decoding, np.array([1, 0, 1])))

def test_stats_reset():

    H = rep_code(5)

    lsd = BpLsdDecoder(H,max_iter=1,error_rate=0.1, bp_method='min_sum', ms_scaling_factor=1.0)
    lsd.set_do_stats(True)
    syndrome = np.array([1,1,0,1])
    lsd.decode(syndrome)

    stats = lsd.statistics
    assert stats['lsd_order'] == 0
    assert stats["lsd_method"] == 1
    assert len(stats["bit_llrs"]) == H.shape[1]
    assert len(stats["individual_cluster_stats"])>0
    assert len(stats["global_timestep_bit_history"])>0

    syndrome = np.array([0,0,0,0])
    lsd.decode(syndrome)

    stats = lsd.statistics
    assert len(stats["bit_llrs"]) == 0
    assert len(stats["individual_cluster_stats"])==0
    assert len(stats["global_timestep_bit_history"])==0
