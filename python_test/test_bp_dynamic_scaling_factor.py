import pytest
import numpy as np
from ldpc.bp_decoder._bp_decoder import BpDecoder
from ldpc import BpOsdDecoder
from ldpc.bplsd_decoder import BpLsdDecoder
from ldpc.belief_find_decoder import BeliefFindDecoder
from ldpc.bp_flip import BpFlipDecoder

def test_dynamic_scaling_factor_damping_initialization():
    pcm = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    channel_probs = [0.1, 0.2, 0.3]
    max_iter = 10
    damping_factor = 0.1

    decoder = BpDecoder(pcm, channel_probs=channel_probs, bp_method="ms", max_iter=max_iter, dynamic_scaling_factor_damping=damping_factor)

    assert decoder.dynamic_scaling_factor_damping == damping_factor, "Dynamic scaling factor damping not set correctly."

def test_dynamic_scaling_factor_damping_effect():
    pcm = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    channel_probs = [0.1, 0.2, 0.3]
    max_iter = 10
    damping_factor = 0.1

    decoder = BpDecoder(pcm, channel_probs=channel_probs, bp_method="ms", max_iter=max_iter, dynamic_scaling_factor_damping=damping_factor)

    # Verify that the scaling factors are computed correctly
    expected_factors = [
        1.0 - (1.0 - decoder.ms_scaling_factor) * (2.0 ** (-1 * i * damping_factor))
        for i in range(max_iter)
    ]
    for i, factor in enumerate(expected_factors):
        assert pytest.approx(decoder.ms_scaling_factor_vector[i], rel=1e-6) == factor, f"Scaling factor mismatch at iteration {i}."

def test_dynamic_scaling_factor_with_initial_and_converge_parameters():
    pcm = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    channel_probs = [0.1, 0.2, 0.3]
    
    max_iter = 10
    damping_factor = 0.1
    ms_scaling_factor = 0.5  # Initial scaling factor for testing
    ms_converge_value = 2.0  # Convergence value for the minimum-sum method

    decoder = BpDecoder(pcm, channel_probs=channel_probs, max_iter=max_iter,  bp_method="ms", ms_scaling_factor=ms_scaling_factor, dynamic_scaling_factor_damping=damping_factor, ms_converge_value=ms_converge_value)

    print("ms_scaling_factor_start:", decoder.ms_scaling_factor)
    print("damping_factor:", damping_factor)
    print("ms_converge_value:", decoder.ms_converge_value)
    print("Initial scaling factors:", decoder.ms_scaling_factor_vector)


    # Verify that the scaling factors are recomputed correctly
    expected_factors = [
        ms_converge_value - (ms_converge_value - ms_scaling_factor) * (2.0 ** (-1 * i * damping_factor))
        for i in range(max_iter)
    ]
    for i, factor in enumerate(expected_factors):
        assert pytest.approx(decoder.ms_scaling_factor_vector[i], rel=1e-6) == factor, f"Scaling factor mismatch at iteration {i} after update."

def test_dynamic_scaling_factor_damping_bplsd():
    pcm = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    damping_factor = 0.2

    decoder = BpLsdDecoder(pcm, error_rate = 0.1, bp_method="ms", dynamic_scaling_factor_damping=damping_factor)
    assert decoder.dynamic_scaling_factor_damping == damping_factor, "Damping factor not set correctly."

    updated_damping_factor = 0.5
    decoder.dynamic_scaling_factor_damping = updated_damping_factor
    assert decoder.dynamic_scaling_factor_damping == updated_damping_factor, "Damping factor update failed."


def test_dynamic_scaling_factor_damping_bposd():
    pcm = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    damping_factor = 0.3

    decoder = BpOsdDecoder(pcm, error_rate=0.1, bp_method="ms", dynamic_scaling_factor_damping=damping_factor)
    assert decoder.dynamic_scaling_factor_damping == damping_factor, "Damping factor not set correctly."

    updated_damping_factor = 0.6
    decoder.dynamic_scaling_factor_damping = updated_damping_factor
    assert decoder.dynamic_scaling_factor_damping == updated_damping_factor, "Damping factor update failed."

def test_dynamic_scaling_factor_damping_belief_find():
    pcm = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    damping_factor = 0.4

    decoder = BeliefFindDecoder(pcm, error_rate=0.1, bp_method="ms", dynamic_scaling_factor_damping=damping_factor)
    assert decoder.dynamic_scaling_factor_damping == damping_factor, "Damping factor not set correctly."

    updated_damping_factor = 0.7
    decoder.dynamic_scaling_factor_damping = updated_damping_factor
    assert decoder.dynamic_scaling_factor_damping == updated_damping_factor, "Damping factor update failed."

