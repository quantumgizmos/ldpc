import pytest
import numpy as np
from ldpc.bp_decoder._bp_decoder import BpDecoder

def test_dynamic_scaling_factor_damping_initialization():
    pcm = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    channel_probs = [0.1, 0.2, 0.3]
    max_iter = 10
    damping_factor = 0.1

    decoder = BpDecoder(pcm, channel_probs=channel_probs, max_iter=max_iter, dynamic_scaling_factor_damping=damping_factor)

    assert decoder.dynamic_scaling_factor_damping == damping_factor, "Dynamic scaling factor damping not set correctly."

def test_dynamic_scaling_factor_damping_effect():
    pcm = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    channel_probs = [0.1, 0.2, 0.3]
    max_iter = 10
    damping_factor = 0.1

    decoder = BpDecoder(pcm, channel_probs=channel_probs, max_iter=max_iter, dynamic_scaling_factor_damping=damping_factor)

    # Verify that the scaling factors are computed correctly
    expected_factors = [
        1.0 - (1.0 - decoder.ms_scaling_factor) * (2.0 ** (-1 * i * damping_factor))
        for i in range(max_iter)
    ]
    for i, factor in enumerate(expected_factors):
        assert pytest.approx(decoder.ms_scaling_factor_vector[i], rel=1e-6) == factor, f"Scaling factor mismatch at iteration {i}."

def test_dynamic_scaling_factor_damping_update():
    pcm = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    channel_probs = [0.1, 0.2, 0.3]
    max_iter = 10
    initial_damping_factor = 0.1
    updated_damping_factor = 0.2

    decoder = BpDecoder(pcm, channel_probs=channel_probs, max_iter=max_iter, dynamic_scaling_factor_damping=initial_damping_factor)

    # Update the damping factor
    decoder.dynamic_scaling_factor_damping = updated_damping_factor

    # Verify that the scaling factors are recomputed correctly
    expected_factors = [
        1.0 - (1.0 - decoder.ms_scaling_factor) * (2.0 ** (-1 * i * updated_damping_factor))
        for i in range(max_iter)
    ]
    for i, factor in enumerate(expected_factors):
        assert pytest.approx(decoder.ms_scaling_factor_vector[i], rel=1e-6) == factor, f"Scaling factor mismatch at iteration {i} after update."
