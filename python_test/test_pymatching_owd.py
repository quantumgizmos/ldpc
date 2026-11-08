import numpy as np
import pytest
import pymatching
import stim
from ldpc.ckt_noise.sinter_overlapping_window_decoder import SinterDecoder_PyMatching_OWD


def test_pymatching_owd_weights_log_odds():
    # Verify that PyMatchingOverlappingWindowDecoder computes weights as log((1-p)/p)
    # matching the Bernoulli log-odds definition in PyMatching.
    dem_text = "error(0.01) D0 D1 L0\nerror(0.1) D0\nerror(0.1) D1\n"
    dem = stim.DetectorErrorModel(dem_text)

    compiled = SinterDecoder_PyMatching_OWD(
        window=2, commit=1, decodings=1, num_checks=1
    ).compile_decoder_for_dem(dem=dem)

    # Syndrome [1, 1] packed as binary 3 (0b11)
    prediction = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=np.array([[3]], dtype=np.uint8)
    )

    baseline = pymatching.Matching.from_detector_error_model(dem).decode(
        np.array([1, 1], dtype=np.uint8)
    )

    assert prediction[0, 0] == baseline[0] == 0
