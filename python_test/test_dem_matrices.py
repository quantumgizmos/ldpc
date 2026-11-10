import pathlib
import tempfile
import stim
import numpy as np
from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices
from ldpc.sinter_decoders import SinterBpOsdDecoder


def test_dem_matrices_zero_prob_error():
    dem_control = stim.DetectorErrorModel("error(0.125) D0 L0\n")
    m_control = detector_error_model_to_check_matrices(dem_control)

    dem_zero = stim.DetectorErrorModel("error(0.125) D0 L0\nerror(0) D0\n")
    m_zero = detector_error_model_to_check_matrices(dem_zero)

    assert np.array_equal(m_control.observables_matrix.toarray(), m_zero.observables_matrix.toarray())
    assert np.array_equal(m_control.priors, m_zero.priors)
    assert np.array_equal(m_control.check_matrix.toarray(), m_zero.check_matrix.toarray())

    # Explicit matrix and priors assertions
    assert np.array_equal(m_zero.check_matrix.toarray(), [[1]])
    assert np.array_equal(m_zero.observables_matrix.toarray(), [[1]])
    assert np.array_equal(m_zero.priors, [0.125])


def test_sinter_bposd_decoder_zero_prob_error():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        dem_path = tmp_path / "model.dem"
        dem_path.write_text("error(0.125) D0 L0\nerror(0) D0\n")
        dets_path = tmp_path / "dets.b8"
        obs_path = tmp_path / "obs.b8"
        # Detector bytes 00 01 (two shots, detector 0: shot 0=0, shot 1=1)
        dets_path.write_bytes(bytes([0x00, 0x01]))

        decoder = SinterBpOsdDecoder(max_iter=5, bp_method="ms", osd_method="osd_cs", osd_order=0)
        decoder.decode_via_files(
            num_shots=2,
            num_dets=1,
            num_obs=1,
            dem_path=dem_path,
            dets_b8_in_path=dets_path,
            obs_predictions_b8_out_path=obs_path,
            tmp_dir=tmp_path / "work",
        )
        # Prediction bytes 00 01 (shot 0=0, shot 1=1)
        assert obs_path.read_bytes() == bytes([0x00, 0x01])
