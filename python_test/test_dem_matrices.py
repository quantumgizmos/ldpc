import stim
import numpy as np
from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices


def test_dem_matrices_zero_prob_error():
    dem_control = stim.DetectorErrorModel("error(0.125) D0 L0\n")
    m_control = detector_error_model_to_check_matrices(dem_control)

    dem_zero = stim.DetectorErrorModel("error(0.125) D0 L0\nerror(0) D0\n")
    m_zero = detector_error_model_to_check_matrices(dem_zero)

    assert np.array_equal(m_control.observables_matrix.toarray(), m_zero.observables_matrix.toarray())
    assert np.array_equal(m_control.priors, m_zero.priors)
    assert np.array_equal(m_control.check_matrix.toarray(), m_zero.check_matrix.toarray())
