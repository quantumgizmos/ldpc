from not_an_arb_ckt_simulator import (
    rep_code,
    get_stabilizer_time_steps,
    stim_circuit_from_time_steps,
)
import numpy as np
from typing import Union, List, Union
from scipy.sparse import csr_matrix
import stim
import numpy as np
from pymatching import Matching
import sinter
from time import time
import matplotlib.pyplot as plt
from pymatching import Matching

from dem_matrices import detector_error_model_to_check_matrices


def overlapping_window(
    p: float,
    pcm: csr_matrix,
    logicals: csr_matrix,
    decodings: int,
    window: int,
    commit: int,
    decompose_errors: bool = True,
    shots: int = 10_000,
):

    # define relevant parameters
    num_checks, num_bits = pcm.shape
    rounds = (window - commit) + decodings * commit

    # create the circuit to sample from
    time_steps, measurement_qubits = get_stabilizer_time_steps(pcm)
    circuit = stim_circuit_from_time_steps(
        pcm,
        logicals,
        time_steps,
        measurement_qubits,
        rounds=rounds - 2,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )

    # create the detector error model
    dem = circuit.detector_error_model(decompose_errors=decompose_errors)
    sampler = dem.compile_sampler()

    # create relevant detector check matrices and observables
    dem_matrices = detector_error_model_to_check_matrices(dem)

    # sample from the detector error model
    detector_data, obs_data, _ = sampler.sample(shots=shots)
    obs_data = obs_data.astype(np.uint8)
    detector_data = detector_data.astype(np.uint8)
    # decode the data
    total_errs = 0

    weights = np.log1p(dem_matrices.priors) - np.log(dem_matrices.priors)
    eps = 1e-14
    min_weight = np.log1p(eps) - np.log(eps)

    # dense version of the correction
    total_corr = np.zeros((shots, dem_matrices.check_matrix.shape[1]), dtype=np.bool_)

    for decoding in range(decodings):
        # for obs, sample in zip(obs_data, detector_data):
        commit_inds, dec_inds, synd_commit_inds, synd_dec_inds = current_round_inds(
            dem_matrices.check_matrix, decoding, window, commit, num_checks
        )

        decoder = Matching.from_check_matrix(
            dem_matrices.check_matrix[synd_dec_inds, :],
            weights=weights,
        )
        for i in range(shots):
            corr = decoder.decode(detector_data[i][synd_dec_inds])

            if decoding != decodings - 1:
                # determine the partial correction / commit the correction
                total_corr[i][commit_inds] = corr[commit_inds]
                # modify syndrome to reflect the correction
                detector_data[i][synd_dec_inds] ^= (
                    dem_matrices.check_matrix @ total_corr[i] % 2
                )[synd_dec_inds]

            else:
                # This is the final decoding, commit all
                total_corr[i][dec_inds] = corr[dec_inds]

        # once all shots have been decoded for this round, update the weights
        weights[commit_inds] = min_weight

    for shot in range(shots):
        if not np.array_equal(
            (dem_matrices.observables_matrix @ total_corr[shot]) % 2, obs_data[shot]
        ):
            total_errs += 1

    # for obs, prediction in zip(obs_data, predictions):
    #     if not np.array_equal(prediction, obs):
    #         total_errs += 1

    return total_errs / shots


def current_round_inds(
    dcm: csr_matrix,
    decoding: int,
    window: int,
    commit: int,
    num_checks: int,
) -> np.ndarray:
    """
    Get the indices of the current round in the detector syndrome.
    """
    # detector indices or dcm.shape[0] indices
    num_checks_decoding = num_checks * window
    num_checks_commit = num_checks * commit
    start = decoding * commit * num_checks
    end_commit = start + num_checks_commit
    end_decoding = start + num_checks_decoding

    # DCM.shape[1] indices / ERROR MECHANICSM
    min_index = dcm[start, :].nonzero()[1][0]

    if end_commit < dcm.shape[0]:
        max_index_commit = dcm[end_commit, :].nonzero()[1][-1]
    else:
        max_index_commit = dcm.shape[1] - 1

    if end_decoding < dcm.shape[0]:
        max_index_decoding = dcm[end_decoding, :].nonzero()[1][-1]
    else:
        max_index_decoding = dcm.shape[1] - 1

    # error mechanism indices
    # commit_inds = np.arange(min_index, max_index_commit + 1)
    # decoding_inds = np.arange(min_index, max_index_decoding + 1)

    # use slices instead of np.arange
    commit_inds = slice(min_index, max_index_commit + 1)
    decoding_inds = slice(min_index, max_index_decoding + 1)

    # detector indices
    # synd_commit_inds = np.arange(start, end_commit)
    # synd_decoding_inds = np.arange(start, end_decoding)

    # use slices instead of np.arange
    synd_commit_inds = slice(start, end_commit)
    synd_decoding_inds = slice(start, end_decoding)

    return commit_inds, decoding_inds, synd_commit_inds, synd_decoding_inds


if __name__ == "__main__":

    for decodings in [2, 3, 4, 5]:
        fig, ax = plt.subplots()
        ps = np.geomspace(0.02, 0.05, 6)
        for d in [5, 9, 13]:
            pcm, logicals = rep_code(d)
            # errs = overlapping_window(0.04, pcm, logicals, 1, 2 * d, 2 * d)
            error_rates = [
                overlapping_window(p, pcm, logicals, decodings, 2 * d, d) for p in ps
            ]
            ax.plot(ps, error_rates, label=f"d={d}", marker="o")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Physical Error Rate")
        ax.set_ylabel("Logical Error Rate")
        ax.set_title(f"Decodings: {decodings}")
        plt.legend()
        plt.show()
