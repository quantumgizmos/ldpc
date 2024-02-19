import sys

from ldpc.bplsd_decoder._bplsd_decoder import BpLsdDecoder
from ldpc.bposd_decoder._bposd_decoder import BpOsdDecoder

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
        decoder: str = "matching",
        bp_params=None,
        ignore_decomposition_failures: bool = False,
):
    # define relevant parameters
    if bp_params is None and decoder != "matching":
        bp_params = {
            "max_iter": 10,
            "ms_scaling_factor": 0.6,
            "method": 'osd_cs',
            "bp_method": "minimum_sum",
            "schedule": 'parallel',
            "omp_thread_count": 1,
            "random_schedule_seed": 0,
            "serial_schedule_order": None,
            "bits_per_step": 1,
            "order": 0,
        }
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
    dem = circuit.detector_error_model(decompose_errors=decompose_errors,
                                       ignore_decomposition_failures=ignore_decomposition_failures)
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
    # set eps to mininum res of float32
    eps = sys.float_info.min
    min_weight = np.log1p(eps) - np.log(eps)

    dcm = dem_matrices.check_matrix

    # dense version of the correction
    total_corr = np.zeros((shots, dcm.shape[1]), dtype=np.uint8)

    for decoding in range(decodings):
        # for obs, sample in zip(obs_data, detector_data):
        commit_inds, dec_inds, synd_commit_inds, synd_dec_inds = current_round_inds(
            dcm, decoding, window, commit, num_checks
        )

        round_dcm = dcm[synd_dec_inds, :]
        decoder = get_decoder(bp_params, round_dcm, weights, decoder)

        for i in range(shots):
            corr = decoder.decode(detector_data[i][synd_dec_inds])

            if decoding != decodings - 1:
                # determine the partial correction / commit the correction
                total_corr[i][commit_inds] += corr[commit_inds]
                # modify syndrome to reflect the correction
                detector_data[i][synd_dec_inds] ^= round_dcm @ total_corr[i] % 2

            else:
                # This is the final decoding, commit all
                total_corr[i][dec_inds] += corr[dec_inds]

        # once all shots have been decoded for this round, update the weights
        weights[commit_inds] = min_weight

    for shot in range(shots):
        if not np.array_equal(
                (dem_matrices.observables_matrix @ total_corr[shot]) % 2, obs_data[shot]
        ):
            total_errs += 1

    return total_errs / shots


def get_decoder(bp_params, round_dcm, weights, decoder: str = 'matching'):
    if decoder == "matching":
        decoder = Matching.from_check_matrix(
            round_dcm,
            weights=weights,
        )
    elif decoder == "bposd":
        decoder = BpOsdDecoder(
            pcm=round_dcm,
            channel_probs=weights,
            max_iter=bp_params["max_iter"],
            ms_scaling_factor=bp_params["ms_scaling_factor"],
            osd_method=bp_params["method"],
            bp_method=bp_params["bp_method"],
            schedule=bp_params["schedule"],
            omp_thread_count=bp_params["omp_thread_count"],
            random_schedule_seed=bp_params["random_schedule_seed"],
            serial_schedule_order=bp_params["serial_schedule_order"],
            osd_order=bp_params["order"],
        )
    elif decoder == "bplsd":
        decoder = BpLsdDecoder(
            pcm=round_dcm,
            error_channel=weights,
            max_iter=bp_params["max_iter"],
            ms_scaling_factor=bp_params["ms_scaling_factor"],
            lsd_method=bp_params["method"],
            bp_method=bp_params["bp_method"],
            schedule=bp_params["schedule"],
            omp_thread_count=bp_params["omp_thread_count"],
            random_schedule_seed=bp_params["random_schedule_seed"],
            serial_schedule_order=bp_params["serial_schedule_order"],
            bits_per_step=bp_params["bits_per_step"],
            lsd_order=bp_params["order"],
        )
    return decoder

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

    min_index = dcm[slice(start, end_commit), :].nonzero()[1].min()
    max_index_commit = dcm[slice(start, end_commit), :].nonzero()[1].max()
    max_index_decoding = dcm[slice(start, end_decoding), :].nonzero()[1].max()

    # use slices instead of np.arange
    commit_inds = slice(min_index, max_index_commit)
    decoding_inds = slice(min_index, max_index_decoding)

    # use slices instead of np.arange
    synd_commit_inds = slice(start, end_commit)
    synd_decoding_inds = slice(start, end_decoding)

    return commit_inds, decoding_inds, synd_commit_inds, synd_decoding_inds


if __name__ == "__main__":

    for decodings in [1, 2, 3, 4, 5]:
        fig, ax = plt.subplots()
        ps = np.geomspace(0.02, 0.08, 6)
        for d in [5, 9, 13]:
            pcm, logicals = (rep_code(d))
            # errs = overlapping_window(0.04, pcm, logicals, 1, 2 * d, 2 * d)
            error_rates = [
                overlapping_window(p, pcm, logicals, decodings, 2 * d, d, decoder='bposd') for p in ps
            ]
            ax.plot(ps, error_rates, label=f"d={d}", marker="o")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Physical Error Rate")
        ax.set_ylabel("Logical Error Rate")
        ax.set_title(f"Decodings: {decodings}")
        plt.legend()
        plt.show()
