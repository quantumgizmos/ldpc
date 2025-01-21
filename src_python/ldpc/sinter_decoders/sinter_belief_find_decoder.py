import stim
import numpy as np
import pathlib
from ldpc.belief_find_decoder import BeliefFindDecoder
import sinter
from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices


class SinterBeliefFindDecoder(sinter.Decoder):
    """
    A sinter decoder that combines Belief Propagation (BP) with the Union Find Decoder (UFD) algorithm.

    The BeliefFindDecoder is designed to decode binary linear codes by initially attempting BP decoding, and if that fails,
    it falls back to the Union Find Decoder algorithm. The UFD algorithm is based on the principles outlined in
    https://arxiv.org/abs/1709.06218, with an option to utilise a more general version as described in
    https://arxiv.org/abs/2103.08049 for LDPC codes by setting `uf_method=True`.

    Parameters
    ----------
    pcm : Union[np.ndarray, scipy.sparse.spmatrix]
        The parity check matrix for the code.
    error_rate : Optional[float], optional
        The probability of a bit being flipped in the received codeword, by default None.
    error_channel : Optional[List[float]], optional
        A list of probabilities specifying the probability of each bit being flipped in the received codeword.
        Must be of length equal to the block length of the code, by default None.
    max_iter : Optional[int], optional
        The maximum number of iterations for the decoding algorithm, by default 0.
    bp_method : Optional[str], optional
        The belief propagation method used. Must be one of {'product_sum', 'minimum_sum'}, by default 'minimum_sum'.
    ms_scaling_factor : Optional[float], optional
        The scaling factor used in the minimum sum method, by default 1.0.
    schedule : Optional[str], optional
        The scheduling method used. Must be one of {'parallel', 'serial'}, by default 'parallel'.
    omp_thread_count : Optional[int], optional
        The number of OpenMP threads used for parallel decoding, by default 1.
    random_schedule_seed : Optional[int], optional
        Whether to use a random serial schedule order, by default 0.
    serial_schedule_order : Optional[List[int]], optional
        A list of integers specifying the serial schedule order. Must be of length equal to the block length of the code,
        by default None.
    uf_method : str, optional
        The method used to solve the local decoding problem in each cluster. Choose from: 1) 'inversion' or 2) 'peeling'.
        By default set to 'inversion'. The 'peeling' method is only suitable for LDPC codes with point like syndromes.
        The inversion method can be applied to any parity check matrix.
    bits_per_step : int, optional
        Specifies the number of bits added to the cluster in each step of the UFD algorithm. If no value is provided, this is set the block length of the code.
    uf_method : str, optional
        The method used to solve the local decoding problem in each cluster. Choose from: 1) 'inversion' or 2) 'peeling'.
        By default set to 'inversion'. The 'peeling' method is only suitable for LDPC codes with point like syndromes.
        The inversion method can be applied to any parity check matrix.
    bits_per_step : int, optional
        Specifies the number of bits added to the cluster in each step of the UFD algorithm. If no value is provided, this is set the block length of the code.

    Notes
    -----
    The `BeliefFindDecoder` class leverages soft information outputted by the BP decoder to guide the cluster growth
    in the UFD algorithm. The number of bits added to the cluster in each step is controlled by the `bits_per_step` parameter.
    The `uf_method` parameter activates a more general version of the UFD algorithm suitable for LDPC codes when set to True.
    """

    def __init__(
        self,
        max_iter=0,
        bp_method="ms",
        ms_scaling_factor=0.625,
        schedule="parallel",
        omp_thread_count=1,
        serial_schedule_order=None,
        uf_method="inversion",
        bits_per_step=0,
    ):
        self.max_iter = max_iter
        self.bp_method = bp_method
        self.ms_scaling_factor = ms_scaling_factor
        self.schedule = schedule
        self.omp_thread_count = omp_thread_count
        self.serial_schedule_order = serial_schedule_order
        self.uf_method = uf_method
        self.bits_per_step = bits_per_step

    def decode_via_files(
        self,
        *,
        num_shots: int,
        num_dets: int,
        num_obs: int,
        dem_path: pathlib.Path,
        dets_b8_in_path: pathlib.Path,
        obs_predictions_b8_out_path: pathlib.Path,
        tmp_dir: pathlib.Path,
    ) -> None:
        """Performs decoding by reading problems from, and writing solutions to, file paths.
        Args:
            num_shots: The number of times the circuit was sampled. The number of problems
                to be solved.
            num_dets: The number of detectors in the circuit. The number of detection event
                bits in each shot.
            num_obs: The number of observables in the circuit. The number of predicted bits
                in each shot.
            dem_path: The file path where the detector error model should be read from,
                e.g. using `stim.DetectorErrorModel.from_file`. The error mechanisms
                specified by the detector error model should be used to configure the
                decoder.
            dets_b8_in_path: The file path that detection event data should be read from.
                Note that the file may be a named pipe instead of a fixed size object.
                The detection events will be in b8 format (see
                https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md ). The
                number of detection events per shot is available via the `num_dets`
                argument or via the detector error model at `dem_path`.
            obs_predictions_b8_out_path: The file path that decoder predictions must be
                written to. The predictions must be written in b8 format (see
                https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md ). The
                number of observables per shot is available via the `num_obs` argument or
                via the detector error model at `dem_path`.
            tmp_dir: Any temporary files generated by the decoder during its operation MUST
                be put into this directory. The reason for this requirement is because
                sinter is allowed to kill the decoding process without warning, without
                giving it time to clean up any temporary objects. All cleanup should be done
                via sinter deleting this directory after killing the decoder.
        """
        self.dem = stim.DetectorErrorModel.from_file(dem_path)
        self.matrices = detector_error_model_to_check_matrices(
            self.dem, allow_undecomposed_hyperedges=True
        )
        self.belief_find = BeliefFindDecoder(
            self.matrices.check_matrix,
            error_channel=list(self.matrices.priors),
            max_iter=self.max_iter,
            bp_method=self.bp_method,
            ms_scaling_factor=self.ms_scaling_factor,
            schedule=self.schedule,
            omp_thread_count=self.omp_thread_count,
            serial_schedule_order=self.serial_schedule_order,
            uf_method=self.uf_method,
            bits_per_step=self.bits_per_step,
        )

        shots = stim.read_shot_data_file(
            path=dets_b8_in_path, format="b8", num_detectors=num_dets
        )
        predictions = np.zeros((num_shots, num_obs), dtype=bool)
        for i in range(num_shots):
            predictions[i, :] = self.decode(shots[i, :])

        stim.write_shot_data_file(
            data=predictions,
            path=obs_predictions_b8_out_path,
            format="b8",
            num_observables=num_obs,
        )

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        corr = self.belief_find.decode(syndrome)
        return (self.matrices.observables_matrix @ corr) % 2
