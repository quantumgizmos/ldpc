import stim
import numpy as np
import pathlib
from ldpc.bposd_decoder import BpOsdDecoder
import sinter
from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices


class SinterBpOsdDecoder(sinter.Decoder):
    """
    Initialize the SinterBPOSDDecoder object.

    Parameters
    ----------
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
    serial_schedule_order : Optional[List[int]], optional
        A list of integers that specify the serial schedule order. Must be of length equal to the block length of the code, by default None.
    osd_method : int, optional
        The OSD method used. Must be one of {'OSD_0', 'OSD_E', 'OSD_CS'}, by default 'OSD_0'.
    osd_order : int, optional
        The OSD order, by default 0.

    Notes
    -----
    This class provides an interface for configuring and using the SinterBPOSDDecoder for quantum error correction.
    """

    def __init__(
        self,
        max_iter=0,
        bp_method="ms",
        ms_scaling_factor=0.625,
        schedule="parallel",
        omp_thread_count=1,
        serial_schedule_order=None,
        osd_method="osd0",
        osd_order=0,
    ):
        self.max_iter = max_iter
        self.bp_method = bp_method
        self.ms_scaling_factor = ms_scaling_factor
        self.schedule = schedule
        self.omp_thread_count = omp_thread_count
        self.serial_schedule_order = serial_schedule_order
        self.osd_method = osd_method
        self.osd_order = osd_order

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
        self.bposd = BpOsdDecoder(
            self.matrices.check_matrix,
            error_channel=list(self.matrices.priors),
            max_iter=self.max_iter,
            bp_method=self.bp_method,
            ms_scaling_factor=self.ms_scaling_factor,
            schedule=self.schedule,
            omp_thread_count=self.omp_thread_count,
            serial_schedule_order=self.serial_schedule_order,
            osd_method=self.osd_method,
            osd_order=self.osd_order,
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
        corr = self.bposd.decode(syndrome)
        return (self.matrices.observables_matrix @ corr) % 2
