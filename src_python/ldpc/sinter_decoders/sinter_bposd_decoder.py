import stim
import numpy as np
import pathlib
from ldpc.bposd_decoder import BpOsdDecoder
from sinter import Decoder
from beliefmatching import detector_error_model_to_check_matrices



class SinterBpOsdDecoder(Decoder):

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
    random_serial_schedule : Optional[int], optional
        Whether to use a random serial schedule order, by default False.
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

    def __init__(self,
                 max_iter=0,
                 bp_method="ms",
                 ms_scaling_factor=0.625,
                 schedule="parallel",
                 omp_thread_count=1,
                 random_serial_schedule=False,
                 serial_schedule_order=None,
                 osd_method="osd0",
                 osd_order=0):

        self.max_iter = max_iter
        self.bp_method = bp_method
        self.ms_scaling_factor = ms_scaling_factor
        self.schedule = schedule
        self.omp_thread_count = omp_thread_count
        self.random_serial_schedule = random_serial_schedule
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
        self.dem = stim.DetectorErrorModel.from_file(dem_path)
        self.matrices = detector_error_model_to_check_matrices(self.dem)
        self.bposd = BpOsdDecoder(self.matrices.check_matrix,
                                  error_channel=list(self.matrices.priors),
                                  max_iter=self.max_iter,
                                  bp_method=self.bp_method,
                                  ms_scaling_factor=self.ms_scaling_factor,
                                  schedule=self.schedule,
                                  omp_thread_count=self.omp_thread_count,
                                  random_serial_schedule=self.random_serial_schedule,
                                  serial_schedule_order=self.serial_schedule_order,
                                  osd_method=self.osd_method,
                                  osd_order=self.osd_order)

        shots = stim.read_shot_data_file(
            path=dets_b8_in_path, format="b8", num_detectors=self.dem.num_detectors
        )
        predictions = np.zeros(
            (shots.shape[0], self.dem.num_observables), dtype=bool)
        for i in range(shots.shape[0]):
            predictions[i, :] = self.decode(shots[i, :])

        stim.write_shot_data_file(
            data=predictions,
            path=obs_predictions_b8_out_path,
            format="b8",
            num_observables=self.dem.num_observables,
        )

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        corr = self.bposd.decode(syndrome)
        return (self.matrices.observables_matrix @ corr) % 2
