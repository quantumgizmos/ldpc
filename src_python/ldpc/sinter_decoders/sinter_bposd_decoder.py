import stim
import numpy as np
import pathlib
from BeliefMatching import detector_error_model_to_check_matrices
from ldpc.sinter_decoders import SinterDecoder
from ldpc.bposd_decoder import BpOsdDecoder


class SinterBpOsdDecoder(SinterDecoder):
    def __init__(self, 
                 bp_method="ms", 
                 ms_scaling_factor=0.625, 
                 schedule="parallel", 
                 osd_method="osd0"):
        self.bp_method = bp_method
        self.ms_scaling_factor = ms_scaling_factor
        self.schedule = schedule
        self.osd_method = osd_method

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
        self.bposd = BpOsdDecoder(self.matrices.check_matrix, error_channel=self.matrices.priors, max_iter=5, bp_method="ms", ms_scaling_factor = 0.625, schedule="parallel", osd_method = "osd0")

        shots = stim.read_shot_data_file(
            path=dets_b8_in_path, format="b8", num_detectors=self.dem.num_detectors
        )
        predictions = np.zeros((shots.shape[0], self.dem.num_observables), dtype=bool)
        for i in range(shots.shape[0]):
            predictions[i, :] = self.decode(shots[i, :])
 
        stim.write_shot_data_file(
            data=predictions,
            path=obs_predictions_b8_out_path,
            format="b8",
            num_observables=self.dem.num_observables,
        )


    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        corr = self.matrices.decode(syndrome)
        if self._bpd.converge:
            return (self.matrices.observables_matrix @ corr) % 2
