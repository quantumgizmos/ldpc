import numpy as np
import stim
from ldpc.bplsd_decoder import BpLsdDecoder

from ldpc.ckt_noise.base_overlapping_window_decoder import BaseOverlappingWindowDecoder
from ldpc.ckt_noise.config import DEFAULT_LSD_DECODER_ARGS, DEFAULT_DECODINGS, DEFAULT_WINDOW, DEFAULT_COMMIT


class LsdOverlappingWindowDecoder(BaseOverlappingWindowDecoder):
    def __init__(self,
                 model: stim.DetectorErrorModel,
                 **decoder_kwargs):
        self.decoder_args = decoder_kwargs
        super().__init__(
            model=model,
            **decoder_kwargs
        )

    def _get_dcm(self):

        """
        Set the detector check matrix for the decoder.

        """
        return self.dem_matrices.check_matrix

    def _get_logical_observables_matrix(self):
        """
        Set the logical observables matrix for the decoder.

        """
        return self.dem_matrices.observables_matrix
    @property
    def _min_weight(self):
        """
        Return the minimum weight of the error channel for the decoder.
        """
        return 0.0

    def _get_weights(self):
        """
        Return the weights for the error channel of the decoder obtained from the detector error model.
        """
        return self.dem_matrices.priors

    def _init_decoder(self, round_dcm: np.ndarray, weights: np.ndarray):
        """
        Initialize the decoder for a given round.
        """
        args = self.decoder_args["lsd_args"] | DEFAULT_LSD_DECODER_ARGS
        decoder = BpLsdDecoder(
            round_dcm,
            error_channel=list(weights),
            lsd_method=args["lsd_method"],
            lsd_order=args["lsd_order"],
            max_iter = args["max_iter"],
            bp_method=args["bp_method"]
        )

        return decoder
