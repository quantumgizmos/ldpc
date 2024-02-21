import numpy as np
import stim
from ldpc.ckt_noise.base_overlapping_window_decoder import BaseOverlappingWindowDecoder
from ldpc import BpOsdDecoder
from ldpc.ckt_noise.config import DEFAULT_BPOSD_DECODER_ARGS, DEFAULT_DECODINGS, DEFAULT_WINDOW, DEFAULT_COMMIT


class BpOsdOverlappingWindowDecoder(BaseOverlappingWindowDecoder):
    def __init__(self,
                 model: stim.DetectorErrorModel,
                 **kwargs):
        decodings = kwargs.get('decodings', DEFAULT_DECODINGS)
        window = kwargs.get('window', DEFAULT_WINDOW)
        commit = kwargs.get('commit', DEFAULT_COMMIT)
        decoder_args = kwargs.get('decoder_args', DEFAULT_BPOSD_DECODER_ARGS)
        super().__init__(
            model=model,
            decodings=decodings,
            window=window,
            commit=commit,
            decoder_args=decoder_args,
        )

    def _get_dcm(self):
        """
        Set the detector check matrix for the decoder.

        """
        return self.dem_matrices.check_matrix

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
        return list(self.dem_matrices.priors)

    def _init_decoder(self, round_dcm: np.ndarray, weights: np.ndarray):
        """
        Initialize the decoder for a given round.
        """

        decoder = BpOsdDecoder(
            round_dcm,
            error_channel=list(weights),
            **self.decoder_args,
        )

        return decoder
