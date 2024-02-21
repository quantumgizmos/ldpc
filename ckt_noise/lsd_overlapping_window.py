import numpy as np
import stim
from ldpc import BpLsdDecoder

from base_overlapping_window_decoder import BaseOverlappingWindowDecoder
from ckt_noise.config import DEFAULT_LSD_DECODER_ARGS, DEFAULT_DECODINGS, DEFAULT_WINDOW, DEFAULT_COMMIT


class LsdOverlappingWindowDecoder(BaseOverlappingWindowDecoder):
    def __init__(self,
                 model: stim.DetectorErrorModel,
                 decodings: int = DEFAULT_DECODINGS,
                 window: int = DEFAULT_WINDOW,
                 commit: int = DEFAULT_COMMIT,
                 decoder_args: dict = DEFAULT_LSD_DECODER_ARGS,
                 ):
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

        decoder = BpLsdDecoder(
            round_dcm,
            error_channel=list(weights),
            **self.decoder_args,
        )

        return decoder
