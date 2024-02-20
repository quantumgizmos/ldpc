import numpy as np
from base_overlapping_window_decoder import BaseOverlappingWindowDecoder
from ldpc import BpOsdDecoder


class BpOsdOverlappingWindowDecoder(BaseOverlappingWindowDecoder):

    def __init__(
        self,
        decodings: int,
        window: int,
        commit: int,
        decoder_args: dict = {},
    ) -> None:
        """A class for decoding stim circuits using the overlapping window approach.

        Parameters
        ----------

        decodings : int
            The number of decodings blocks the circuit is divided into.
        window : int
            The number of rounds in each decoding block.
        commit : int
            The number of rounds the decoding is committed to.
        """

        super().__init__(decodings, window, commit)

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
