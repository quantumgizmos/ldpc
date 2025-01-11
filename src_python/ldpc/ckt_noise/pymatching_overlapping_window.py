import sys
import numpy as np
from ldpc.ckt_noise.base_overlapping_window_decoder import BaseOverlappingWindowDecoder
from pymatching import Matching


class PyMatchingOverlappingWindowDecoder(BaseOverlappingWindowDecoder):
    def __init__(
        self,
        model,
        **decoder_kwargs,
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
        self.decoder_args = decoder_kwargs
        super().__init__(
            model=model,
            **decoder_kwargs,
        )

    def _get_dcm(self):
        """
        Set the detector check matrix for the decoder.

        Note
        ----

        Matching-based decoders require the edge check matrix
        as input as the ordinary `check_matrix` might contain hyperedges.

        """
        return self.dem_matrices.edge_check_matrix

    def _get_logical_observables_matrix(self):
        """
        Set the logical observables matrix for the decoder.
        """

        return self.dem_matrices.edge_observables_matrix

    @property
    def _min_weight(self):
        """
        Return the minimum weight of the error channel for the decoder.
        """

        min_float = sys.float_info.min
        return np.clip(-np.log(min_float), -16777215, +16777215)

    def _get_weights(self):
        """
        Return the weights for the error channel of the decoder obtained from the detector error model.
        """
        probs = self.dem_matrices.hyperedge_to_edge_matrix @ self.dem_matrices.priors
        probs[probs == 0] = 1e-308
        return np.clip(np.log1p(probs) - np.log(probs), -16777215, +16777215)

    def _init_decoder(self, round_dcm: np.ndarray, weights: np.ndarray):
        """
        Initialize the decoder for a given round.
        """
        decoder = Matching.from_check_matrix(round_dcm, weights, **self.decoder_args)

        return decoder
