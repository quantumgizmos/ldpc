import stim
from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices
from scipy.sparse import csr_matrix
import numpy as np


class BaseOverlappingWindowDecoder:
    def __init__(
        self,
        model: stim.DetectorErrorModel,
        decodings: int,
        window: int,
        commit: int,
        num_checks: int,
        **decoder_kwargs,
    ) -> None:
        """A base class for implementing decoders that work on stim circuits using the overlapping window approach.

        Parameters
        ----------

        decodings : int
            The number of decodings blocks the circuit is divided into.
        window : int
            The number of rounds in each decoding block.
        commit : int
            The number of rounds the decoding is committed to.
        """

        self.decodings = decodings
        self.window = window
        self.commit = commit
        self.num_checks = num_checks

        self.dem_matrices = detector_error_model_to_check_matrices(
            model, allow_undecomposed_hyperedges=True
        )
        self.num_detectors = model.num_detectors

        # assert that the number of detectors is a integer multiple of the number of rounds
        rounds = (self.window - self.commit) + self.decodings * self.commit
        if not self.num_detectors % rounds == 0:
            raise ValueError(
                f"The number of detectors must be a multiple of the number of rounds. There are {self.num_detectors} detectors and "
                f"{rounds} rounds."
                "Dem matrices must be decomposed into a number of rounds that is a multiple of the number of detectors."
                f"You expected {self.num_checks * rounds}"
            )

        self.dcm = self._get_dcm()
        self.logical_observables_matrix = self._get_logical_observables_matrix()

    def _get_dcm(self) -> csr_matrix:
        """
        Get the detector check matrix.
        """

        raise NotImplementedError("This method must be implemented by the subclass.")

    def _get_logical_observables_matrix(self):
        """
        Set the logical observables matrix.
        """

        raise NotImplementedError("This method must be implemented by the subclass.")

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode a syndrome using the overlapping window approach.


        Parameters
        ----------
        syndrome : np.ndarray
            A single shot of syndrome data. This should be a binary array with a length equal to the
            number of detectors in the `stim.Circuit` or `stim.DetectorErrorModel`. E.g. the syndrome might be
            one row of shot data sampled from a `stim.CompiledDetectorSampler`.

        Returns
        -------
        np.ndarray
            A binary numpy array `predictions` which predicts which observables were flipped.
            Its length is equal to the number of observables in the `stim.Circuit` or `stim.DetectorErrorModel`.
            `predictions[i]` is 1 if the decoder predicts observable `i` was flipped and 0 otherwise.
        """

        corr = self._corr_multiple_rounds(syndrome)
        return (self.logical_observables_matrix @ corr) % 2

    def _corr_multiple_rounds(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode a syndrome using the overlapping window approach.

        Parameters
        ----------
        syndrome : np.ndarray
            A single shot of syndrome data. This should be a binary array with a length equal to the
            number of detectors in the `stim.Circuit` or `stim.DetectorErrorModel`. E.g. the syndrome might be
            one row of shot data sampled from a `stim.CompiledDetectorSampler`.

        Returns
        -------
        np.ndarray
            A binary numpy array `predictions` which predicts which observables were flipped.
            Its length is equal to the number of observables in the `stim.Circuit` or `stim.DetectorErrorModel`.
            `predictions[i]` is 1 if the decoder predicts observable `i` was flipped and 0 otherwise.
        """
        total_corr = np.zeros(self.dcm.shape[1], dtype=np.uint8)
        weights = self._get_weights()

        for decoding in range(self.decodings):
            commit_inds, dec_inds, synd_commit_inds, synd_dec_inds = current_round_inds(
                dcm=self.dcm,
                decoding=decoding,
                window=self.window,
                commit=self.commit,
                num_checks=self.num_checks,
            )

            round_dcm = self.dcm[synd_dec_inds, :]

            decoder = self._get_decoder(decoding, round_dcm, weights)

            corr = decoder.decode(syndrome[synd_dec_inds])

            if decoding != self.decodings - 1:
                # determine the partial correction / commit the correction
                total_corr[commit_inds] += corr[commit_inds]
                # modify syndrome to reflect the correction
                syndrome[synd_dec_inds] ^= round_dcm @ total_corr % 2

            else:
                # This is the final decoding, commit all
                total_corr[dec_inds] += corr[dec_inds]

            # once all shots have been decoded for this round, update the weights
            weights[commit_inds] = self._min_weight

        return total_corr

    def decode_batch(
        self,
        shots: np.ndarray,
        *,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        shots : np.ndarray
            A binary numpy array of dtype `np.uint8` or `bool` with shape `(num_shots, num_detectors)`, where
            here `num_shots` is the number of shots and `num_detectors` is the number of detectors in the `stim.Circuit` or `stim.DetectorErrorModel`.

        Returns
        -------
        np.ndarray
            A 2D numpy array `predictions` of dtype bool, where `predictions[i, :]` is the output of
            `self.decode(shots[i, :])`.
        """

        if bit_packed_shots:
            shots = np.unpackbits(shots, axis=1, bitorder="little")[
                :, : self.num_detectors
            ]

        corrs = self._corr_multiple_rounds_batch(shots)

        predictions = np.zeros(
            (shots.shape[0], self.logical_observables_matrix.shape[0]), dtype=bool
        )
        for i in range(shots.shape[0]):
            predictions[i, :] = (self.logical_observables_matrix @ corrs[i]) % 2

        if bit_packed_predictions:
            predictions = np.packbits(predictions, axis=1, bitorder="little")

        return predictions

    def _corr_multiple_rounds_batch(self, shots: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        shots : np.ndarray
            A binary numpy array of dtype `np.uint8` or `bool` with shape `(num_shots, num_detectors)`, where
            here `num_shots` is the number of shots and `num_detectors` is the number of detectors in the `stim.Circuit` or `stim.DetectorErrorModel`.

        Returns
        -------
        np.ndarray
            A 2D numpy array `corrs` of dtype `np.uint8`, where `corrs[i, :]` is the output of
            `self._corr_multiple_rounds(shots[i, :])`.
        """

        total_corr = np.zeros((shots.shape[0], self.dcm.shape[1]), dtype=np.uint8)
        weights = self._get_weights()
        num_shots = shots.shape[0]

        for decoding in range(self.decodings):
            commit_inds, dec_inds, synd_commit_inds, synd_dec_inds = current_round_inds(
                dcm=self.dcm,
                decoding=decoding,
                window=self.window,
                commit=self.commit,
                num_checks=self.num_checks,
            )

            round_dcm = self.dcm[synd_dec_inds, :]

            decoder = self._get_decoder(decoding, round_dcm, weights)

            for i in range(num_shots):
                corr = decoder.decode(shots[i][synd_dec_inds])
                if decoding != self.decodings - 1:
                    # determine the partial correction / commit the correction
                    total_corr[i][commit_inds] += corr[commit_inds]
                    # modify syndrome to reflect the correction
                    shots[i][synd_dec_inds] ^= round_dcm @ total_corr[i] % 2

                else:
                    # This is the final decoding, commit all
                    total_corr[i][dec_inds] += corr[dec_inds]

            # once all shots have been decoded for this round, update the weights
            weights[commit_inds] = self._min_weight

        return total_corr

    def _get_weights(self) -> np.ndarray:
        """
        Obtain the decoder weights from the priors.
        """

        # This should be np.log1p(dem_matrices.priors) - np.log(dem_matrices.priors) for matching decoders
        # This should be dem_matrices.priors for BP type decoders.

        raise NotImplementedError("This method must be implemented by the subclass.")

    @property
    def _min_weight(self) -> float:
        """
        The minimum weight for the decoder.
        """

        # This should be 0. for BP type decoders.
        # This should be np.log1p(min_float) - np.log(min_float) for matching type decoders.

        raise NotImplementedError("This method must be implemented by the subclass.")

    def _get_decoder(self, decoding: int, round_dcm: np.ndarray, weights: np.ndarray):
        """
        Returns the decoder for a given round.

        Parameters
        ----------
        decoding : int
            The decoding round.

        round_dcm : np.ndarray
            The detector check matrix for the current round.

        weights : np.ndarray
            The weights for the error channel of the decoder.

        Returns
        -------
        Decoder
            The decoder for the current round.

        """

        # If no decoder has been initialized for `decoding`, initialize it
        if not hasattr(self, "_decoders"):
            self._decoders = {}

        if decoding not in self._decoders:
            self._decoders[decoding] = self._init_decoder(round_dcm, weights)

        return self._decoders[decoding]

    def _init_decoder(self, round_dcm: np.ndarray, weights: np.ndarray):
        """
        Initialize the decoder for a given round.
        """

        raise NotImplementedError("This method must be implemented by the subclass.")


def current_round_inds(
    dcm: csr_matrix,
    decoding: int,
    window: int,
    commit: int,
    num_checks: int,
) -> tuple:
    """
    Get the indices of the current round in the detector syndrome.

    Parameters
    ----------
    dcm : csr_matrix
        The detector check matrix.

    decoding : int
        The current decoding round.

    window : int
        The number of rounds in each decoding block.

    commit : int
        The number of rounds the decoding is committed to.

    num_checks : int
        The number of checks CSS code check matrix.

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
    commit_inds = slice(min_index, max_index_commit + 1)
    decoding_inds = slice(min_index, max_index_decoding + 1)

    # use slices instead of np.arange
    synd_commit_inds = slice(start, end_commit)
    synd_decoding_inds = slice(start, end_decoding)

    return commit_inds, decoding_inds, synd_commit_inds, synd_decoding_inds
