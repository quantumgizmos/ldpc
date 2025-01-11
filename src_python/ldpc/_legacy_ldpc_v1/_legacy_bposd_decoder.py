from ldpc.bposd_decoder import BpOsdDecoder
import numpy as np
import warnings


class bposd_decoder(BpOsdDecoder):
    """
    A class implementing a belief propagation plus ordered statistics decoding for LDPC codes

    Parameters
    ----------
    parity_check_matrix: numpy.ndarray or spipy.sparse
        The parity check matrix of the binary code in numpy.ndarray or spipy.sparse format.
    error_rate: float64, optional
        The bit error rate.
    max_iter: int, optional
        The maximum number of iterations for the BP decoder. If max_iter==0, the BP algorithm
        will iterate n times, where n is the block length of the code.
    bp_method: str or int, optional
        The BP method. Currently three methods are implemented: 1) "ps": product sum updates;
        2) "ms": min-sum updates; 3) "msl": min-sum log updates
    ms_scaling_factor: float64, optional
        Sets the min-sum scaling factor for the min-sum BP method
    channel_probs: list, optional
        This parameter can be used to set the initial error channel across all bits.
    osd_order: str or int, optional
        Sets the OSD order.
    osd_method: str or int, optional
        The OSD method. Currently three methods are availbe: 1) "osd_0": Zero-oder OSD; 2) "osd_e": exhaustive OSD;
        3) "osd_cs": combination-sweep OSD.

    """

    def __init__(
        self,
        parity_check_matrix,
        error_rate=None,
        max_iter=0,
        bp_method="ps",
        ms_scaling_factor=1.0,
        channel_probs=[None],
        osd_method="osd_0",
        osd_order=0,
    ):
        warnings.warn(
            "This is the old syntax for the `bposd_decoder` from `ldpc v1`. Use the `BpOsdDecoder` class from `ldpc v2` for additional features."
        )

        # BP method
        if str(bp_method).lower() in ["prod_sum", "product_sum", "ps", "0", "prod sum"]:
            bp_method = "ps"
        elif str(bp_method).lower() in [
            "min_sum",
            "minimum_sum",
            "ms",
            "1",
            "minimum sum",
            "min sum",
        ]:
            bp_method = "ms"  # method 1 is not working (see issue 1). Defaulting to the log version of bp.
        else:
            raise ValueError(f"BP method '{bp_method}' is invalid.\
                            Please choose from the following methods:'product_sum',\
                            'minimum_sum'")

        # OSD method
        if str(osd_method).lower() in ["OSD_0", "osd_0", "0", "osd0"]:
            osd_method = "osd_0"
            osd_order = 0
        elif str(osd_method).lower() in ["osd_e", "1", "osde", "exhaustive", "e"]:
            osd_method = "osd_e"
            if osd_order > 15:
                print(
                    "WARNING: Running the 'OSD_E' (Exhaustive method) with search depth greater than 15 is not recommended. Use the 'osd_cs' method instead."
                )
        elif str(osd_method).lower() in [
            "osd_cs",
            "2",
            "osdcs",
            "combination_sweep",
            "combination_sweep",
            "cs",
        ]:
            osd_method = "osd_cs"
        else:
            raise ValueError(
                f"ERROR: OSD method '{osd_method}' invalid. Please choose from the following methods: 'OSD_0', 'OSD_E' or 'OSD_CS'."
            )

        # error channel setup
        error_channel = np.zeros(parity_check_matrix.shape[1]).astype(float)
        if channel_probs[0] is not None:
            for j in range(parity_check_matrix.shape[1]):
                error_channel[j] = channel_probs[j]
            # self.error_rate=np.mean(channel_probs)
        if channel_probs[0] is not None:
            if len(channel_probs) != parity_check_matrix.shape[1]:
                raise ValueError(
                    f"The length of the channel probability vector must be eqaul to the block length n={parity_check_matrix.shape[1]}."
                )
        elif error_rate != 0:
            pass
        else:
            raise ValueError(
                "Either the error_rate or channel_probs must be specified."
            )

        if channel_probs[0] is not None:
            for j in range(parity_check_matrix.shape[1]):
                error_channel[j] = channel_probs[j]
            # self.error_rate=np.mean(channel_probs)
        else:
            error_channel = None

        self.error_rate = error_rate
        self.max_iter = int(max_iter)
        self.ms_scaling_factor = float(ms_scaling_factor)
        self.bp_method = bp_method
        self.osd_method = osd_method
        self.osd_order = osd_order
        self.error_channel = error_channel

    @property
    def channel_probs(self):
        return self.error_channel

    def update_channel_probs(self, channel):
        """
        Function updates the channel probabilities for each bit in the BP decoder.

        Parameters
        ----------
        channel: numpy.ndarray
            A list of the channel probabilities for each bit

        Returns
        -------
        NoneType
        """
        self.error_channel = channel
