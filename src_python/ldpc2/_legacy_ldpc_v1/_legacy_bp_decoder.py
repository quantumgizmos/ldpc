import scipy.sparse
import numpy as np
from ldpc2.bp_decoder import BpDecoder
from scipy.sparse import spmatrix
import warnings

class bp_decoder(BpDecoder):
    '''
    Legacy ldpc_v1 function
    ----------

    A class implementing a belief propagation decoder for LDPC codes

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
    input_vector_type: str, optional
        Use this paramter to specify the input type. Choose either: 1) 'syndrome' or 2) 'received_vector'.
        Note, it is only necessary to specify this value when the parity check matrix is square. When the
        parity matrix is non-square the input vector type is inferred automatically from its length.
    '''

    def __init__(self,parity_check_matrix,**kwargs):
        warnings.warn("This is the old syntax for the `bp_decoder` from `ldpc v1`. Use the `BpDecoder` class from `ldpc v2` for additional features.")

        #Load in optional parameters (and set defaults)
        error_rate=kwargs.get("error_rate",None)
        max_iter=kwargs.get("max_iter",0)
        bp_method=kwargs.get("bp_method",0)
        ms_scaling_factor=kwargs.get("ms_scaling_factor",1.0)
        channel_probs=kwargs.get("channel_probs",None)
        input_vector_type=kwargs.get("input_vector_type",-1)

        #BP method
        if str(bp_method).lower() in ['prod_sum','product_sum','ps','0','prod sum']:
            bp_method="ps"
        elif str(bp_method).lower() in ['min_sum','minimum_sum','ms','1','minimum sum','min sum']:
            bp_method="ms" # method 1 is not working (see issue 1). Defaulting to the log version of bp.
        else: raise ValueError(f"BP method '{bp_method}' is invalid.\
                            Please choose from the following methods:'product_sum',\
                            'minimum_sum'")
        
        BpDecoder(
            parity_check_matrix,
            error_rate=error_rate,
            error_channel=channel_probs,
            max_iter=max_iter,
            bp_method=bp_method,
            ms_scaling_factor=ms_scaling_factor
            )
    
    @property
    def channel_probs(self):
        return self.error_channel
    
    def update_channel_probs(self,channel):
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
        





