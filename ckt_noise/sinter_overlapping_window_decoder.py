from sinter import Decoder

class SinterOverlappingWindowDecoder(Decoder):

    def __init__(self,
                 decodings: int,
                    window: int,
                    commit: int,
                 ) -> None:
    """
    A class for decoding stim circuits using the overlapping window approach.
    
    Parameters
    ----------

    decodings : int
        The number of decodings blocks the circuit is divided into.
    window : int
        The number of rounds in each decoding block.    
    commit : int
        The number of rounds the decoding is committed to.
    """    
    
        