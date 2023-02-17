import numpy as np

class gf2sparse:

    def __init__(self, pcm: np.ndarray, empty: bool = False)->None:
        """
        Test docstring

        Parameters
        -----------
        pcm: numpy.ndarray
            Parity check matrix
        empty: bool, Optional
        """

    @property
    def T(self)->gf2sparse:
        """
        The transpose
        """