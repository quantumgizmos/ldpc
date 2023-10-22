import numpy as np
from scipy.sparse import spmatrix
class BeliefFindDecoder(BpDecoderBase):
    def decode(self,syndrome):