import numpy as np
from ldpc.codes import rep_code, ring_code

H = ring_code(30)

from ldpc.bp_decoder import SoftInfoBpDecoder

SoftInfoBpDecoder(H, error_rate=0.1)