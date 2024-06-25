import pytest
import numpy as np
from ldpc.belief_find_decoder import BeliefFindDecoder
from ldpc.codes import hamming_code, rep_code, ring_code

def test_peeling_input():

    pcm = rep_code(3)
    decoder = BeliefFindDecoder(pcm, error_rate = 0.1, uf_method="peeling")

    pcm = ring_code(3)
    decoder = BeliefFindDecoder(pcm, error_rate = 0.1, uf_method="peeling")

    pcm = hamming_code(3)

    with pytest.raises(ValueError):
        decoder = BeliefFindDecoder(pcm, error_rate = 0.1, uf_method="peeling")


if __name__ == "__main__":
    test_peeling_input()




