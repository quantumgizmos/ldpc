import numpy as np
import pytest
from ldpc.belief_find_decoder import BeliefFindDecoder
from ldpc.codes import hamming_code, rep_code, ring_code


def test_peeling_input():
    pcm = rep_code(3)
    BeliefFindDecoder(pcm, error_rate=0.1, uf_method="peeling")

    pcm = ring_code(3)
    BeliefFindDecoder(pcm, error_rate=0.1, uf_method="peeling")

    pcm = hamming_code(3)

    with pytest.raises(ValueError):
        BeliefFindDecoder(pcm, error_rate=0.1, uf_method="peeling")


def test_zero_syndrome():
    pcm = rep_code(5)
    zero_syndrome = np.zeros(pcm.shape[0], dtype=np.uint8)

    for uf_method in ["inversion", "peeling"]:
        bf = BeliefFindDecoder(pcm, error_rate=0.1, uf_method=uf_method)
        bf.decode(np.array([1, 1, 0, 1]))
        decoding = bf.decode(zero_syndrome)
        assert np.array_equal(decoding, np.zeros(pcm.shape[1]))
        assert decoding.dtype == zero_syndrome.dtype
        assert bf.converge


if __name__ == "__main__":
    test_peeling_input()
