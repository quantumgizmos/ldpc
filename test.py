from ldpc.bp_decoder import BpDecoder
# from udlr.gf2sparse import PluDecomposition
import ldpc

print(ldpc.__file__)

from ldpc.codes import hamming_code

print(hamming_code(3))