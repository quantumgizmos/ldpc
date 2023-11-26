from ldpc.codes import hamming_code, rep_code, ring_code
import scipy.sparse
import numpy as np
from ldpc import BpDecoder, bp_decoder, BpOsdDecoder, bposd_decoder

def test_scipy_sparse_matrices():

    # Ring code

    # default scipy input
    bpd = BpDecoder(ring_code(3), error_rate=0.1, input_vector_type='syndrome', bp_method="ms")
    bpd = bp_decoder(ring_code(3), error_rate=0.1, input_vector_type='syndrome', bp_method="ms")

    # numpy input
    bpd = BpDecoder(ring_code(3).toarray(), error_rate=0.1, input_vector_type='syndrome', bp_method="ms")
    bpd = bp_decoder(ring_code(3).toarray(), error_rate=0.1, input_vector_type='syndrome', bp_method="ms")
    
    # scipy sparse csr
    bpd = BpDecoder(scipy.sparse.csr_matrix(ring_code(3)), error_rate=0.1, input_vector_type='syndrome', bp_method="ms")
    bpd = bp_decoder(scipy.sparse.csr_matrix(ring_code(3)), error_rate=0.1, input_vector_type='syndrome', bp_method="ms")

    # scipy sparse csc
    bpd = BpDecoder(scipy.sparse.csc_matrix(ring_code(3)), error_rate=0.1, input_vector_type='syndrome', bp_method="ms")
    bpd = bp_decoder(scipy.sparse.csc_matrix(ring_code(3)), error_rate=0.1, input_vector_type='syndrome', bp_method="ms")

    # hamming code

    # default scipy input
    bpd = BpDecoder(hamming_code(3), error_rate=0.1, input_vector_type='syndrome', bp_method="ms")
    bpd = bp_decoder(hamming_code(3), error_rate=0.1, input_vector_type='syndrome', bp_method="ms")

    # numpy input
    bpd = BpDecoder(hamming_code(3).toarray(), error_rate=0.1, input_vector_type='syndrome', bp_method="ms")
    bpd = bp_decoder(hamming_code(3).toarray(), error_rate=0.1, input_vector_type='syndrome', bp_method="ms")
    
    # scipy sparse csr
    bpd = BpDecoder(scipy.sparse.csr_matrix(hamming_code(3)), error_rate=0.1, input_vector_type='syndrome', bp_method="ms")
    bpd = bp_decoder(scipy.sparse.csr_matrix(hamming_code(3)), error_rate=0.1, input_vector_type='syndrome', bp_method="ms")

    # scipy sparse csc
    bpd = BpDecoder(scipy.sparse.csc_matrix(hamming_code(3)), error_rate=0.1, input_vector_type='syndrome', bp_method="ms")
    bpd = bp_decoder(scipy.sparse.csc_matrix(hamming_code(3)), error_rate=0.1, input_vector_type='syndrome', bp_method="ms")

def test_osd_scipy_sparse_matrices():

    # Ring code

    # default scipy input
    bposd = BpOsdDecoder(ring_code(3), error_rate=0.1, bp_method="ms")
    bposd = bposd_decoder(ring_code(3), error_rate=0.1, bp_method="ms")

    # numpy input
    bposd = BpOsdDecoder(ring_code(3).toarray(), error_rate=0.1, bp_method="ms")
    bposd = bposd_decoder(ring_code(3).toarray(), error_rate=0.1, bp_method="ms")
    
    # scipy sparse csr
    bposd = BpOsdDecoder(scipy.sparse.csr_matrix(ring_code(3)), error_rate=0.1,  bp_method="ms")
    bposd = bposd_decoder(scipy.sparse.csr_matrix(ring_code(3)), error_rate=0.1,  bp_method="ms")

    # scipy sparse csc
    bposd = BpOsdDecoder(scipy.sparse.csc_matrix(ring_code(3)), error_rate=0.1,  bp_method="ms")
    bposd = bposd_decoder(scipy.sparse.csc_matrix(ring_code(3)), error_rate=0.1, bp_method="ms")

    # hamming code

    # default scipy input
    bposd = BpOsdDecoder(hamming_code(3), error_rate=0.1,  bp_method="ms")
    bposd = bposd_decoder(hamming_code(3), error_rate=0.1,  bp_method="ms")

    # numpy input
    bposd = BpOsdDecoder(hamming_code(3).toarray(), error_rate=0.1,  bp_method="ms")
    bposd = bposd_decoder(hamming_code(3).toarray(), error_rate=0.1,  bp_method="ms")
    
    # scipy sparse csr
    bposd = BpOsdDecoder(scipy.sparse.csr_matrix(hamming_code(3)), error_rate=0.1,  bp_method="ms")
    bposd = bposd_decoder(scipy.sparse.csr_matrix(hamming_code(3)), error_rate=0.1,  bp_method="ms")

    # scipy sparse csc
    bposd = BpOsdDecoder(scipy.sparse.csc_matrix(hamming_code(3)), error_rate=0.1, bp_method="ms")
    bposd = bposd_decoder(scipy.sparse.csc_matrix(hamming_code(3)), error_rate=0.1, bp_method="ms")