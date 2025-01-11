from ldpc.codes import hamming_code, ring_code
import scipy.sparse
from ldpc import BpDecoder, bp_decoder, BpOsdDecoder, bposd_decoder


def test_scipy_sparse_matrices():
    # Ring code

    # default scipy input
    BpDecoder(
        ring_code(3), error_rate=0.1, input_vector_type="syndrome", bp_method="ms"
    )
    bp_decoder(
        ring_code(3), error_rate=0.1, input_vector_type="syndrome", bp_method="ms"
    )

    # numpy input
    BpDecoder(
        ring_code(3).toarray(),
        error_rate=0.1,
        input_vector_type="syndrome",
        bp_method="ms",
    )
    bp_decoder(
        ring_code(3).toarray(),
        error_rate=0.1,
        input_vector_type="syndrome",
        bp_method="ms",
    )

    # scipy sparse csr
    BpDecoder(
        scipy.sparse.csr_matrix(ring_code(3)),
        error_rate=0.1,
        input_vector_type="syndrome",
        bp_method="ms",
    )
    bp_decoder(
        scipy.sparse.csr_matrix(ring_code(3)),
        error_rate=0.1,
        input_vector_type="syndrome",
        bp_method="ms",
    )

    # scipy sparse csc
    BpDecoder(
        scipy.sparse.csc_matrix(ring_code(3)),
        error_rate=0.1,
        input_vector_type="syndrome",
        bp_method="ms",
    )
    bp_decoder(
        scipy.sparse.csc_matrix(ring_code(3)),
        error_rate=0.1,
        input_vector_type="syndrome",
        bp_method="ms",
    )

    # hamming code

    # default scipy input
    BpDecoder(
        hamming_code(3), error_rate=0.1, input_vector_type="syndrome", bp_method="ms"
    )
    bp_decoder(
        hamming_code(3), error_rate=0.1, input_vector_type="syndrome", bp_method="ms"
    )

    # numpy input
    BpDecoder(
        hamming_code(3).toarray(),
        error_rate=0.1,
        input_vector_type="syndrome",
        bp_method="ms",
    )
    bp_decoder(
        hamming_code(3).toarray(),
        error_rate=0.1,
        input_vector_type="syndrome",
        bp_method="ms",
    )

    # scipy sparse csr
    BpDecoder(
        scipy.sparse.csr_matrix(hamming_code(3)),
        error_rate=0.1,
        input_vector_type="syndrome",
        bp_method="ms",
    )
    bp_decoder(
        scipy.sparse.csr_matrix(hamming_code(3)),
        error_rate=0.1,
        input_vector_type="syndrome",
        bp_method="ms",
    )

    # scipy sparse csc
    BpDecoder(
        scipy.sparse.csc_matrix(hamming_code(3)),
        error_rate=0.1,
        input_vector_type="syndrome",
        bp_method="ms",
    )
    bp_decoder(
        scipy.sparse.csc_matrix(hamming_code(3)),
        error_rate=0.1,
        input_vector_type="syndrome",
        bp_method="ms",
    )


def test_osd_scipy_sparse_matrices():
    # Ring code

    # default scipy input
    BpOsdDecoder(ring_code(3), error_rate=0.1, bp_method="ms")
    bposd_decoder(ring_code(3), error_rate=0.1, bp_method="ms")

    # numpy input
    BpOsdDecoder(ring_code(3).toarray(), error_rate=0.1, bp_method="ms")
    bposd_decoder(ring_code(3).toarray(), error_rate=0.1, bp_method="ms")

    # scipy sparse csr
    BpOsdDecoder(scipy.sparse.csr_matrix(ring_code(3)), error_rate=0.1, bp_method="ms")
    bposd_decoder(scipy.sparse.csr_matrix(ring_code(3)), error_rate=0.1, bp_method="ms")

    # scipy sparse csc
    BpOsdDecoder(scipy.sparse.csc_matrix(ring_code(3)), error_rate=0.1, bp_method="ms")
    bposd_decoder(scipy.sparse.csc_matrix(ring_code(3)), error_rate=0.1, bp_method="ms")

    # hamming code

    # default scipy input
    BpOsdDecoder(hamming_code(3), error_rate=0.1, bp_method="ms")
    bposd_decoder(hamming_code(3), error_rate=0.1, bp_method="ms")

    # numpy input
    BpOsdDecoder(hamming_code(3).toarray(), error_rate=0.1, bp_method="ms")
    bposd_decoder(hamming_code(3).toarray(), error_rate=0.1, bp_method="ms")

    # scipy sparse csr
    BpOsdDecoder(
        scipy.sparse.csr_matrix(hamming_code(3)), error_rate=0.1, bp_method="ms"
    )
    bposd_decoder(
        scipy.sparse.csr_matrix(hamming_code(3)), error_rate=0.1, bp_method="ms"
    )

    # scipy sparse csc
    BpOsdDecoder(
        scipy.sparse.csc_matrix(hamming_code(3)), error_rate=0.1, bp_method="ms"
    )
    bposd_decoder(
        scipy.sparse.csc_matrix(hamming_code(3)), error_rate=0.1, bp_method="ms"
    )
