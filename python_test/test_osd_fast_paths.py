import numpy as np
import pytest
import scipy.sparse

from ldpc import BpDecoder, BpOsdDecoder
from ldpc.mod2 import rank
from ldpc.codes import hamming_code, rep_code, ring_code


def hgp(h1, h2):
    h1, h2 = scipy.sparse.csr_matrix(h1), scipy.sparse.csr_matrix(h2)
    m1, n1 = h1.shape
    m2, n2 = h2.shape
    return scipy.sparse.hstack(
        [scipy.sparse.kron(h1, scipy.sparse.eye(n2)), scipy.sparse.kron(scipy.sparse.eye(m1), h2.T)]
    ).tocsr().astype(np.uint8)


def random_code(rng, n, m, col_w):
    rows = np.concatenate([rng.choice(m, size=col_w, replace=False) for _ in range(n)])
    cols = np.repeat(np.arange(n), col_w)
    return scipy.sparse.csr_matrix((np.ones(n * col_w, dtype=np.uint8), (rows, cols)), shape=(m, n))


CODES = {
    "hamming4": hamming_code(4),
    "surface5": hgp(rep_code(5), rep_code(5)),
    "toric4": hgp(ring_code(4), ring_code(4)),
    "random": random_code(np.random.default_rng(0), 120, 80, 3),
}


def weight(decoding, probs):
    return np.sum(np.log(1 / probs[decoding == 1]))


@pytest.mark.parametrize("name", CODES)
@pytest.mark.parametrize("osd_method, osd_order", [("osd_0", 0), ("osd_cs", 4), ("osd_cs", 8), ("osd_e", 3)])
def test_osd_decodings_satisfy_the_syndrome(name, osd_method, osd_order):
    h = scipy.sparse.csr_matrix(CODES[name])
    rng = np.random.default_rng(1)
    p = 0.08
    decoder = BpOsdDecoder(h, error_rate=p, max_iter=3, bp_method="product_sum", osd_method=osd_method,
                           osd_order=osd_order)
    probs = np.full(h.shape[1], p)
    for _ in range(30):
        syndrome = h @ (rng.random(h.shape[1]) < p).astype(np.uint8) % 2
        decoding = decoder.decode(syndrome)
        assert np.array_equal(h @ decoding % 2, syndrome)
        if not decoder.converge:  # OSD runs only when BP does not converge
            assert np.array_equal(h @ decoder.osd0_decoding % 2, syndrome)
            assert weight(decoder.osdw_decoding, probs) <= weight(decoder.osd0_decoding, probs) + 1e-9


def test_osd_cs_order_larger_than_the_number_of_non_pivot_columns():
    # rep_code(7) has one non-pivot column: an order above 1 adds no pair candidates
    h = scipy.sparse.csr_matrix(rep_code(7))
    syndromes = [np.array(s, dtype=np.uint8) for s in np.ndindex(*(2,) * h.shape[0])]
    large = BpOsdDecoder(h, error_rate=0.1, max_iter=1, osd_method="osd_cs", osd_order=10)
    small = BpOsdDecoder(h, error_rate=0.1, max_iter=1, osd_method="osd_cs", osd_order=1)
    for s in syndromes:
        assert np.array_equal(large.decode(s), small.decode(s))
        assert np.array_equal(h @ large.decode(s) % 2, s)


def test_osd_cs_full_column_rank():
    # no non-pivot columns: the sweep has no candidates and returns the OSD-0 solution, which is the
    # unique solution. One BP iteration fails on most syndromes of this matrix, so OSD runs.
    h = scipy.sparse.csr_matrix(np.array([
        [0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0], [0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 1, 1], [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1], [0, 1, 1, 0, 1, 0], [0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 1, 1],
    ], dtype=np.uint8))
    assert rank(h) == h.shape[1]
    for order in (1, 3):
        decoder = BpOsdDecoder(h, error_rate=0.1, max_iter=1, osd_method="osd_cs", osd_order=order)
        osd_runs = 0
        for bits in np.ndindex(*(2,) * h.shape[1]):
            error = np.array(bits, dtype=np.uint8)
            decoding = decoder.decode((h @ error % 2).astype(np.uint8))
            if not decoder.converge:
                osd_runs += 1
                assert np.array_equal(decoder.osd0_decoding, error)
                assert np.array_equal(decoding, error)
        assert osd_runs > 0


def test_osd_large_matrix_uses_the_sparse_path():
    # above the dense-matrix budget (m * n / 64 words) both OSD-0 and OSD-CS fall back to the sparse LU
    h = scipy.sparse.csr_matrix(rep_code(50000))
    rng = np.random.default_rng(0)
    error = (rng.random(h.shape[1]) < 0.02).astype(np.uint8)
    syndrome = (h @ error % 2).astype(np.uint8)
    for method, order in (("osd_0", 0), ("osd_cs", 3)):
        decoder = BpOsdDecoder(h, error_rate=0.02, max_iter=1, osd_method=method, osd_order=order)
        assert np.array_equal(h @ decoder.decode(syndrome) % 2, syndrome)


def test_osd_inconsistent_syndrome_does_not_fail():
    # duplicated rows make H rank-deficient; a syndrome that differs on them has no solution
    h = scipy.sparse.vstack([CODES["random"], CODES["random"][:5]]).tocsr()
    syndrome = np.zeros(h.shape[0], dtype=np.uint8)
    syndrome[0] = 1
    for method, order in (("osd_0", 0), ("osd_cs", 5)):
        decoder = BpOsdDecoder(h, error_rate=0.05, max_iter=2, osd_method=method, osd_order=order)
        assert decoder.decode(syndrome).shape == (h.shape[1],)


def test_product_sum_bp_matches_its_definition():
    # one parallel product-sum iteration from the channel llrs, computed directly
    h = scipy.sparse.csr_matrix(CODES["surface5"]).astype(np.uint8)
    rng = np.random.default_rng(3)
    p = rng.uniform(0.01, 0.1, h.shape[1])
    decoder = BpDecoder(h, error_channel=p, max_iter=1, bp_method="product_sum", schedule="parallel",
                        input_vector_type="syndrome")
    syndrome = h @ (rng.random(h.shape[1]) < 0.2).astype(np.uint8) % 2
    assert syndrome.any()  # an all-zero syndrome returns without running BP
    decoder.decode(syndrome)
    dense = h.toarray()
    prior = np.log((1 - p) / p)
    llr = prior.copy()
    for i, row in enumerate(dense):
        cols = np.flatnonzero(row)
        t = np.tanh(prior[cols] / 2)
        for a, j in enumerate(cols):
            prod = np.prod(np.delete(t, a))
            llr[j] += (-1.0) ** int(syndrome[i]) * np.log((1 + prod) / (1 - prod))
    assert np.allclose(decoder.log_prob_ratios, llr, rtol=1e-12, atol=1e-12)


def test_update_channel_probs_array_and_list_agree():
    h = scipy.sparse.csr_matrix(CODES["random"])
    rng = np.random.default_rng(4)
    probs = rng.uniform(0.01, 0.1, h.shape[1])
    a = BpDecoder(h, error_rate=0.05, max_iter=20, input_vector_type="syndrome")
    b = BpDecoder(h, error_rate=0.05, max_iter=20, input_vector_type="syndrome")
    a.update_channel_probs(probs)
    b.update_channel_probs(list(probs))
    assert np.array_equal(a.channel_probs, b.channel_probs)
    assert np.array_equal(a.channel_probs, probs)
    strided = rng.uniform(0.01, 0.1, 2 * h.shape[1])[::2]  # a non-contiguous source
    assert not strided.flags.c_contiguous
    a.update_channel_probs(strided)
    b.update_channel_probs(list(strided))
    assert np.array_equal(a.channel_probs, b.channel_probs)
    assert np.array_equal(a.channel_probs, strided)
    a.error_channel = strided[::-1]  # the setter takes the same path
    assert np.array_equal(a.channel_probs, strided[::-1])


def reference_product_sum(dense, channel, syndrome, max_iter):
    """ldpc 2.4.1's parallel product-sum loop, operation for operation (same libm calls, same order)."""
    import math

    def ieee_log_ratio(x):  # log((1 + x) / (1 - x)) with C's saturation to +-inf instead of Python's exceptions
        num, den = 1 + x, 1 - x
        if den == 0:
            return math.inf
        return -math.inf if num == 0 else math.log(num / den)
    rows = [list(np.flatnonzero(r)) for r in dense]
    cols = [list(np.flatnonzero(c)) for c in dense.T]
    prior = [math.log((1 - q) / q) for q in channel]
    b2c = {(i, j): prior[j] for i, r in enumerate(rows) for j in r}
    c2b = {}
    llr, decoding = [0.0] * len(cols), [0] * len(cols)
    for it in range(1, max_iter + 1):
        for i, r in enumerate(rows):
            temp = 1.0
            for j in r:
                c2b[i, j] = temp
                temp *= math.tanh(b2c[i, j] / 2)
            temp = 1.0
            for j in reversed(r):
                c2b[i, j] *= temp
                c2b[i, j] = (-1.0 if syndrome[i] else 1.0) * ieee_log_ratio(c2b[i, j])
                temp *= math.tanh(b2c[i, j] / 2)
        for j, c in enumerate(cols):
            temp = prior[j]
            for i in c:
                b2c[i, j] = temp
                temp += c2b[i, j]
            llr[j], decoding[j] = temp, int(temp <= 0)
        if np.array_equal(dense @ np.array(decoding) % 2, syndrome):
            return np.array(llr), np.array(decoding, dtype=np.uint8), it, True
        for j, c in enumerate(cols):
            temp = 0.0
            for i in reversed(c):
                b2c[i, j] += temp
                temp += c2b[i, j]
    return np.array(llr), np.array(decoding, dtype=np.uint8), max_iter, False


@pytest.mark.parametrize("name", ["surface5", "random"])
def test_product_sum_bp_is_bitwise_the_reference_loop(name):
    # many iterations, repeated decodes and channel updates on one decoder: the change propagation and the memoised
    # tanh / log must reproduce every llr byte, decoding, iteration count and convergence flag of the plain loop
    h = scipy.sparse.csr_matrix(CODES[name]).astype(np.uint8)
    dense = h.toarray()
    rng = np.random.default_rng(5)
    channel = np.full(h.shape[1], 0.1)
    decoder = BpDecoder(h, error_channel=channel, max_iter=25, bp_method="product_sum", schedule="parallel",
                        input_vector_type="syndrome")
    iterations = set()
    for shot in range(12):
        if shot % 4 == 3:
            channel = rng.uniform(0.02, 0.2, h.shape[1])
            decoder.update_channel_probs(channel)
        syndrome = (dense @ (rng.random(h.shape[1]) < 0.12).astype(np.uint8) % 2).astype(np.uint8)
        if not syndrome.any():
            continue
        decoding = decoder.decode(syndrome)
        llr, ref_decoding, it, converged = reference_product_sum(dense, channel, syndrome, 25)
        assert np.asarray(decoder.log_prob_ratios, dtype=np.float64).tobytes() == llr.tobytes()
        assert np.array_equal(decoding, ref_decoding)
        assert decoder.iter == it
        assert bool(decoder.converge) == converged
        iterations.add(it)
    assert max(iterations) > 1

