import numpy as np
import scipy.sparse
import time
from ldpc.union_find_decoder import UnionFindDecoder


# Helper function to decode a batch of syndromes and return the
# average time per decode along with the decode result of the first
# syndrome (for correctness comparisons).
def _benchmark_decode(hx, syndromes, thread_count):
    dec = UnionFindDecoder(hx, uf_method="")
    dec.omp_thread_count = thread_count
    t0 = time.perf_counter()
    for syn in syndromes:
        dec.decode(np.ascontiguousarray(syn))
    avg_time = (time.perf_counter() - t0) / len(syndromes)
    first_out = dec.decode(np.ascontiguousarray(syndromes[0]))
    return avg_time, first_out


def test_union_find_parallel_benchmark():
    hx = scipy.sparse.load_npz("python_test/pcms/hx_surface_20.npz").tocsr()
    rng = np.random.default_rng(0)
    # Using 128 samples keeps the runtime manageable while still providing
    # a reasonable estimate of performance. Larger sample sizes caused the
    # decoder to hang in this environment.
    num_samples = 128
    thread_counts = [1, 2, 4, 8]
    # Higher error rates occasionally caused the decoder to hang during
    # testing, so we restrict the range here.
    ps = np.linspace(0.01, 0.05, 3)

    results = {}
    for p in ps:
        errors = (rng.random((num_samples, hx.shape[1])) < p).astype(np.uint8)
        syndromes = (hx.dot(errors.T) % 2).astype(np.uint8).T

        avg_1, ref = _benchmark_decode(hx, syndromes, thread_counts[0])
        results[(p, thread_counts[0])] = avg_1
        print(f"p={p:.2f} threads={thread_counts[0]} avg_time={avg_1:.6f}s")

        for t in thread_counts[1:]:
            avg_t, out = _benchmark_decode(hx, syndromes, t)
            results[(p, t)] = avg_t
            print(f"p={p:.2f} threads={t} avg_time={avg_t:.6f}s")
            assert np.array_equal(out, ref)

        assert results[(p, thread_counts[-1])] <= results[(p, thread_counts[0])] * 2
