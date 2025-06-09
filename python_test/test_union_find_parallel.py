import numpy as np
import scipy.sparse
import time
from ldpc.union_find_decoder import UnionFindDecoder

def test_union_find_parallel_benchmark():
    hx = scipy.sparse.load_npz('python_test/pcms/hx_surface_20.npz')
    syndrome = np.zeros(hx.shape[0], dtype=np.uint8)
    syndrome[0] = 1
    syndrome[1] = 1
    dec = UnionFindDecoder(hx, uf_method="")
    dec.omp_thread_count = 1
    t0 = time.perf_counter()
    out1 = dec.decode(syndrome)
    t1 = time.perf_counter() - t0
    dec.omp_thread_count = 4
    t0 = time.perf_counter()
    out2 = dec.decode(syndrome)
    t2 = time.perf_counter() - t0
    assert np.array_equal(out1, out2)
    # Simple check that multi-threaded decode is not significantly slower
    assert t2 <= t1 * 2
