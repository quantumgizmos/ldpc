import scipy.sparse
import numpy as np
from tqdm import tqdm

from ldpc.bp_decoder import BpDecoder
from ldpc.belief_find_decoder import BeliefFindDecoder
from ldpc.union_find_decoder import UnionFindDecoder
from ldpc.bposd_decoder import BpOsdDecoder
from ldpc.bplsd_decoder import BpLsdDecoder

from ldpc.noise_models import generate_bsc_error

import time

def quantum_mc_sim(hx, lx, error_rate, run_count, seed, DECODER, run_label):
    np.random.seed(seed)
    fail = 0
    min_logical = hx.shape[1]
    
    start_time = time.time()  # record start time

    additional_stats = []

    for i in range(run_count):
        error = generate_bsc_error(hx.shape[1], error_rate)
        z = hx@error%2
        decoding = DECODER.decode(z)
        residual = (decoding + error) %2

        if isinstance(DECODER, BpDecoder):
            if not DECODER.converge:
                fail+=1
                continue

        if isinstance(DECODER, BpLsdDecoder) and (not DECODER.converge):
            clss = DECODER.cluster_size_stats
            additional_stats.append(clss)

        if np.any((lx@residual)%2):
            fail+=1
            if(np.sum(residual)<min_logical):
                min_logical = np.sum(residual)
                # print(f"New min logical: {min_logical}")

    end_time = time.time()  # record end time

    runtime = end_time - start_time  # compute runtime
    
    ler, min_logical, speed = (f"ler: {fail/run_count}", f"min logical: {min_logical}", f"{run_count/runtime:.2f}it/s")

    print(run_label)
    print(ler)
    print(min_logical)
    print(speed)
    print()

    return ler, min_logical, speed, additional_stats


def test_400_16_6_hgp():

    hx = scipy.sparse.load_npz("python_test/pcms/hx_400_16_6.npz")
    lx = scipy.sparse.load_npz("python_test/pcms/lx_400_16_6.npz")

    error_rate = 0.05
    run_count = 1000
    seed = 42
    max_iter = 10

    print(f"Code: [[400, 16, 6]] HGP, error rate: {error_rate}, bp iterations:, {max_iter}, run count: {run_count}, seed: {seed}")
    print("...................................................")
    print()

    decoder = BpOsdDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ms", ms_scaling_factor = 0.625, schedule="parallel", osd_method = "osd0")
    ler, min_logical, speed, _ = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder, "Min-sum osd-0 parallel schedule")
 
    decoder = BpOsdDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ms", ms_scaling_factor = 0.625, schedule="parallel", osd_method = "osd_cs", osd_order = 5)
    ler, min_logical, speed, _ = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Min-sum osd-cs-5 parallel schedule")

    decoder = BpOsdDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ms", ms_scaling_factor = 0.625, schedule="parallel", osd_method = "osd_e", osd_order = 5)
    ler, min_logical, speed, _ = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Min-sum osd-e-5 parallel schedule")

    decoder = BpOsdDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ms", ms_scaling_factor = 0.625, schedule="serial", osd_method = "osd0")
    ler, min_logical, speed, _ = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Min-sum osd-0 serial schedule")
  
    decoder = BpOsdDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ps", schedule="parallel", osd_method = "osd0")
    ler, min_logical, speed, _ = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Prod-sum osd-0 parallel schedule")

    decoder = BeliefFindDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ms", ms_scaling_factor=0.625, schedule="parallel", uf_method="inversion", bits_per_step=1)
    ler, min_logical, speed, _ = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Belief-find parallel schedule")

    decoder = BpLsdDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ms", ms_scaling_factor=0.625, schedule="parallel", bits_per_step=1)
    ler, min_logical, speed, _ = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Min-sum LSD parallel schedule")

# def test_toric_20():



#     hx = scipy.sparse.load_npz("python_test/pcms/hx_toric_20.npz")
#     lx = scipy.sparse.load_npz("python_test/pcms/lx_toric_20.npz")

#     error_rate = 0.05
#     run_count = 500
#     seed = 42
#     max_iter = 10

#     print(f"Code: [[800, 2, 20]] Toric, error rate: {error_rate}, bp iterations:, {max_iter}, run count: {run_count}, seed: {seed}")
#     print("...................................................")
#     print()

#     decoder = BpOsdDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ms", ms_scaling_factor = 0.625, schedule="parallel", osd_method = "osd0")
#     ler, min_logical, speed, _ = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder, "Min-sum osd-0 parallel schedule")
 
#     decoder = BpOsdDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ms", ms_scaling_factor = 0.625, schedule="parallel", osd_method = "osd_cs", osd_order = 5)
#     ler, min_logical, speed, _ = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Min-sum osd-cs-5 parallel schedule")

#     decoder = BpOsdDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ms", ms_scaling_factor = 0.625, schedule="parallel", osd_method = "osd_e", osd_order = 5)
#     ler, min_logical, speed, _ = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Min-sum osd-e-5 parallel schedule")

#     decoder = BpOsdDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ms", ms_scaling_factor = 0.625, schedule="serial", osd_method = "osd0")
#     ler, min_logical, speed, _ = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Min-sum osd-0 serial schedule")
  
#     decoder = BpOsdDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ps", schedule="parallel", osd_method = "osd0")
#     ler, min_logical, speed, _ = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Prod-sum osd-0 parallel schedule")

#     decoder = BeliefFindDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ms", ms_scaling_factor=0.625, schedule="parallel", uf_method="peeling", bits_per_step=1)
#     ler, min_logical, speed, _ = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Belief-find parallel schedule")

#     decoder = BpLsdDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ms", ms_scaling_factor=0.625, schedule="parallel", bits_per_step=1)
#     ler, min_logical, speed, _ = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Min-sum LSD parallel schedule")


# def test_cl_size():

#     hx = scipy.sparse.load_npz("python_test/pcms/hx_400_16_6.npz")
#     lx = scipy.sparse.load_npz("python_test/pcms/lx_400_16_6.npz")


#     hx = scipy.sparse.load_npz("python_test/pcms/hx_toric_20.npz")
#     lx = scipy.sparse.load_npz("python_test/pcms/lx_toric_20.npz")

#     error_rate = 0.02
#     run_count = 1000
#     seed = np.random.randint(2e9)
#     max_iter = 1

#     print(f"Code: [[400, 16, 6]] HGP, error rate: {error_rate}, bp iterations:, {max_iter}, run count: {run_count}, seed: {seed}")
#     print("...................................................")
#     print()

#     decoder = BpLsdDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ms", ms_scaling_factor=0.625, schedule="parallel", bits_per_step=1)
#     ler, min_logical, speed, clss = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Min-sum LSD parallel schedule")

#     for cls in clss:
#         print(cls.tolist())

if __name__ == "__main__":

    test_cl_size()