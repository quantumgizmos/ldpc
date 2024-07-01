import json
import time

import numpy as np
import scipy.sparse
from ldpc.belief_find_decoder import BeliefFindDecoder
from ldpc.bp_decoder import BpDecoder
from ldpc.bplsd_decoder import BpLsdDecoder
from ldpc.bposd_decoder import BpOsdDecoder
from ldpc.noise_models import generate_bsc_error


def quantum_mc_sim(hx, lx, error_rate, run_count, seed, DECODER, run_label, DEBUG = False):
    np.random.seed(seed)
    fail = 0
    min_logical = hx.shape[1]
    
    start_time = time.time()  # record start time

    additional_stats = []

    for i in range(run_count):
        if DEBUG: print(f"Iteration: {i}")
        error = generate_bsc_error(hx.shape[1], error_rate)
        z = hx@error%2
        if DEBUG: print(f"Syndrome: {np.nonzero(z)[0].__repr__()}")
        decoding = DECODER.decode(z)
        residual = (decoding + error) %2


        if isinstance(DECODER, BpDecoder):
            if not DECODER.converge:
                fail+=1
                continue

        if isinstance(DECODER, BpLsdDecoder) and (not DECODER.converge):
            clss = DECODER.statistics
            additional_stats.append(clss)
            DECODER.set_additional_stat_fields(error, z, error)

        if np.any((lx@residual)%2):
            fail+=1
            if(np.sum(residual)<min_logical):
                min_logical = np.sum(residual)
                # print(f"New min logical: {min_logical}")

    end_time = time.time()  # record end time

    runtime = end_time - start_time  # compute runtime

    ler, min_logical, speed, fails = (f"ler: {fail / run_count}",
                                      f"min logical: {min_logical}",
                                      f"{run_count / runtime:.2f}it/s",
                                      f"nr fails {fail}")

    print(run_label)
    print(ler)
    print(min_logical)
    print(speed)
    print(fails)
    print()

    return ler, min_logical, speed, additional_stats


def test_400_16_6_hgp():
    hx = scipy.sparse.load_npz("python_test/pcms/hx_400_16_6.npz")
    lx = scipy.sparse.load_npz("python_test/pcms/lx_400_16_6.npz")

    error_rate = 0.01
    run_count = 10000
    seed = 42
    max_iter = 5
    osd_order = 3
    print()
    print(f"Code: [[400, 16, 6]] HGP, error rate: {error_rate}, bp iterations:, {max_iter}, run count: {run_count}, seed: {seed}")
    print("...................................................")
    print()

    decoder = BpOsdDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ms", ms_scaling_factor = 0.625, schedule="parallel", osd_method = "osd0")
    ler, min_logical, speed, _ = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder, "Min-sum osd-0 parallel schedule")
    
    decoder = BpOsdDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ms", ms_scaling_factor=0.625,
                           schedule="parallel", osd_method="osd_cs", osd_order=osd_order)
    ler, min_logical, speed, _ = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,
                                                f"Min-sum osd-cs-{osd_order} parallel schedule")
    
    decoder = BpOsdDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ms", ms_scaling_factor=0.625,
                           schedule="parallel", osd_method="osd_e", osd_order=osd_order)
    ler, min_logical, speed, _ = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,
                                                f"Min-sum osd-e-{osd_order} parallel schedule")
    
    decoder = BpOsdDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ms", ms_scaling_factor = 0.625, schedule="serial", osd_method = "osd0")
    ler, min_logical, speed, _ = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Min-sum osd-0 serial schedule")
    
    decoder = BpOsdDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ps", schedule="parallel", osd_method = "osd0")
    ler, min_logical, speed, _ = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Prod-sum osd-0 parallel schedule")
    
    decoder = BeliefFindDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ms", ms_scaling_factor=0.625, schedule="parallel", uf_method="inversion", bits_per_step=1)
    ler, min_logical, speed, _ = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Belief-find parallel schedule")

    decoder = BpLsdDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ms", ms_scaling_factor=0.625,
                           schedule="parallel", bits_per_step=1, lsd_order=0, lsd_method="osd_e")
    ler, min_logical, speed, _ = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,
                                                f"Min-sum LSD-0 parallel schedule")

    decoder = BpLsdDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ms", ms_scaling_factor=0.625,
                           schedule="parallel", lsd_order=osd_order, lsd_method="lsd_cs")
    # decoder.set_do_stats(True)
    ler, min_logical, speed, _ = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,
                                                f"Min-sum LSD-{osd_order} parallel schedule")
    # print(json.dumps(decoder.statistics))

def test_toric_20():
    hx = scipy.sparse.load_npz("python_test/pcms/hx_toric_20.npz")
    lx = scipy.sparse.load_npz("python_test/pcms/lx_toric_20.npz")

    error_rate = 0.05
    run_count = 1000
    seed = 42
    max_iter = 10

    print(f"Code: [[800, 2, 20]] Toric, error rate: {error_rate}, bp iterations:, {max_iter}, run count: {run_count}, seed: {seed}")
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

    decoder = BeliefFindDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ms", ms_scaling_factor=0.625, schedule="parallel", uf_method="peeling", bits_per_step=1)
    ler, min_logical, speed, _ = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Belief-find parallel schedule")

    decoder = BpLsdDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ms", ms_scaling_factor=0.625,
                           schedule="parallel", bits_per_step=1, lsd_method = "lsd_cs", lsd_order=5)
    ler, min_logical, speed, _ = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Min-sum LSD parallel schedule")

def test_surface_20():
    hx = scipy.sparse.load_npz("python_test/pcms/hx_surface_20.npz")
    lx = scipy.sparse.load_npz("python_test/pcms/lx_surface_20.npz")

    error_rate = 0.05
    run_count = 1000
    seed = 42
    max_iter = 5

    print(hx.shape)

    print(f"Code: [[761, 1, 20]] Surface, error rate: {error_rate}, bp iterations:, {max_iter}, run count: {run_count}, seed: {seed}")
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

    decoder = BeliefFindDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ms", ms_scaling_factor=0.625, schedule="parallel", uf_method="peeling", bits_per_step=1)
    ler, min_logical, speed, _ = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Belief-find parallel schedule")

    decoder = BpLsdDecoder(hx, error_rate=error_rate, max_iter=max_iter, bp_method="ms", ms_scaling_factor=0.625,
                           schedule="parallel", lsd_order=5)
    ler, min_logical, speed, _ = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Min-sum LSD parallel schedule")


if __name__ == "__main__":

    test_400_16_6_hgp()
    test_toric_20()
    test_surface_20()