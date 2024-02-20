import scipy.sparse
import numpy as np
from tqdm import tqdm

from ldpc.bp_decoder import BpDecoder
from ldpc.belief_find_decoder import BeliefFindDecoder
from ldpc.union_find_decoder import UnionFindDecoder
from ldpc.bposd_decoder import BpOsdDecoder

from ldpc.noise_models import generate_bsc_error

import time

class GdDecoder(BpDecoder):
    def decode(self,syndrome):
        return self.gd_decode(syndrome,10)

def quantum_mc_sim(hx, lx, error_rate, run_count, seed, DECODER, run_label):
    np.random.seed(seed)
    fail = 0
    min_logical = hx.shape[1]
    
    start_time = time.time()  # record start time

    for i in range(run_count):
        # print(i)
        error = generate_bsc_error(hx.shape[1], error_rate)
        z = hx@error%2
        # print(np.nonzero(z)[0])
        decoding = DECODER.decode(z)
        residual = (decoding + error) %2

        if isinstance(DECODER, (BpDecoder,GdDecoder)):
            if not DECODER.converge:
                fail+=1
                continue

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

    return ler, min_logical, speed

def test_882_24_24():

    hx = np.loadtxt("python_test/pcms/lifted_product_[[882,24,24]]_hx.txt").astype(int)
    lx = np.loadtxt("python_test/pcms/lifted_product_[[882,24,24]]_lx.txt").astype(int)

    error_rate = 0.05
    run_count = 50
    seed = 42

    decoder = BpOsdDecoder(hx, error_rate=error_rate, max_iter=50, bp_method="ms", ms_scaling_factor = 0.625, schedule="parallel", osd_method = "osd0")
    ler, min_logical, speed = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder, "Min-sum osd-0 parallel schedule")
        
    gd_decoder = GdDecoder(hx, error_rate=error_rate, max_iter=50, bp_method="ps", schedule="parallel")

    syndrome = hx@generate_bsc_error(hx.shape[1], error_rate)%2

    decoding = gd_decoder.decode(syndrome)
    
    ler, min_logical, speed = quantum_mc_sim(hx, lx, error_rate, run_count, seed, gd_decoder, "Min-sum BpGd parallel schedule")




def test_400_16_6_hgp():

    hx = scipy.sparse.load_npz("python_test/pcms/hx_400_16_6.npz")
    lx = scipy.sparse.load_npz("python_test/pcms/lx_400_16_6.npz")

    error_rate = 0.05
    run_count = 50
    seed = 42

    decoder = BpOsdDecoder(hx, error_rate=error_rate, max_iter=50, bp_method="ms", ms_scaling_factor = 0.625, schedule="parallel", osd_method = "osd0")
    ler, min_logical, speed = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder, "Min-sum osd-0 parallel schedule")
        
    gd_decoder = GdDecoder(hx, error_rate=error_rate, max_iter=50, bp_method="ps", schedule="parallel")

    syndrome = hx@generate_bsc_error(hx.shape[1], error_rate)%2

    decoding = gd_decoder.decode(syndrome)
    
    ler, min_logical, speed = quantum_mc_sim(hx, lx, error_rate, run_count, seed, gd_decoder, "Min-sum BpGd parallel schedule")



 
    # decoder = BpOsdDecoder(hx, error_rate=error_rate, max_iter=5, bp_method="ms", ms_scaling_factor = 0.625, schedule="parallel", osd_method = "osd_cs", osd_order = 5)
    # ler, min_logical, speed = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Min-sum osd-cs-5 parallel schedule")

    # decoder = BpOsdDecoder(hx, error_rate=error_rate, max_iter=5, bp_method="ms", ms_scaling_factor = 0.625, schedule="parallel", osd_method = "osd_e", osd_order = 5)
    # ler, min_logical, speed = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Min-sum osd-e-5 parallel schedule")

    # decoder = BpOsdDecoder(hx, error_rate=error_rate, max_iter=5, bp_method="ms", ms_scaling_factor = 0.625, schedule="serial", osd_method = "osd0")
    # ler, min_logical, speed = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Min-sum osd-0 serial schedule")
  
    # decoder = BpOsdDecoder(hx, error_rate=error_rate, max_iter=5, bp_method="ps", schedule="parallel", osd_method = "osd0")
    # ler, min_logical, speed = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Prod-sum osd-0 parallel schedule")

    # decoder = BeliefFindDecoder(hx, error_rate=error_rate, max_iter=5, bp_method="ms", ms_scaling_factor=0.625, schedule="parallel", uf_method="inversion", bits_per_step=1)
    # ler, min_logical, speed = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Belief-find parallel schedule")


def test_toric_20():

    hx = scipy.sparse.load_npz("python_test/pcms/hx_toric_20.npz")
    lx = scipy.sparse.load_npz("python_test/pcms/lx_toric_20.npz")

    error_rate = 0.05
    run_count = 500
    seed = 42

    decoder = BpOsdDecoder(hx, error_rate=error_rate, max_iter=5, bp_method="ms", ms_scaling_factor = 0.625, schedule="parallel", osd_method = "osd0")
    ler, min_logical, speed = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder, "Min-sum osd-0 parallel schedule")
 
    decoder = BpOsdDecoder(hx, error_rate=error_rate, max_iter=5, bp_method="ms", ms_scaling_factor = 0.625, schedule="parallel", osd_method = "osd_cs", osd_order = 5)
    ler, min_logical, speed = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Min-sum osd-cs-5 parallel schedule")

    decoder = BpOsdDecoder(hx, error_rate=error_rate, max_iter=5, bp_method="ms", ms_scaling_factor = 0.625, schedule="parallel", osd_method = "osd_e", osd_order = 5)
    ler, min_logical, speed = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Min-sum osd-e-5 parallel schedule")

    decoder = BpOsdDecoder(hx, error_rate=error_rate, max_iter=5, bp_method="ms", ms_scaling_factor = 0.625, schedule="serial", osd_method = "osd0")
    ler, min_logical, speed = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Min-sum osd-0 serial schedule")
  
    decoder = BpOsdDecoder(hx, error_rate=error_rate, max_iter=5, bp_method="ps", schedule="parallel", osd_method = "osd0")
    ler, min_logical, speed = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Prod-sum osd-0 parallel schedule")

    decoder = BeliefFindDecoder(hx, error_rate=error_rate, max_iter=5, bp_method="ms", ms_scaling_factor=0.625, schedule="parallel", uf_method="peeling", bits_per_step=1)
    ler, min_logical, speed = quantum_mc_sim(hx, lx, error_rate, run_count, seed, decoder,"Belief-find parallel schedule")

if __name__ == "__main__":
    test_882_24_24()
    # test_toric_20()