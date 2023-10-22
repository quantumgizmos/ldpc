from ldpc.belief_find_decoder import bp_decoder
import numpy as np
import scipy.sparse


from tqdm import tqdm
def mc_sim(qcode: hgp, error_rate: float = 0.1, runs: int=10, seed: int = 99, DECODER = None)->float:

    hx: scipy.sparse.csr_matrix = scipy.sparse.csr_matrix(qcode.hx)
    lx: scipy.sparse.csr_matrix = scipy.sparse.csr_matrix(qcode.lx)
    error: np.ndarray = np.zeros(hx.shape[1]).astype(np.uint8)

    decoding_success = 0
    np.random.seed(seed)
    for _ in tqdm(range(runs)):

        #generate error
        for i in range(hx.shape[0]):
            rand = np.random.random()
            if rand < error_rate:
                error[i] = 1
            else:
                error[i] = 0

        # print(hx)
        # print(error)
        syndrome = hx@error%2 #calculate syndrome
 
        decoding = DECODER.decode(syndrome) #decode syndrome

        residual_error = (error + decoding) % 2

        #check whether residual error is in the codespace
        if np.any(hx@residual_error%2): continue
        if not np.any(lx@residual_error%2):
            decoding_success+=1