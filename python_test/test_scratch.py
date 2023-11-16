import numpy as np

import ldpc.codes
import ldpc.code_util

import numpy as np
import ldpc.codes
from ldpc import BpDecoder
from ldpc import BpOsdDecoder
from ldpc import BeliefFindDecoder
from ldpc.monte_carlo_simulation import MonteCarloBscSimulation
import ldpc.mod2

if __name__ == "__main__":

    H = ldpc.codes.rep_code(5)
    error_rate = 0.001
    dec = BpOsdDecoder(H, error_rate=error_rate, max_iter = 0, bp_method="product_sum", input_vector_type = "syndrome", osd_method="osd_e", osd_order = 1)
    assert dec.osd_order == 1

    span = ldpc.mod2.row_span(ldpc.codes.rep_code(5))
    print(span.toarray())

    H = ldpc.codes.hamming_code(3)

    d = ldpc.mod2.compute_exact_code_distance(H)

    print(d)

    import numpy as np
    import ldpc.codes
    from ldpc import BeliefFindDecoder

    H = ldpc.codes.ring_code(3)

    ## The 
    bf = BeliefFindDecoder(
                H,
                error_rate = 0.1,
                bp_method = 'product_sum',
                max_iter = 1,
                schedule = 'serial',
                matrix_solve = True, # If matrix_solve is set to False, union-find clusters are solved using a peeling decoder
                bits_per_step = 1 ## this is the number of bits by which clusters are expanded in each growth step 
            )
    
    error = np.random.randint(size=H.shape[0], low=0, high=2).astype(np.uint8)
    syndrome = H@error % 2
    

    print(f"Syndrome: {syndrome.__repr__()}")

    decoding = bf.decode(syndrome)
    print(f"Decoding: {decoding}")
    decoding_syndrome = H@decoding % 2
    print(f"Decoding syndrome: {decoding_syndrome}")





    # syndrome 



    # H = np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    #    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    #    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
    #    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    #    [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
    #    [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    #    [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    #    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    #    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0]])
    

    # n,k,d = ldpc.code_util.compute_code_parameters(H,timeout_seconds=0.1)
    # print(f"n = {n}, k = {k}, d = {d}")

    # # exit(22)


    # for _ in range(10000):
    #     H = ldpc.codes.random_binary_code(8,15,4, variance=1)
    #     n,k,d = ldpc.code_util.compute_code_parameters(H,timeout_seconds=0.001)

    #     if d > 4:
    #         print(f"n = {n}, k = {k}, d = {d}")
    #         print(H.toarray().__repr__())