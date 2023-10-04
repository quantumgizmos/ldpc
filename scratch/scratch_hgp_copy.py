import numpy as np
import scipy.sparse as sp

from tqdm import tqdm

from ldpc import bp_decoder as bp_og_syntax_decoder
from ldpc import bposd_decoder as osd_og_syntax_decoder


# from qec.css import CssCode
# from qec.hgp import HyperGraphProductCode
# from qec.codes import ToricCode

# qcode = ToricCode(200)

# print(qcode)

# hx = qcode.hx
# hz = qcode.hz
# lx = qcode.lx
# lz = qcode.lz

# sp.save_npz("hx200.npz", hx)
# sp.save_npz("hz200.npz", hz)
# sp.save_npz("lx200.npz", lx)
# sp.save_npz("lz200.npz", lz)

hx = sp.load_npz("scratch/hx200.npz")
hz = sp.load_npz("scratch/hz200.npz")
lx = sp.load_npz("scratch/lx200.npz")
lz = sp.load_npz("scratch/lz200.npz")

run_count = 1000
error_rate = 0.01

# from python_bp import PyBp
# pybp = PyBp(hx.toarray(),error_rate=error_rate, max_iter=50)

osd_og_syntax = osd_og_syntax_decoder(hx,error_rate=error_rate, bp_method='ps', ms_scaling_factor=0.625, max_iter=10,osd_order=5,osd_method="osd_e")
bp_og_syntax = bp_og_syntax_decoder(hx,error_rate=error_rate, bp_method='ms', ms_scaling_factor=0.625, max_iter=10)

seed = 40

for DECODER in [osd_og_syntax]:
    np.random.seed(seed)
    fail = 0

    for i in tqdm(range(run_count)):

        error = np.random.binomial(1, error_rate, hx.shape[1]).astype(np.uint8)
        z = hx@error%2

        # print(np.count_nonzero(z))

        decoding = DECODER.decode(z)

        # assert np.array_equal(hx@decoding%2, z)

        residual = (decoding + error) %2

        # if DECODER == bp:
        #     if not DECODER.converge:
        #         fail+=1
        #         continue

        if np.any((lx@residual)%2):
            fail+=1

    print(f"ler: {fail/run_count}")