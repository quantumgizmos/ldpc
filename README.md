# ldpc2

A C++ rewrite of the `LDPC` package for decoding low density parity check checks. New features include:

- A new C++ template class `gf2sparse`. This is a more flexible implementation of the `mod2sparse` data structure used in the LDPCv1. This will make it much easier to expand the package.
- Serial (and custom) schedules for the classical BP decoder.
- Openmp support the BP decoder.
- An implementation of weighted union find (with Peeling and inversion solvers).
- An implementation of belief-find (https://arxiv.org/abs/2203.04948)
- An implementation of the Kuo and Lai memory belief propagation decoder (https://arxiv.org/abs/2104.13659)

## ToDos

`LDPC2` is still a work in progress. Things that still need to be done:
- More decoders could be implemented (eg. small set-flip).
- The LU decomposition routine needs to optimised (it is still slower than the `mod2sparse` version).
- Functions need to be properly documented.
- Proper test coverage is required.
- The Peeling version of union-find only works for the Toric code. A routine for matching to the boundary needs to be implemented.
- Soft syndrome BP (https://arxiv.org/abs/2205.02341)

## Dependencies

The only dependency is `robin_set` implementation of unordered sets, /Copyright (c) 2017 Thibaut Goetghebuer-Planchon/. This is used in the union find decoder.

## Python - Installation from source

The C++ source code can be found in src_cpp. Python bindings are implemented using Cython and can be found in src/ldpc2. To install the Python version of the repository follows the instructions below: 

- Download the repo.
- Navigate to the root.
- Pip install with `python>=3.6`.
Note: installation requires a `C` compiler. Eg. `gcc` on Linux or `clang` on Windows.

```
git clone git@github.com:quantumgizmos/ldpc2.git
cd ldpc2
pip install -Ue
```

## Installation from PyPi

Not yet implemented.

## C++ usage

This is a header only library. The CMakeLists.txt script provides an example of how to use the header files. An example of the basic features of the `GF2sparse` data structure can be found in `cpp_examples/main.cpp`.
 

## Decoding the Toric code

### Code construction

I can construct a Toric code using the hypergraph product as follows:


```python
import numpy as np
import scipy.sparse
from ldpc.codes import ring_code
from bposd.hgp import hgp

D = 10
h = ring_code(D)
toric_code = hgp(h,h)
toric_code.test()
```

    <Unnamed CSS code>, (2,4)-[[200,2,None]]
     -Block dimensions: Pass
     -PCMs commute hz@hx.T==0: Pass
     -PCMs commute hx@hz.T==0: Pass
     -lx \in ker{hz} AND lz \in ker{hx}: Pass
     -lx and lz anticommute: Pass
     -<Unnamed CSS code> is a valid CSS code w/ params (2,4)-[[200,2,None]]





    True



## Monte Carlo simulation setup


```python
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

    logical_error_rate = 1.0 - decoding_success/runs
    return logical_error_rate
```

### Union-find decoding


```python
from ldpc2.uf_decoder import uf_decoder

DECODER = uf_decoder(toric_code.hx)
logical_error_rate = mc_sim(toric_code,error_rate = 0.07,runs=10000,seed=42, DECODER=DECODER)
print(f"Logical error rate: {logical_error_rate}")
```

    100%|██████████| 10000/10000 [00:02<00:00, 4438.02it/s]

    Logical error rate: 0.0031999999999999806


    


### BP decoding



```python

from ldpc2.bp_decoder import bp_decoder

DECODER = bp_decoder(toric_code.hx, error_rate = 0.07, max_iter = 50, schedule = "serial",bp_method="product_sum")
logical_error_rate = mc_sim(toric_code,error_rate = 0.07,runs=10000,seed=42, DECODER=DECODER)
print(f"Logical error rate: {logical_error_rate}")
```

    100%|██████████| 10000/10000 [00:06<00:00, 1547.70it/s]

    Logical error rate: 0.13319999999999999


    


### BP+OSD Decoding


```python
from ldpc2.bposd_decoder import bposd_decoder

DECODER = bposd_decoder(toric_code.hx, error_rate = 0.07, max_iter = 20, schedule = "serial",bp_method="product_sum",osd_method = "osd_cs", osd_order = 40)
logical_error_rate = mc_sim(toric_code,error_rate = 0.07,runs=10000,seed=42, DECODER=DECODER)
print(f"Logical error rate: {logical_error_rate}")
```

    100%|██████████| 10000/10000 [00:08<00:00, 1188.08it/s]

    Logical error rate: 0.0026000000000000467


    


### BP+Union find (Belief find)




```python
from ldpc2.bf_decoder import bf_decoder

DECODER = bposd_decoder(toric_code.hx, error_rate = 0.07, max_iter = 20, schedule = "serial",bp_method="product_sum")
logical_error_rate = mc_sim(toric_code,error_rate = 0.07,runs=10000,seed=42, DECODER=DECODER)
print(f"Logical error rate: {logical_error_rate}")
```

    100%|██████████| 10000/10000 [00:04<00:00, 2208.53it/s]

    Logical error rate: 0.0026000000000000467


    

