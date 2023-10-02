# LDPC version 2

A C++ rewrite of the `LDPC` package for decoding low density parity check checks. New features include:

- A new C++ template class `GF2Sparse`. This is a more flexible implementation of the `mod2sparse` data structure used in the LDPCv1. This will make it much easier to expand the package.
- Serial (and custom) schedules for the classical BP decoder.
- An implementation of weighted union find (with Peeling and inversion solvers).
- An implementation of belief-find (https://arxiv.org/abs/2203.04948)
- An implementation of the Kuo and Lai memory belief propagation decoder (https://arxiv.org/abs/2104.13659)
- Flip and P-flip decoders (https://aps.arxiv.org/abs/2212.06985)

## ToDos

`ldpc` is still a work in progress. Things that still need to be done:
- More decoders could be implemented (eg. small set-flip, https://arxiv.org/abs/1810.03681).
- ~~The LU decomposition routine needs to optimised (it is still slower than the `mod2sparse` version) (getting there...)~~
- ~~Soft syndrome BP (https://arxiv.org/abs/2205.02341)~~
- ~~Make a Cython wrapper for the `GF2Sparse<T>` data structure~~
- Layered schedules (hybrid serial + parallel) (in progress). Serial version complete. Hybrid possible with OpenMp?
- Stabiliser inactivation BP (https://arxiv.org/abs/2205.06125)
- Generalised BP (https://arxiv.org/abs/2212.03214)
- Functions need to be properly documented (in progress)
- Proper test coverage is required (C++ has 100%, Python tests still need to expanded).
- The Peeling version of union-find only works for the Toric code. A routine for matching to the boundary needs to be implemented.
- STIM integration for circuit level noise.

## Python - Installation from source

The C++ source code can be found in src_cpp. Python bindings are implemented using Cython and can be found in src/ldpc. To install the Python version of the repository follows the instructions below: 

- Download the repo.
- Navigate to the root.
- Download submodules `git submodule update --init --recursive`
- Pip install with `python>=3.8`.
Note: installation requires a `C` compiler. Eg. `gcc` on Linux or `clang` on Windows.

```
git clone git@github.com:qec-codes/ldpc2.git
cd ldpc2
git submodule update --init --recursive
pip install -Ue
```

## Installation from Test PyPi

This package is currenlty hosted on TestPyPi. Installation requires Python>=3.8. To install, run the following `pip` commands.

```
pip install -U numpy scipy udlr
pip install -i https://test.pypi.org/simple/ ldpc
```

## Quickstart

I have included some *demo* codes in the `ldpc.codes` module. By default, parity check matrices are now represented as `scipy.sparse.csr_matrix` objects.


```python
from ldpc.codes import hamming_code

H = hamming_code(3)

H
```




    <3x7 sparse matrix of type '<class 'numpy.uint8'>'
    	with 12 stored elements in Compressed Sparse Row format>




```python
## To get the dense representaiton of the code, we can use the `scipy.sparse.toarray()` method
H.toarray()
```




    array([[0, 0, 0, 1, 1, 1, 1],
           [0, 1, 1, 0, 0, 1, 1],
           [1, 0, 1, 0, 1, 0, 1]], dtype=uint8)




```python
# Ring code

from ldpc.codes import ring_code

H = ring_code(4)
H.toarray()
```




    array([[1, 1, 0, 0],
           [0, 1, 1, 0],
           [0, 0, 1, 1],
           [1, 0, 0, 1]], dtype=uint8)




```python
# Full rank repetition code

from ldpc.codes import rep_code
H = rep_code(4)
H.toarray()
```




    array([[1, 1, 0, 0],
           [0, 1, 1, 0],
           [0, 0, 1, 1]], dtype=uint8)



# Calculating code properties

Code properties can be calculated with the help of GF2 linear algebra functions in the `udlr.gf2sparse` package. This module is a Python wrapper for my *Up-Down-Left-Right* (UDLR) sparse matrix library written in C++ (this can be installed from https://github.com/qec-codes/udlr). See examples below


```python
from ldpc.codes import hamming_code
import udlr.gf2sparse as gf2sparse

# The rank 4 Hamming code
H = hamming_code(4)

# Physical bits
physical_bit_count = H.shape[1]

# Logical bits (by the Rank-Nullity Theorem)
logical_bit_count = physical_bit_count - gf2sparse.rank(H)

# Print code parameters
print(f"[n = {physical_bit_count}, k = {logical_bit_count}]")
```

    [n = 15, k = 11]



```python
# You can also get a basis of the codewords using the kernel function

codeword_basis = gf2sparse.kernel(H)

codeword_basis.toarray()
```




    array([[1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
           [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
           [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
           [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
           [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]], dtype=uint8)




```python
# Finally, you can use the `gf2sparse.PluDecomposition` class to build your
# own linear algebra routines. Eg. we can write a rref function
import scipy.sparse
import numpy as np
import udlr.gf2sparse as gf2sparse
from typing import Tuple,List
from ldpc.codes import hamming_code

def rref(H: scipy.sparse.spmatrix) -> Tuple[List[int], scipy.sparse.spmatrix]:
    """
    Compute the Reduced Row Echelon Form (RREF) of a given sparse matrix.

    Parameters
    ----------
    H : scipy.sparse.spmatrix
        The input sparse matrix for which the RREF is to be computed.

    Returns
    -------
    Tuple[List[int], scipy.sparse.spmatrix]
        A tuple containing the pivot columns and the RREF of the input matrix.
        - The first element is a list of integers representing the pivot columns.
        - The second element is the RREF of the input matrix as a scipy.sparse.spmatrix.
    """
    
    # Initialise the PLU decomposition class with the input matrix H.
    # We set `full_reduce = False`. This means the funciton does not reduced entries above the diagonal
    # We set `lower_triangular=False`. This means the function does not calculate the lower triangular component
    # of the decmomposition.
    plu_H = gf2sparse.PluDecomposition(H, full_reduce=False, lower_triangular=False)  
    
    # Extract the upper triangular matrix (U) from the PLU decomposition to get the RREF
    rref_H = plu_H.U
    
    # Extract the pivot columns from the PLU decomposition
    pivots_columns = plu_H.pivots

    # Return a tuple containing the pivot columns and the RREF of H
    return (pivots_columns, rref_H)

pivots, H_rref = rref(H)

pivots
H_rref.toarray()
```




    array([[1, 0, 1, ..., 1, 0, 1],
           [0, 1, 1, ..., 0, 1, 1],
           [0, 0, 0, ..., 1, 1, 1],
           ...,
           [0, 0, 0, ..., 1, 1, 1],
           [0, 0, 0, ..., 1, 1, 1],
           [0, 0, 0, ..., 1, 1, 1]], dtype=uint8)



The `udlr.gf2sparse` library is fast. E.g. the RREF of a $32,767$ bit Hamming code can be computed in $<1s$


```python
import udlr.gf2sparse as gf2sparse
from ldpc.codes import hamming_code

H = hamming_code(15)

print(f"[n = {H.shape[1]}, k = {H.shape[1]-gf2sparse.rank(H)}]")
rref(H)
```

    [n = 32767, k = 32752]





    (array([    0,     1,     3,     7,    15,    31,    63,   127,   255,
              511,  1023,  2047,  4095,  8191, 16383], dtype=int32),
     <15x32767 sparse matrix of type '<class 'numpy.uint8'>'
     	with 245760 stored elements in Compressed Sparse Row format>)



## Belief propagation decoding

The belief propagation decoder in LDPCv2 has undergone a complete rewrite to add new functionality and make it easier to extend. There is also new syntax to bring the packge in line with modern Python standards (however, the old syntax should still work). New features include:

- Serial (and layered) schedules.
- Sparse matrix input for all decoders.
- approx. 30% improvement in speed.




```python
from ldpc.bp_decoder import BpDecoder
from ldpc.codes import hamming_code
import numpy as np

# Rank 4 Hamming code
H = hamming_code(4)

# Call the decoder class
decoder = BpDecoder(H, error_rate = 0.1, bp_method="minimum_sum", ms_scaling_factor=0.9, schedule="serial")

syndrome = np.array([1,1,1,1])

dec = decoder.decode(syndrome)
dec
```




    array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])



The sparse matrix input means that the decoder can now be initialised by quickly. E.g. let's initialise a decoder for a massive $1$ million bit repetition code.


```python
from ldpc.bp_decoder import BpDecoder
from ldpc.codes import rep_code
import numpy as np

H = rep_code(1_000_000)
decoder = BpDecoder(H, error_rate = 0.01, schedule = "serial")
```

## BP+OSD Decoding

For decoding quantum codes, it is sometimes better to use a belief propagation + ordered statistics decoder (BP+OSD). The BP+OSD implementation in LDPCv2 is more scalable than the LDPCv1. The perforance improvements can be attributed to a new row reduction routine that preserves sparsity as much as possible.




```python
from ldpc.bposd_decoder import BpOsdDecoder
from ldpc.codes import hamming_code
import numpy as np

# Rank 10 Hamming code
H = hamming_code(10)

# Call the decoder class
decoder = BpOsdDecoder(H, error_rate = 0.1, bp_method="minimum_sum", ms_scaling_factor=0.9, schedule="serial", osd_method = "Exhaustive", osd_order = 5)

syndrome = np.array([1,1,1,1,0,0,0,0,0,0])

dec = decoder.decode(syndrome)
dec
```




    array([0, 0, 0, ..., 0, 0, 0])



# Random Serial Schedules

In Du Crest et al. 2023 (https://arxiv.org/abs/2308.13377v1) it is shown that using a random schedule at each iteration can improve convergence. Random scheduling can now be activated in the LDPCv2 by setting a nonzero `random_schedule_seed` when the decoder is initialised. 



```python
from ldpc.bposd_decoder import BpOsdDecoder
from ldpc.codes import hamming_code
import numpy as np

# Rank 10 Hamming code
H = hamming_code(10)

# Call the decoder class
decoder = BpOsdDecoder(H, error_rate = 0.1, bp_method="minimum_sum", ms_scaling_factor=0.9, schedule="serial", osd_method = "Exhaustive", osd_order = 5, random_schedule_seed = 10)

syndrome = np.array([1,1,1,1,0,0,0,0,0,0])

dec = decoder.decode(syndrome)
dec
```




    array([0, 0, 0, ..., 0, 0, 0])


