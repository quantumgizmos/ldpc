![Build](https://github.com/quantumgizmos/ldpc/workflows/Build/badge.svg)

# LDPC
This module provides a suite of tools for building and benmarking low density parity check (LDPC) codes. Features include functions for mod2 (binary) arithmatic and a fast implementation of the belief propagation decoder.

## Installation from PyPi (recommended method)

Installtion from [PyPi](https://pypi.org/project/ldpc/) requires Python>=3.6.
To install via pip, run:

```
pip install ldpc
```

## Installation (from source)

Installation from source requires Python>=3.6 and a local C compiler (eg. 'gcc' in Linux or 'clang' in Windows). The LDPC package can then be installed by running:

```
git clone https://github.com/quantumgizmos/ldpc.git
cd ldpc
pip install -e ldpc
```

## Dependencies
This package makes use of the `mod2sparse` data structure from Radford Neal's [Software for Low Density Parity Check Codes](https://www.cs.toronto.edu/~radford/ftp/LDPC-2012-02-11/index.html) C package.

## Quick start

### Parity check matrices

In this package error correction codes are represented in terms of their parity check matrix stored in `numpy.ndarray` format. As an example, the parity check matrix for the repetition code can be loaded from the `ldpc.codes` submodule as follows:


```python
import numpy as np
from ldpc.codes import rep_code
n=5 #specifies the lenght of the repetition code
H=rep_code(n) #returns the repetition code parity check matrix
print(H)
```

    [[1 1 0 0 0]
     [0 1 1 0 0]
     [0 0 1 1 0]
     [0 0 0 1 1]]


To compute the [n,k,d] code parameters we can use functions from the `ldpc.mod2` and `ldpc.code_util` submodules. Below is an example showing how to calculate the code parameters of the Hamming code:


```python
from ldpc.codes import hamming_code #function for generating Hamming codes
from ldpc.mod2 import rank #function for calcuting the mod2 rank
from ldpc.code_util import compute_code_distance #function for calculting the code distance

H=hamming_code(3)
print(H)
n=H.shape[1] #block length of the code
k=n-rank(H) #the dimension of the code computed using the rank-nullity theorem.
d=compute_code_distance(H) #computes the code distance
print(f"Hamming code parameters: [n={n},k={k},d={d}]")
```

    [[0 0 0 1 1 1 1]
     [0 1 1 0 0 1 1]
     [1 0 1 0 1 0 1]]
    Hamming code parameters: [n=7,k=4,d=3]


Note that computing the code distance quickly becomes intractable for larger parity check matrices. The `ldpc.code_util.compute_code_distance` should therefore only be used for small codes.

## Belief propagation decoding

To decode using belief propagation, first load an istance of the `ldpc.bp_decoder` class.



```python
from ldpc import bp_decoder
H=rep_code(3)
n=H.shape[1]

bpd=bp_decoder(
    H, #the parity check matrix
    error_rate=0.1, # the error rate on each bit
    max_iter=n, #the maximum iteration depth for BP
    bp_method="product_sum", #BP method. The other option is `minimum_sum'
    channel_probs=[None] #channel probability probabilities. Will overide error rate.
)
```

To decode an error, calculate a syndrome and call the `bp_decoder.decode` function:


```python
error=np.array([0,1,0])
syndrome=H@error%2
decoding=bpd.decode(syndrome)
print(f"Error: {error}")
print(f"Syndrome: {syndrome}")
print(f"Decoding: {decoding}")
```

    Error: [0 1 0]
    Syndrome: [1 1]
    Decoding: [0 1 0]


If the code bits are subject to different error rates, a channel probability vector can be provided instead of the error rate.


```python
bpd=bp_decoder(
    H, 
    max_iter=n,
    bp_method="product_sum", 
    channel_probs=[0.1,0,0.1] #channel probability probabilities. Will overide error rate.
)

error=np.array([1,0,1])
syndrome=H@error%2
decoding=bpd.decode(syndrome)
print(f"Error: {error}")
print(f"Syndrome: {syndrome}")
print(f"Decoding: {decoding}")
```

    Error: [1 0 1]
    Syndrome: [1 1]
    Decoding: [1 0 1]


## Example: error correction over the binary symmetric channel


```python
import numpy as np
from ldpc.codes import rep_code
from ldpc import bp_decoder

n=13
error_rate=0.3
runs=5
H=rep_code(n)

#BP decoder class. Make sure this is defined outside the loop
bpd=bp_decoder(H,error_rate=error_rate,max_iter=n,bp_method="product_sum")
error=np.zeros(n).astype(int) #error vector

for _ in range(runs):
    for i in range(n):
        if np.random.random()<error_rate:
            error[i]=1
        else: error[i]=0
    syndrome=H@error %2 #calculates the error syndrome
    print(f"Error: {error}")
    print(f"Syndrome: {syndrome}")
    decoding=bpd.decode(syndrome)
    print(f"Decoding: {error}\n")
```

    Error: [1 0 1 0 1 0 1 1 0 0 1 0 0]
    Syndrome: [1 1 1 1 1 1 0 1 0 1 1 0]
    Decoding: [1 0 1 0 1 0 1 1 0 0 1 0 0]
    
    Error: [1 0 0 0 0 0 1 1 0 0 0 0 0]
    Syndrome: [1 0 0 0 0 1 0 1 0 0 0 0]
    Decoding: [1 0 0 0 0 0 1 1 0 0 0 0 0]
    
    Error: [0 0 0 0 1 0 0 0 0 0 1 0 0]
    Syndrome: [0 0 0 1 1 0 0 0 0 1 1 0]
    Decoding: [0 0 0 0 1 0 0 0 0 0 1 0 0]
    
    Error: [0 1 1 1 1 0 0 1 1 1 0 1 1]
    Syndrome: [1 0 0 0 1 0 1 0 0 1 1 0]
    Decoding: [0 1 1 1 1 0 0 1 1 1 0 1 1]
    
    Error: [1 0 0 0 0 0 1 0 0 0 0 0 0]
    Syndrome: [1 0 0 0 0 1 1 0 0 0 0 0]
    Decoding: [1 0 0 0 0 0 1 0 0 0 0 0 0]
    



```python

```
