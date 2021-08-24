# ldpc
This module provides a suite of tools for building and benmarking low density parity check (LDPC) codes. Features include functions for mod2 (binary) arithmatic and a fast implementation of the belief propagation decoder.

## Installation from PyPi

Installtion from PyPi requires Python>=3.6.
To install via pip, run:

```
pip install ldpc
```

## Installation (from source)

Installation from sources requires Python>=3.6 and a local C compiler (eg. 'gcc' in Linux or 'clang' in Windows). Once these requirements have been met, navigate to the repository root and install using pip:

```
pip install -e ldpc
```

## Basic usage

In this package error correction codes are represented in terms of their parity check matrix. The parity check matrix for the repetition code can be loaded as follows:


```python
import numpy as np
from ldpc.codes import rep_code
n=5 #specifies the lenght of the repetition code
H=rep_code(n) #returns the repetition code parity check matrix
print(H)
```




    array([[1, 1, 0, 0, 0],
           [0, 1, 1, 0, 0],
           [0, 0, 1, 1, 0],
           [0, 0, 0, 1, 1]])



We now generate a random error and calculate it's syndome


```python
error=np.zeros(n).astype(int) #error vector
error[np.random.randint(n)]=1 #error inserted on a random bit
syndrome=H@error %2 #calculates the error syndrome
print(f"Error: {error}")
print(f"Syndrome: {syndrome}")
```

    Error: [1 0 0 0 0]
    Syndrome: [1 0 0 0]


To decode we first call the bp_decoder class:


```python
from ldpc import bp_decoder
bpd=bp_decoder(H,error_rate=0.05,max_iter=n) #BP decoder class
decoding=bpd.decode(syndrome)
print(f"Decoding: {error}")
```

    Decoding: [1 0 0 0 0]


## Error correction over the binary symmetric channel


```python
n=5
error_rate=0.1
runs=5
H=rep_code(n)
bpd=bp_decoder(H,error_rate=error_rate,max_iter=n,bp_method="product_sum") #BP decoder class. Make sure this is defined outside the loop
error=np.zeros(n).astype(int) #error vector

for _ in range(runs):
    for i in range(n):
        if np.random.random()<error_rate:
            error[i]=1
        else: error[i]=0
    error[np.random.randint(n)]=1 #error inserted on a random bit
    syndrome=H@error %2 #calculates the error syndrome
    print(f"Error: {error}")
    print(f"Syndrome: {syndrome}")
    decoding=bpd.decode(syndrome)
    print(f"Decoding: {error}\n")
```

    Error: [0 0 0 0 1]
    Syndrome: [0 0 0 1]
    Decoding: [0 0 0 0 1]
    
    Error: [0 1 0 0 0]
    Syndrome: [1 1 0 0]
    Decoding: [0 1 0 0 0]
    
    Error: [1 0 0 0 0]
    Syndrome: [1 0 0 0]
    Decoding: [1 0 0 0 0]
    
    Error: [0 1 1 0 1]
    Syndrome: [1 0 1 1]
    Decoding: [0 1 1 0 1]
    
    Error: [0 0 0 1 0]
    Syndrome: [0 0 1 1]
    Decoding: [0 0 0 1 0]
    



```python

```
