# LDPC version 2

A C++ rewrite of the `LDPC` package for decoding low density parity check checks. New features include:

- A new C++ template class `GF2Sparse`. This is a more flexible implementation of the `mod2sparse` data structure used in the LDPCv1. This will make it much easier to expand the package.
- Serial (and custom) schedules for the classical BP decoder.
- Openmp support the BP decoder.
- An implementation of weighted union find (with Peeling and inversion solvers).
- An implementation of belief-find (https://arxiv.org/abs/2203.04948)
- An implementation of the Kuo and Lai memory belief propagation decoder (https://arxiv.org/abs/2104.13659)
- Flip and P-flip decoders (https://aps.arxiv.org/abs/2212.06985)

## ToDos

`ldpc` is still a work in progress. Things that still need to be done:
- More decoders could be implemented (eg. small set-flip, https://arxiv.org/abs/1810.03681).
- The LU decomposition routine needs to optimised (it is still slower than the `mod2sparse` version) (getting there...)
- Functions need to be properly documented.
- Proper test coverage is required (in progress).
- The Peeling version of union-find only works for the Toric code. A routine for matching to the boundary needs to be implemented.
- Soft syndrome BP (https://arxiv.org/abs/2205.02341) (almost done)
- Layered schedules (hybrid serial + parallel) (in progress)
- Stabiliser inactivation BP (https://arxiv.org/abs/2205.06125)
- Generalised BP (https://arxiv.org/abs/2212.03214)
- Make a Cython wrapper for the `GF2Sparse<T>` data structure

## Dependencies

The only dependency is `robin_set` implementation of unordered sets, /Copyright (c) 2017 Thibaut Goetghebuer-Planchon/. This is used in the union find decoder.

## Python - Installation from source

The C++ source code can be found in src_cpp. Python bindings are implemented using Cython and can be found in src/ldpc. To install the Python version of the repository follows the instructions below: 

- Download the repo.
- Navigate to the root.
- Pip install with `python>=3.8`.
Note: installation requires a `C` compiler. Eg. `gcc` on Linux or `clang` on Windows.

```
git clone git@github.com:quantumgizmos/ldpc.git
cd ldpc
pip install -Ue
```

## Installation from PyPi

Not yet implemented.

## C++ usage

This is a header only package. See `cpp_test` directory for examples of use.