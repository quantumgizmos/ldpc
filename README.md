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
pip install -U numpy scipy
pip install -i https://test.pypi.org/simple/ ldpc
```