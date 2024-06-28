# LDPC version 2 (Beta version)

A C++ rewrite of the `LDPC` package for decoding low density parity check checks. Warning, this version is still in development, so breaking changes may occur.

## New features

- A new C++ template class `GF2Sparse`. This is a more flexible implementation of the `mod2sparse` data structure used in the LDPCv1. This will make it much easier to expand the package.
- Serial schedules for the BP decoder.
- Run-time improvements for BP+OSD OSD-0. The decoder now implements the fast-syndrome OSD-0 implementation (https://arxiv.org/abs/1904.02703), where Gaussian elimination is terminated as soon as the syndrome becomes linearly dependent on the reduced columns.
- BP+LSD: Belief propagation plus localised statistics decoding. A parallel decoding algorithm that matches the perforance of BP+OSD. Note that the version implemented currenlty runs in serial. We are working on the parallel version! See our paper: https://arxiv.org/abs/2406.18655
- The union-find matching decoder (https://arxiv.org/abs/1709.06218). This is an implementation of the Delfosse-Nickerson union-find decoder that is suitable for decoding surface codes and other codes with "matchable" syndromes.
- The BeliefFind decoder. A decoder that first runs belief propagation, and falls back on union-find if if the BP decoder fails to converge as proposed by Oscar Higgott in https://arxiv.org/abs/2203.04948
- Flip and P-flip decoders as introduced by Thomas Scruby in https://arxiv.org/abs/2212.06985.
- Improved GF2 linear algebra routines (useful for computing code parameters)

## Documentation

The documentation for `LDPCv2` can be found [here](https://roffe.eu/software/ldpc2)

## ToDos

`LDPCv2` is still a work in progress. ToDos below:
- I'm struggling to get the package to compile properly via GitHub actions for M1 Macs. Any help would be appreciated!
- Implement parallel version of BP+LSD algorithm using OpenMP.
- Improve support for parallel processing across the package.
- More decoders could be implemented (eg. small set-flip, https://arxiv.org/abs/1810.03681)
- Stabiliser inactivation BP (https://arxiv.org/abs/2205.06125)
- Generalised BP (https://arxiv.org/abs/2212.03214)
- Functions need to be properly documented (in progress)
- STIM integration
- More functionality for studying classical codes. Eg. support for received vector decoding and the AWGN noise channel.
- Proper test coverage is required (C++ has 100%, Python tests still need to expanded).

## Python - Installation from source

The C++ source code can be found in src_cpp. Python bindings are implemented using Cython and can be found in src/ldpc. To install the Python version of the repository follows the instructions below: 

- Download the repo.
- Navigate to the root.
- Pip install with `python>=3.8`.
Note: installation requires a `C` compiler. Eg. `gcc` on Linux or `clang` on Windows.

```
git clone git@github.com:quantumgizmos/ldpc_v2.git
cd ldpc
pip install -Ue .
```

## BP+LSD Quickstart

Usage of the new BP+LSD decoder from https://arxiv.org/abs/2406.18655. Similar to BP+OSD, the LSD decoder can be applied to any parity check matrix. We recommend you start with `lsd_order=0`. The speed/accuracy trade-off for higher order values can be explored from there. Example below:

```python
import numpy as np
import ldpc.codes
from ldpc.bplsd_decoder import BpLsdDecoder

H = ldpc.codes.hamming_code(5)

## The
bp_osd = BpLsdDecoder(
            H,
            error_rate = 0.1,
            bp_method = 'product_sum',
            max_iter = 2,
            schedule = 'serial',
            lsd_method = 'lsd_cs',
            lsd_order = 0
        )

syndrome = np.random.randint(size=H.shape[0], low=0, high=2).astype(np.uint8)

print(f"Syndrome: {syndrome}")
decoding = bp_osd.decode(syndrome)
print(f"Decoding: {decoding}")
decoding_syndrome = H@decoding % 2
print(f"Decoding syndrome: {decoding_syndrome}")
``` 

