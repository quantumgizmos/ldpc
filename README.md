## LDPC: Software for Decoding Classical and Quantum Codes

LDPC Version 2: A C++ rewrite of the `LDPCv1` package for decoding low density parity check checks.
Warning, whilst efforts have been made to provide backwards compatability with LDPCv1, the new version may introduce breaking changes.

## Documentation

The documentation for `LDPCv2` can be found [here](https://roffe.eu/software/ldpc)

## Installation

The easiest way to install the package is via pip. Python versions `>=3.9` are supported.

```pip install -U ldpc```

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

## LDPCv1
If your package requires LDPCv1, this can be installed from PyPi as follows:

```pip install -U ldpc==0.1.60```

## New features

- A new C++ template class `GF2Sparse`. This is a more flexible implementation of the `mod2sparse` data structure used in the LDPCv1. This will make it much easier to expand the package.
- Serial schedules for the BP decoder.
- Run-time improvements for BP+OSD OSD-0. The decoder now implements the fast-syndrome OSD-0 implementation (https://arxiv.org/abs/1904.02703), where Gaussian elimination is terminated as soon as the syndrome becomes linearly dependent on the reduced columns.
- BP+LSD: Belief propagation plus localised statistics decoding. A parallel decoding algorithm that matches the perforance of BP+OSD. Note that the version implemented currenlty runs in serial. We are working on the parallel version! See our paper: https://arxiv.org/abs/2406.18655
- The union-find matching decoder (https://arxiv.org/abs/1709.06218). This is an implementation of the Delfosse-Nickerson union-find decoder that is suitable for decoding surface codes and other codes with "matchable" syndromes.
- The BeliefFind decoder. A decoder that first runs belief propagation, and falls back on union-find if if the BP decoder fails to converge as proposed by Oscar Higgott in https://arxiv.org/abs/2203.04948
- Flip and P-flip decoders as introduced by Thomas Scruby in https://arxiv.org/abs/2212.06985.
- Improved GF2 linear algebra routines (useful for computing code parameters)

## ToDos

`LDPCv2` is still a work in progress. Ongoing projects are listed below:
- Implement parallel version of BP+LSD algorithm using OpenMP.
- Improve support for parallel processing across the package.
- More decoders could be implemented (eg. small set-flip, https://arxiv.org/abs/1810.03681)
- Stabiliser inactivation BP (https://arxiv.org/abs/2205.06125)
- Generalised BP (https://arxiv.org/abs/2212.03214)
- Functions need to be properly documented (in progress)
- Further STIM integration
- More functionality for studying classical codes. Eg. support for received vector decoding and the AWGN noise channel.

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

## Attribution

If you use this software in your research please cite as follows:

```
@software{Roffe_LDPC_Python_tools_2022,
author = {Roffe, Joschka},
title = {{LDPC: Python tools for low density parity check codes}},
url = {https://pypi.org/project/ldpc/},
year = {2022}
}
```

If you have used the BP+OSD class for quantum error correction, please also cite the following paper:

```
@article{roffe_decoding_2020,
   title={Decoding across the quantum low-density parity-check code landscape},
   volume={2},
   ISSN={2643-1564},
   url={http://dx.doi.org/10.1103/PhysRevResearch.2.043423},
   DOI={10.1103/physrevresearch.2.043423},
   number={4},
   journal={Physical Review Research},
   publisher={American Physical Society (APS)},
   author={Roffe, Joschka and White, David R. and Burton, Simon and Campbell, Earl},
   year={2020},
   month={Dec}
}
```
