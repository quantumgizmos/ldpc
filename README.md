![](https://github.com/quantumgizmos/ldpc/workflows/Build/badge.svg)

# LDPC
This module provides a suite of tools for building and benmarking low density parity check (LDPC) codes. Features include functions for mod2 (binary) arithmatic and a fast implementation of the belief propagation decoder.

## Documentation

The full documentation can be found [here](https://roffe.eu/software/ldpc/index.html).

## Installation from PyPi (recommended method)

Installtion from [PyPi](https://pypi.org/project/ldpc/) requires Python>=3.8.
To install via pip, run:

```
pip install -U ldpc
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
