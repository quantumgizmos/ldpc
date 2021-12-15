# Setup

## Installation from PyPi (recommended method)

Installtion from [PyPi](https://pypi.org/project/ldpc/) requires Python>=3.6.
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