# Setup

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