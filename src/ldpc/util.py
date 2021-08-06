import os
import pathlib
import ldpc
def get_include():
    path = os.path.dirname(ldpc.__file__)
    return path