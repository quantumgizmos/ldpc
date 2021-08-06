import os
import ldpc
def get_include():
    path = os.path.dirname(ldpc.__file__)
    return path