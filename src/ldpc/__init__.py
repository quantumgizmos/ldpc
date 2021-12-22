import os
from .bp_decoder import bp_decoder
from .osd import bposd_decoder
from . import __file__

def get_include():
    path = os.path.dirname(__file__)
    return path

f=open(get_include()+"/VERSION")
__version__=f.read()
f.close()

