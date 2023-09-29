import pkg_resources
__version__ = pkg_resources.get_distribution('ldpc2').version

from ldpc2.bp_decoder import BpDecoder
from ldpc2.bposd_decoder import BpOsdDecoder

# Legacy syntax
from ldpc2.bp_decoder import bp_decoder
from ldpc2.bposd_decoder import bposd_decoder 


