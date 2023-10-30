import pkg_resources
__version__ = pkg_resources.get_distribution('ldpc').version

from ldpc.bp_decoder import BpDecoder
from ldpc.bposd_decoder import BpOsdDecoder
from ldpc.bp_decoder import SoftInfoBpDecoder

# Legacy syntax
from ldpc.bp_decoder import bp_decoder
from ldpc.bposd_decoder import bposd_decoder 


