import importlib.metadata

__version__ = importlib.metadata.version("ldpc")

from ldpc.bp_decoder import BpDecoder
from ldpc.bposd_decoder import BpOsdDecoder
from ldpc.bp_decoder import SoftInfoBpDecoder
from ldpc.belief_find_decoder import BeliefFindDecoder
from ldpc.sinter_decoders import SinterBpOsdDecoder

# Legacy syntax
from ldpc.bp_decoder import bp_decoder
from ldpc.bposd_decoder import bposd_decoder
