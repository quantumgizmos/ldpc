import pkg_resources
__version__ = pkg_resources.get_distribution('ldpc2').version


# import warnings

# try:
#     from ldpc2.mbp_decoder import mbp_decoder
#     from ldpc2.bp_decoder import bp_decoder
#     from ldpc2.bposd_decoder import bposd_decoder
#     from ldpc2.bf_decoder import bf_decoder
#     from ldpc2.uf_decoder import uf_decoder
#     # from ldpc2.bp_decoder2 import bp_decoder
# except ModuleNotFoundError:
#     warnings.warn("Package installation incomplete")
