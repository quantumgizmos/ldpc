from ldpc.codes import rep_code
from bposd.hgp import hgp
import numpy as np
from ldpc import bposd_decoder

h=rep_code(3)
surface_code=hgp(h1=h,h2=h,compute_distance=True) #nb. set compute_distance=False for larger codes

bpd=bposd_decoder(
    surface_code.hz,#the parity check matrix
    error_rate=0.05,
    channel_probs=[None], #assign error_rate to each qubit. This will override "error_rate" input variable
    max_iter=surface_code.N, #the maximum number of iterations for BP)
    bp_method="ms",
    ms_scaling_factor=0, #min sum scaling factor. If set to zero the variable scaling factor method is used
    osd_method="osd_cs", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
    osd_order=7 #the osd search depth
    )

error=np.zeros(surface_code.N).astype(int)
error[[5,12]]=1
syndrome=(surface_code.hz@error %2).astype(np.uint8)
bpd.decode(syndrome)