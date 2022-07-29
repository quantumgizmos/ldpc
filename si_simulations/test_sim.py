from distutils.log import error
import numpy as np
from si_sim import css_decode_sim as css_si
from bposd.css_decode_sim import css_decode_sim

hx=np.loadtxt("si_simulations/hx.txt").astype(int)
hz=np.loadtxt("si_simulations/hz.txt").astype(int)

css_si(hx=hx,hz=hz, error_rate=0.05, xyz_error_bias=[0,0,1],target_runs=1000, osd_order=-1, si_decode=True)
# css_decode_sim(hx=hx,hz=hz, error_rate=0.05, xyz_error_bias=[0,0,1],target_runs=1000, osd_order=40)