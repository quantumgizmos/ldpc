import numpy as np
from ldpc.codes import rep_code
from ldpc.bp_decode_sim import classical_decode_sim

d=500

pcm=rep_code(d)
error_rate=0.3

output_dict={}
output_dict['code_type']=f"rep_code_{d}"

output_dict=classical_decode_sim(
    pcm,
    error_rate,
    target_runs=1000,
    max_iter=30,
    seed=100,
    bp_method='ms',
    ms_scaling_factor=1,
    output_file="classical_bp_decode_sim_output.json",
    output_dict=output_dict
)

print(output_dict)

