import numpy as np
from tqdm import tqdm
import json
import time
import datetime

from ldpc import bp_decoder
from .mod2 import rank


def classical_decode_sim(
        pcm,
        error_rate=0,
        max_iter=0,
        target_runs=100,
        seed=0,
        bp_method="ps",
        ms_scaling_factor=0.625,
        output_file=None,
        save_interval=3,
        error_bar_precision_cutoff=1e-2,
        output_dict={}
):

    output_dict['run_type']='classical_decode_sim'

    start_date=datetime.datetime.fromtimestamp(time.time()).strftime("%A, %B %d, %Y %H:%M:%S")
    output_dict.update(dict(start_date=start_date))

    if error_rate<=0:
        raise Exception("Error: The error_rate input variable to simulate_bp_osd is not specified. It must have a value in the range 0.0<error_rate<1.0")

    m,n=pcm.shape
    if max_iter==0: max_iter=n

    output_dict['n']=n
    k=n-rank(pcm)
    output_dict['k']=k
    

  
    bpd=bp_decoder(
        pcm,
        error_rate,
        max_iter=max_iter,
        bp_method=bp_method,
        ms_scaling_factor=ms_scaling_factor
    )

    print(f"BP Method: {bpd.bp_method}")

    if seed==0: seed=np.random.randint(low=1,high=2**32-1)
    np.random.seed(seed)
    print(f"RNG Seed: {seed}")

    if bpd.bp_method=="mininum_sum":
        output_dict['ms_scaling_factor']=ms_scaling_factor


    bp_success=0
    bp_converge_count=0

    output_dict.update(dict(seed=seed,target_runs=target_runs,error_rate=error_rate,max_iter=max_iter,bp_method=bp_method))

    start_time = time.time()
    save_time=start_time
    pbar=tqdm(range(1,target_runs+1),disable=False,ncols=0)
    error=np.zeros(n).astype(int)

    for run_count in pbar:

        for j in range(n):
            if np.random.random()<error_rate:
                error[j]=1
            else: error[j]=0

        syndrome=pcm@error%2

        bp_decoding=bpd.decode(syndrome)

        if bpd.converge:
            bp_converge_count+=1
            if np.array_equal(bp_decoding,error): bp_success+=1

        bp_frame_error_rate=1.0-bp_success/run_count
        bp_frame_error_rate_eb=np.sqrt((1-bp_frame_error_rate)*bp_frame_error_rate/run_count)
        bp_converge_failure_rate=1-bp_converge_count/run_count

        pbar.set_description(f"BP Error Rate: {bp_frame_error_rate*100:.3g}±{bp_frame_error_rate_eb*100:.2g}%")

        current_time=time.time()
        elapsed_time=current_time-start_time
        save_loop=current_time-save_time

        if int(save_loop)>save_interval or run_count==target_runs:
            save_time=time.time()
            output_dict.update(dict(bp_success_count=bp_success,bp_converge_failure_rate=bp_converge_failure_rate,run_count=run_count,bp_frame_error_rate=bp_frame_error_rate,bp_frame_error_rate_eb=bp_frame_error_rate_eb))
            output_dict['elapsed_sim_time']=time.strftime('%H:%M:%S', time.gmtime(elapsed_time))


            if output_file!=None:
                f=open(output_file,"w+")
                print(json.dumps(output_dict,sort_keys=True, indent=4),file=f)
                f.close()

            if bp_frame_error_rate_eb>0 and bp_frame_error_rate_eb/bp_frame_error_rate < error_bar_precision_cutoff:
                print("\nTarget error bar precision reached. Stopping simulation...")
                break


    return json.dumps(output_dict,sort_keys=True, indent=4)