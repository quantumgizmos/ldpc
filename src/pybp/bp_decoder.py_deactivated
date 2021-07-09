import numpy as np
import pybp.bp_decoder_c

#wrapper to return cython class
def bp_decoder(H,error_rate=None,max_iter=0,bp_method=0,ms_scaling_factor=1.0,channel_probs=[None]):

        m=H.shape[0]
        n=H.shape[1]
        
        #Error rate
        if error_rate!=None:
            if error_rate<=0 or error_rate>=1.0:
                raise Exception("ERROR: The error rate must be in the range 0.0<error_rate<1.0")

        #BP iterations
        if max_iter<0: raise Exception('The maximum number of iterations must a positive number')
        if max_iter==0: max_iter=n        

        #BP method
        if str(bp_method).lower() in ['prod_sum','product_sum','ps','0','prod sum']:
            bp_method=0
        elif str(bp_method).lower() in ['min_sum','mininum_sum','ms','1','mininum sum','min sum']:
            bp_method=1
        elif str(bp_method).lower() in ['prod_sum_log','product_sum_log','ps_log','2','psl']:
            bp_method=2
        elif str(bp_method).lower() in ['min_sum_log','mininum_sum_log','ms_log','3','mininum sum_log','msl']:
            bp_method=3
        else: raise Exception(f"BP method '{bp_method}' invalid. Please choose from the following methods: 'product_sum' or 'mininum_sum'.")
        
        if channel_probs[0]!=None:
            if len(channel_probs)!=n:
                raise Exception(f"Error: the channel probability vector must have lenght equal to the block size n={n}.")

        return pybp.bp_decoder_c.bp_decoder(H,error_rate,max_iter=max_iter,bp_method=bp_method,ms_scaling_factor=ms_scaling_factor,channel_probs=channel_probs)
