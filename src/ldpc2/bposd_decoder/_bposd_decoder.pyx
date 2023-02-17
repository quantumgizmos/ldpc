#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
# from scipy.sparse import spmatrix

cdef class bposd_decoder(bp_decoder_base):
  
    def __cinit__(self,pcm, error_rate=None, error_channel=None, max_iter=0, bp_method=1, ms_scaling_factor=1.0, schedule=0, omp_thread_count = 1, random_serial_schedule = 0, serial_schedule_order = None, osd_method=0,osd_order=0):
        self.MEMORY_ALLOCATED=False

        # OSD method
        if str(osd_method).lower() in ['OSD_0','osd_0','0','osd0']:
            osd_method=0
            osd_order=0
        elif str(osd_method).lower() in ['osd_e','osde','exhaustive','e']:
            osd_method=0
            if osd_order>15:
                print("WARNING: Running the 'OSD_E' (Exhaustive method) with search depth greater than 15 is not recommended. Use the 'osd_cs' method instead.")
        elif str(osd_method).lower() in ['osd_cs','1','osdcs','combination_sweep','combination_sweep','cs']:
            osd_method=1
        else:
            raise ValueError(f"ERROR: OSD method '{osd_method}' invalid. Please choose from the following methods: 'OSD_0', 'OSD_E' or 'OSD_CS'.")

        self.osd_order=int(osd_order)
        self.osd_method=int(osd_method)

        self.osd = new osd_decoder_cpp(self.pcm,self.osd_method,self.osd_order,self.error_channel)
        self.MEMORY_ALLOCATED=True

    def __del__(self):
        if self.MEMORY_ALLOCATED:
            del self.osd

    def decode(self,syndrome):
        
        if not len(syndrome)==self.m:
            raise ValueError(f"The syndrome must have length {self.m}. Not {len(syndrome)}.")
        cdef int i
        DTYPE = syndrome.dtype
        out = np.zeros(self.n,dtype=DTYPE)
        
        for i in range(self.m): self.syndrome[i] = syndrome[i]
        
        self.bpd.decode(syndrome)

        if(self.bpd.converge):
            for i in range(self.n): out[i] = self.bpd.decoding[i]
        else:
            self.osd.decode(self.syndrome,self.bpd.log_prob_ratios)
            for i in range(self.n): out[i] = self.osd.osdw_decoding[i]
        return out