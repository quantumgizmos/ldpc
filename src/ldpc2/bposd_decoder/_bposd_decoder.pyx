#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
import warnings
# from scipy.sparse import spmatrix

cdef class BpOsdDecoder(BpDecoderBase):
  
    def __cinit__(self,pcm, error_rate=None, error_channel=None, max_iter=0, bp_method=1, ms_scaling_factor=1.0, schedule=0, omp_thread_count = 1, random_serial_schedule = 0, serial_schedule_order = None, osd_method=0,osd_order=0):
        self.MEMORY_ALLOCATED=False


        ## set up OSD with default values and channel probs from BP
        self.osdD = new OsdDecoderCpp(self.pcm, 0, 0, self.bpd.channel_probs)
        self.osd_order=int(osd_order)
        self.osd_method=int(osd_method)

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


    @property
    def osd_method(self):
        return self.osdD.osd_method

    @osd_method.setter
    def osd_method(self, method: str):
        # OSD method
        if str(method).lower() in ['OSD_0','osd_0','0','osd0']:
            self.osdD.osd_method=0
            self.osdD.osd_order=0
        elif str(method).lower() in ['osd_e','osde','exhaustive','e']:
            self.osdD.osd_method=0
        elif str(osd_method).lower() in ['osd_cs','1','osdcs','combination_sweep','combination_sweep','cs']:
            osd_method=1
        else:
            raise ValueError(f"ERROR: OSD method '{osd_method}' invalid. Please choose from the following methods: 'OSD_0', 'OSD_E' or 'OSD_CS'.")

    @property
    def osd_order(self):
        return self.osdD.osd_order

    @osd_order.setter
    def osd_order(self, order: int):
        # OSD order
        if order<0:
            raise ValueError(f"ERROR: OSD order '{order}' invalid. Please choose a positive integer.")
                
        if self.osdD.method == 0 and osd_order>15:
            warnings.warn("WARNING: Running the 'OSD_E' (Exhaustive method) with search depth\
            greater than 15 is not recommended. Use the 'osd_cs' method instead.")
        
        self.osdD.osd_order=order

    @property
    def decoding(self) -> np.ndarray:
        """
        Returns the current decoded output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """
        out = np.zeros(self.n).astype(int)
        for i in range(self.n):
            out[i] = self.osD.osdw_decoding[i]
        return out

    @property
    def osd0_decoding(self) -> np.ndarray:
        """
        Returns the current decoded output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """
        out = np.zeros(self.n).astype(int)
        for i in range(self.n):
            out[i] = self.osdD.osd0_decoding[i]
        return out

