#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
from libc.stdlib cimport malloc, calloc, free
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, make_shared
cimport numpy as np
ctypedef np.uint8_t uint8_t

cdef extern from "bp.hpp" namespace "bp" nogil:
    cdef const vector[int] NULL_INT_VECTOR

    cdef cppclass BpEntry "cybp_entry":
        BpEntry() except +
        uint8_t value
        bool at_end()

    cdef cppclass BpSparse "bp::BpSparse":
        BpSparse(int m, int n, int entry_count) except +
        BpEntry* insert_entry(int i, int j)
        BpEntry* get_entry(int i, int j)
        vector[uint8_t]& mulvec(vector[uint8_t]& input_vector, vector[uint8_t]& output_vector)
        int m
        int n

    cdef cppclass BpDecoderCpp "bp::BpDecoder":
        BpDecoderCpp(shared_ptr[BpSparse] pcm, vector[double]& error_channel, int max_iter, int bp_method, double ms_scaling_factor, int schedule, int omp_threads, vector[int] serial_schedule,int random_schedule) except +
        vector[uint8_t]& decode(vector[uint8_t]& syndrome)
        vector[uint8_t]& soft_info_decode_serial(vector[double]& soft_syndrome, double cutoff, double sigma)
        vector[uint8_t] decoding
        vector[double] log_prob_ratios
        vector[double] channel_probs
        vector[double] soft_syndrome
        int converge
        int max_iter
        int omp_thread_count
        vector[int] serial_schedule_order
        int random_schedule
        int iterations
        int random_serial_schedule
        int bp_method
        int schedule
        double ms_scaling_factor
        void set_omp_thread_count(int omp_threads)



cdef class BpDecoderBase:
    cdef int m, n
    cdef shared_ptr[BpSparse] pcm
    cdef vector[uint8_t] _syndrome
    cdef vector[double] _error_channel
    cdef vector[int] _serial_schedule_order
    cdef bool MEMORY_ALLOCATED
    cdef BpDecoderCpp *bpd
    cdef str user_dtype
    
cdef class BpDecoder(BpDecoderBase):
    pass

cdef class SoftInfoBpDecoder(BpDecoderBase):
    cdef double sigma
    cdef double cutoff
    pass
