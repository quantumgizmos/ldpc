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

    cdef cppclass bp_entry "cybp_entry":
        bp_entry() except +
        uint8_t value
        bool at_end()

    cdef cppclass bp_sparse "bp::BpSparse":
        bp_sparse(int m, int n) except +
        bp_entry* insert_entry(int i, int j)
        bp_entry* get_entry(int i, int j)
        vector[uint8_t]& mulvec(vector[uint8_t]& input_vector, vector[uint8_t]& output_vector)
        int m
        int n

    cdef cppclass bp_decoder_cpp "bp::BpDecoder":
        bp_decoder_cpp(shared_ptr[bp_sparse] pcm, vector[double]& error_channel, int max_iter, int bp_method, double ms_scaling_factor, int schedule, int omp_threads, vector[int] serial_schedule,int random_schedule) except +
        vector[uint8_t]& decode(vector[uint8_t]& syndrome)
        vector[uint8_t] decoding
        vector[double] log_prob_ratios
        vector[double] channel_probs
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



cdef class bp_decoder_base:
    cdef shared_ptr[bp_sparse] pcm
    cdef int m, n, schedule
    cdef vector[uint8_t] decoding
    cdef vector[uint8_t] syndrome
    cdef vector[double] error_channel
    cdef double error_rate
    cdef bool MEMORY_ALLOCATED
    cdef bp_decoder_cpp *bpd
    cdef str user_dtype
    cdef vector[int] serial_schedule_order
    cdef int random_serial_schedule

cdef class bp_decoder(bp_decoder_base):
    pass

