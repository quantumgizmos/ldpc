#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
from libc.stdlib cimport malloc, calloc, free
from libcpp cimport bool
from libcpp.vector cimport vector
cimport numpy as np
ctypedef np.uint8_t uint8_t

cdef extern from "bp.hpp" namespace "ldpc::bp":
    
    cdef const vector[int] NULL_INT_VECTOR

    cdef enum BpMethod:
        PRODUCT_SUM = 0
        MINIMUM_SUM = 1

    cdef enum BpInputType:
        SYNDROME = 0
        RECEIVED_VECTOR = 1
        AUTO = 2

    cdef enum BpSchedule:
        SERIAL = 0
        PARALLEL = 1
        SERIAL_RELATIVE = 2

    cdef cppclass BpEntry "ldpc::bp::BpEntry":
        BpEntry() except +
        bool at_end()

    cdef cppclass BpSparse "ldpc::bp::BpSparse":
        int m
        int n
        BpSparse() except +
        BpSparse(int m, int n, int entry_count) except +
        BpEntry& insert_entry(int i, int j)
        BpEntry& get_entry(int i, int j)
        vector[uint8_t]& mulvec(vector[uint8_t]& input_vector, vector[uint8_t]& output_vector)
        vector[uint8_t] mulvec(vector[uint8_t]& input_vector)

        vector[vector[int]] nonzero_coordinates()
        int entry_count()

        int get_col_degree(int col)
        int get_row_degree(int row)

    cdef cppclass BpDecoderCpp "ldpc::bp::BpDecoder":
            BpDecoderCpp(
                BpSparse& parity_check_matrix,
                vector[double] channel_probabilities,
                int maximum_iterations,
                BpMethod bp_method,
                BpSchedule schedule,
                double min_sum_scaling_factor,
                int omp_threads,
                vector[int] serial_schedule,
                int random_schedule_seed,
                bool random_schedule_at_every_iteration,
                BpInputType bp_input_type) except +
            BpSparse& pcm
            vector[double] channel_probabilities
            int check_count
            int bit_count
            int maximum_iterations
            BpMethod bp_method
            BpSchedule schedule
            double ms_scaling_factor
            vector[uint8_t] decoding
            vector[uint8_t] candidate_syndrome
            vector[double] log_prob_ratios
            vector[double] initial_log_prob_ratios
            vector[double] soft_syndrome
            vector[int] serial_schedule_order
            int iterations
            int omp_thread_count
            bool converge
            int random_schedule_seed
            bool random_schedule_at_every_iteration
            vector[uint8_t] decode(vector[uint8_t]& syndrome)
            vector[uint8_t] soft_info_decode_serial(vector[double]& soft_syndrome, double cutoff, double sigma)
            void set_omp_thread_count(int count)
            BpInputType bp_input_type

cdef class BpDecoderBase:
    cdef BpSparse *pcm
    cdef int m, n
    cdef vector[uint8_t] _syndrome
    cdef vector[double] _error_channel
    cdef vector[int] _serial_schedule_order
    cdef bool MEMORY_ALLOCATED
    cdef BpDecoderCpp *bpd
    cdef str user_dtype
    # cdef int random_schedule_seed
    
cdef class BpDecoder(BpDecoderBase):
    cdef vector[uint8_t] _received_vector
    pass

cdef class SoftInfoBpDecoder(BpDecoderBase):
    cdef double sigma
    cdef double cutoff
    pass
