#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
from libc.stdlib cimport malloc, calloc, free
from libcpp cimport bool
from libcpp cimport long
from libcpp.string cimport string
from libcpp.vector cimport vector
cimport numpy as np
from ldpc.bp_decoder cimport BpSparse, BpEntry, BpDecoderBase
ctypedef np.uint8_t uint8_t
from libcpp.unordered_map cimport unordered_map as cpp_map
from ldpc.bposd_decoder cimport OsdMethod

cdef extern from "lsd.hpp" namespace "ldpc::lsd":

    cdef const vector[double] EMPTY_DOUBLE_VECTOR "ldpc::lsd::EMPTY_DOUBLE_VECTOR"

    cdef struct ClusterStatistics "ldpc::lsd::ClusterStatistics":
        int final_bit_count
        int undergone_growth_steps
        int nr_merges
        bool active
        vector[int] size_history
        int got_valid_in_timestep
        int got_inactive_in_timestep
        int absorbed_by_cluster
        int nr_of_non_zero_check_matrix_entries
        double cluster_pcm_sparsity
        vector[uint8_t] solution



    cdef struct Statistics "ldpc::lsd::Statistics":
        cpp_map[int, ClusterStatistics] individual_cluster_stats
        cpp_map[int, cpp_map[int, vector[int]]] global_timestep_bit_history
        long elapsed_time
        int lsd_order
        OsdMethod lsd_method
        vector[double] bit_llrs
        vector[uint8_t] error
        vector[uint8_t] syndrome
        vector[uint8_t] compare_recover

    cdef cppclass LsdDecoderCpp "ldpc::lsd::LsdDecoder":
        LsdDecoderCpp(BpSparse& pcm, OsdMethod lsd_method, int lsd_order) except +
        vector[uint8_t]& lsd_decode(vector[uint8_t]& syndrome, const vector[double]& bit_weights, int bits_per_step, bool on_the_fly_decode)
        vector[uint8_t] decoding
        Statistics statistics
        bool do_stats
        OsdMethod lsd_method
        int lsd_order
        bool get_do_stats()
        void set_do_stats(bool do_stats)
        void set_additional_stat_fields(vector[int] error, vector[int] syndrome, vector[int] compare_recover)
        void reset_cluster_stats()

cdef class BpLsdDecoder(BpDecoderBase):
    cdef LsdDecoderCpp* lsd
    cdef int bits_per_step
    cdef vector[uint8_t] bplsd_decoding