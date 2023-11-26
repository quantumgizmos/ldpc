#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
from libc.stdlib cimport malloc, calloc, free
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, make_shared
from cython.view cimport array as carray

cimport numpy as np
ctypedef np.uint8_t uint8_t

cdef extern from "sparse_matrix_base.hpp" namespace "ldpc::sparse_matrix_base":
    cdef cppclass CsrMatrix "ldpc::sparse_matrix_base::CsrMatrix":
        int m
        int n
        vector[vector[int]] row_adjacency_list
        int entry_count

cdef extern from "gf2sparse.hpp" namespace "ldpc::gf2sparse":

    cdef cppclass GF2Entry "ldpc::gf2sparse::GF2Entry":
        GF2Entry() except +
        int row_index
        int col_index
        bool at_end()

    cdef cppclass GF2Sparse "ldpc::gf2sparse::GF2Sparse<ldpc::gf2sparse::GF2Entry>":
        int m
        int n
        GF2Sparse() except +
        GF2Sparse(int m, int n) except +
        GF2Entry& insert_entry(int j, int i)
        GF2Entry& get_entry(int i, int j)
        GF2Sparse transpose()
        vector[uint8_t]& mulvec(vector[uint8_t]& input_vector, vector[uint8_t]& output_vector)
        vector[vector[int]] nonzero_coordinates()
        int entry_count()
        vector[vector[int]] row_adjacency_list()
        CsrMatrix to_csr()


cdef extern from "gf2sparse_linalg.hpp" namespace "ldpc::gf2sparse_linalg":

    cdef cppclass RowReduce "ldpc::gf2sparse_linalg::RowReduce<ldpc::gf2sparse::GF2Entry>":
        RowReduce() except +
        RowReduce(GF2Sparse& A) except +
        vector[int] rows
        vector[int] cols
        GF2Sparse L
        GF2Sparse U
        GF2Sparse P
        int rank
        int rref(bool full_reduce, bool lower_triangular)
        vector[uint8_t]& lu_solve(vector[uint8_t]& y)
        void build_p_matrix()
        vector[int] pivots

    CsrMatrix cy_kernel(GF2Sparse* mat)
    CsrMatrix cy_row_complement_basis(GF2Sparse* mat)

from libcpp.vector cimport vector

cdef extern from "gf2dense.hpp" namespace "ldpc::gf2dense":
    int rank_cpp "ldpc::gf2dense::rank" (int row_count, int col_count, vector[vector[int]]& mat)
    vector[vector[int]] gf2dense_kernel "ldpc::gf2dense::kernel" (int row_count, int col_count, vector[vector[int]]& mat)
    vector[int] pivot_rows_cpp "ldpc::gf2dense::pivot_rows" (int row_count, int col_count, vector[vector[int]]& mat)
    vector[vector[int]] row_span_cpp "ldpc::gf2dense::row_span" (int row_count, int col_count, vector[vector[int]]& csr_mat)
    int compute_exact_code_distance_cpp "ldpc::gf2dense::compute_exact_code_distance" (int row_count, int col_count, vector[vector[int]]& csr_mat)

    cdef cppclass DistanceStruct "ldpc::gf2dense::DistanceStruct":
        int min_distance
        int samples_searched
        vector[vector[int]] min_weight_words

    DistanceStruct estimate_code_distance_cpp "ldpc::gf2dense::estimate_code_distance" (int row_count, int col_count, vector[vector[int]]& csr_mat, double timeout_seconds, int number_of_words_to_save)

cdef class PluDecomposition():
    cdef bool _MEM_ALLOCATED
    cdef bool full_reduce
    cdef bool lower_triangular
    cdef bool L_cached
    cdef bool U_cached
    cdef bool P_cached
    cdef int m
    cdef int n
    cdef GF2Sparse* cpcm
    cdef RowReduce* rr
    cdef object Lmat
    cdef object Umat
    cdef object Pmat





    

