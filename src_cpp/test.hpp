/**
 * @file gf2sparse.h
 * @brief A C++ library for sparse matrices in GF(2)
 */

#ifndef GF2SPARSE_H
#define GF2SPARSE_H

#include <iostream>
#include <vector>
#include <memory>
#include <iterator>
#include <algorithm>
#include <limits>
#include <omp.h>
#include "sparse_matrix.hpp"
#include "sparse_matrix_util.hpp"

using namespace std;
using namespace sparse_matrix;

namespace gf2sparse{

/**
 * @brief An entry in a sparse matrix over GF(2)
 */
class GF2Entry: public EntryBase<GF2Entry>{ 
    public: 
        ~GF2Entry(){};
};

/**
 * @brief A sparse matrix over GF(2)
 */
template <class ENTRY_OBJ = GF2Entry>
class GF2Sparse: public SparseMatrixBase<ENTRY_OBJ>{
    public:
        typedef SparseMatrixBase<ENTRY_OBJ> BASE;

        /**
         * @brief Constructor for creating a new GF2Sparse object with the given dimensions
         * @param m The number of rows in the matrix
         * @param n The number of columns in the matrix
         */
        GF2Sparse(int m, int n): BASE::SparseMatrixBase(m,n){}

        /**
         * @brief Destructor for GF2Sparse object
         */
        ~GF2Sparse(){}

        /**
         * @brief Creates a new GF2Sparse object with the given dimensions
         * @param m The number of rows in the matrix
         * @param n The number of columns in the matrix
         * @return A shared pointer to the new GF2Sparse object
         */
        static shared_ptr<GF2Sparse<ENTRY_OBJ>> New(int m, int n){
            return make_shared<GF2Sparse<ENTRY_OBJ>>(m,n);
        }

        /**
         * @brief Inserts a row of entries in compressed sparse row (CSR) format
         * @param row_index The index of the row to insert
         * @param column_indices The indices of the non-zero entries in the row
         */
        void csr_row_insert(int row_index, vector<int>& column_indices);

        /**
         * @brief Inserts a matrix in CSR format
         * @param csr_matrix The matrix to insert
         */
        void csr_insert(vector<vector<int>>& csr_matrix);

        /**
         * @brief Multiplies the matrix by a vector and stores the result in another vector
         * @param input_vector The vector to multiply the matrix with
         * @param output_vector The vector to store the result in
         * @return A reference to the output vector
         */
        vector<uint8_t>& mulvec(vector<uint8_t>& input_vector, vector<uint8_t>& output_vector);

        /**
         * @brief Multiplies the matrix by a vector and returns the result as a new vector
         * @param input_vector The vector to multiply the matrix with
         * @return The resulting vector
         */
        vector<uint8_t> mulvec2(vector<uint8_t>& input_vector);

        /**
         * @brief Multiplies the matrix by a vector with parallel execution using OpenMP and stores the result in another vector
         * @param input_vector The vector to multiply the matrix with
         * @param output_vector The vector to store the result in
         * @return A reference to the output vector
         */
        vector<uint8_t>& mulvec_parallel(vector<uint8_t>& input_vector, vector<uint8_t>& output_vector);

        /**
         * @brief Multiplies the matrix by another matrix and returns the result as a new matrix
         * @tparam ENTRY_OBJ2 The type of entries in the matrix to be multiplied with
         * @param
         * @param mat_right The matrix to multiply with
        * @return The resulting matrix
        */
        template<typename ENTRY_OBJ2>
        shared_ptr<GF2Sparse<ENTRY_OBJ>> matmul(shared_ptr<GF2Sparse<ENTRY_OBJ2>> mat_right);

        /**
         * @brief Adds two rows together
         * @param i The row to overwrite
         * @param j The row to add to row i
         */
        void add_rows(int i, int j);

        /**
         * @brief Transposes the matrix
         * @return A shared pointer to the transposed matrix
         */
        shared_ptr<GF2Sparse<ENTRY_OBJ>> transpose();

        /**
         * @brief Compares two matrices for equality
         * @tparam ENTRY_OBJ2 The type of entries in the matrix to compare with
         * @param matrix2 The matrix to compare with
         * @return True if the matrices are equal, false otherwise
         */
        template <typename ENTRY_OBJ2>
        bool gf2_equal(shared_ptr<GF2Sparse<ENTRY_OBJ2>> matrix2);

};

/**

    @brief Creates an identity matrix of the given size
    @tparam ENTRY_OBJ The type of entries in the matrix
    @param n The size of the identity matrix
    @return A shared pointer to the identity matrix
    */
    template <class ENTRY_OBJ = GF2Entry>
    shared_ptr<GF2Sparse<ENTRY_OBJ>> gf2_identity(int n);

}

/**

    @brief Alias for GF2Entry
    */
    typedef gf2sparse::GF2Entry cygf2_entry;

/**

    @brief Alias for GF2Sparse
    */
    typedef gf2sparse::GF2Sparsegf2sparse::GF2Entry cygf2_sparse;

#endif

