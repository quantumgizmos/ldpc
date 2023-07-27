/**
 * @file gf2linalg.h
 * @brief Header file for the gf2sparse_linalg namespace, which provides algorithms for performing linear algebra over GF(2).
 * @author [Your Name]
 * @date [Date]
 */

#ifndef GF2LINALG_H
#define GF2LINALG_H

#include <iostream>
#include <vector>
#include <memory>
#include <iterator>
#include <algorithm>
#include <limits>
#include "gf2sparse.hpp"
#include "sparse_matrix_util.hpp"

using namespace std;
// using namespace sparse_matrix;
using namespace gf2sparse;

namespace gf2sparse_linalg{

vector<int> NULL_INT_VECTOR = {};

/**
 * @brief Class for performing row reduction of a sparse GF(2) matrix.
 * 
 * @tparam GF2_MATRIX A sparse GF(2) matrix type.
 */
template <class GF2_MATRIX>
class RowReduce{

    public:
    
        GF2_MATRIX A; /**< Input sparse GF(2) matrix. */
        shared_ptr<GF2Sparse<GF2Entry>> L; /**< Lower triangular matrix resulting from LU decomposition of A. */
        shared_ptr<GF2Sparse<GF2Entry>> U; /**< Upper triangular matrix resulting from LU decomposition of A. */
        vector<int> rows; /**< Vector of row indices corresponding to the rows of A in the desired order. */
        vector<int> cols; /**< Vector of column indices corresponding to the columns of A in the desired order. */
        vector<int> inv_rows; /**< Inverse of the rows vector. */
        vector<int> inv_cols; /**< Inverse of the cols vector. */
        vector<bool> pivots; /**< Boolean vector indicating whether each column of A is a pivot column or not. */
        vector<uint8_t> x; /**< Solution to the linear system Ax = y. */
        vector<uint8_t> b; /**< Intermediate vector used in solving the linear system Ax = y. */
        bool FULL_REDUCE; /**< Boolean indicating whether full row reduction was performed or not. */
        bool LU_ALLOCATED; /**< Boolean indicating whether the L and U matrices have been allocated or not. */
        bool LOWER_TRIANGULAR; /**< Boolean indicating whether the L matrix is a lower triangular matrix or not. */
        int rank; /**< Rank of the matrix A. */

        /**
         * @brief Construct a new RowReduce object from a sparse GF(2) matrix.
         * 
         * @param A Input sparse GF(2) matrix.
         */
        RowReduce(GF2_MATRIX A){

            this->A = A;
            this->pivots.resize(this->A->n,false);
            this->cols.resize(this->A->n);
            this->rows.resize(this->A->m);
            this->inv_cols.resize(this->A->n);
            this->inv_rows.resize(this->A->m);
            this->x.resize(this->A->n);
            this->b.resize(this->A->m);
            this->LU_ALLOCATED = false;
            this->LOWER_TRIANGULAR = false;

        }

        ~RowReduce(){}

        /**
         * @brief Allocate memory for the L and U matrices.
         */
        void initiliase_LU(){
            this->U = GF2Sparse<GF2Entry>::New(this->A->m,this->A->n);
            this->L = GF2Sparse<GF2Entry>::New(this->A->m,this->A->m);

            for(int i = 0; i<this->A->m; i++){
                for(auto e: this->A->iterate_row_ptr(i)){
                    this->U->insert_entry(e->row_index, e->col_index);
                }
                if(!this->LOWER_TRIANGULAR) this->L->insert_entry(i,i);
            }
            this->LU_ALLOCATED = true;

        }

    /**
     * @brief Set the order of the columns and rows of the matrix.
     * 
     * @param cols Vector of column indices corresponding to the columns of A in the desired order.
     * @param rows Vector of row indices corresponding to the rows of A in the desired order.
     */
    void set_column_row_orderings(vector<int>& cols = NULL_INT_VECTOR, vector<int>& rows = NULL_INT_VECTOR){
        
        if(cols==NULL_INT_VECTOR){
            for(int i = 0; i<this->A->n; i++){
                this->cols[i] = i;
                this->inv_cols[this->cols[i]] = i;
            }
        }
        else{
            if(cols.size()!=this->A->n) throw invalid_argument("Input parameter `cols`\
            describing the row ordering is of the incorrect length");
            // this->cols=cols;
            for(int i = 0; i<this->A->n; i++){
                this->cols[i] = cols[i];
                inv_cols[cols[i]] = i;
            }
        }

        if(rows==NULL_INT_VECTOR){
            for(int i = 0; i<this->A->m; i++){
                this->rows[i] = i;
                this->inv_rows[this->rows[i]] = i;
            }
        }
        else{
            if(rows.size()!=this->A->m) throw invalid_argument("Input parameter `rows`\
            describing the row ordering is of the incorrect length");
            // this->rows=rows;
            for(int i = 0; i<this->A->m; i++){
                this->rows[i] = rows[i];
                this->inv_rows[rows[i]] = i;
            }
        }

    }

    /**
     * @brief Perform row reduction on the input matrix.
     * 
     * @param full_reduce Boolean indicating whether to perform full row reduction or not.
     * @param lower_triangular Boolean indicating whether to compute the L matrix as a lower triangular matrix or not.
     * @param cols Vector of column indices corresponding to the columns of A in the desired order.
     * @param rows Vector of row indices corresponding to the rows of A in the desired order.
     * @return GF2_MATRIX The upper triangular matrix resulting from the row reduction of A.
     */
    GF2_MATRIX rref(bool full_reduce = false, bool lower_triangular = false, vector<int>& cols = NULL_INT_VECTOR, vector<int>& rows = NULL_INT_VECTOR){

        if(lower_triangular) this->LOWER_TRIANGULAR = true;
        this->set_column_row_orderings(cols,rows);
        this->initiliase_LU();
        int max_rank = min(this->U->m,this->U->n);
        this->rank = 0;
        std::fill(this->pivots.begin(),this->pivots.end(), false);

        for(int j = 0; j<this->U->n; j++){

            int pivot_index = this->cols[j];
   
            if(this->rank == max_rank) break;


            bool PIVOT_FOUND = false;
            int max_row_weight = std::numeric_limits<int>::max();
            int swap_index;
            for(auto e: this->U->iterate_column_ptr(pivot_index)){
                int row_index = e->row_index;

                if(row_index<this->rank) continue;

                int row_weight = this->U->get_row_degree(row_index);
                if(row_index >= this->rank && row_weight<max_row_weight){
                    swap_index = e->row_index;
                    max_row_weight = row_weight;
                    }
                    PIVOT_FOUND=true;
                    this->pivots[j] = true;
                    }
