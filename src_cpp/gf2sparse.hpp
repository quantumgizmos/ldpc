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

class GF2Entry: public EntryBase<GF2Entry>{ 
    public: 
        ~GF2Entry(){};
};

template <class ENTRY_OBJ = GF2Entry>
class GF2Sparse: public SparseMatrixBase<ENTRY_OBJ>{
    public:
        typedef SparseMatrixBase<ENTRY_OBJ> BASE;
        GF2Sparse(int m, int n): BASE::SparseMatrixBase(m,n){}
        ~GF2Sparse(){}

        static shared_ptr<GF2Sparse<ENTRY_OBJ>> New(int m, int n){
            return make_shared<GF2Sparse<ENTRY_OBJ>>(m,n);
        }

        void csr_row_insert(int row_index, vector<int>& column_indices);
        void csr_insert(vector<vector<int>>& csr_matrix);
        vector<uint8_t>& mulvec(vector<uint8_t>& input_vector, vector<uint8_t>& output_vector);
        vector<uint8_t> mulvec2(vector<uint8_t>& input_vector);
        vector<uint8_t>& mulvec_parallel(vector<uint8_t>& input_vector, vector<uint8_t>& output_vector);
        template<typename ENTRY_OBJ2>
        shared_ptr<GF2Sparse<ENTRY_OBJ>> matmul(shared_ptr<GF2Sparse<ENTRY_OBJ2>> mat_right);
        void add_rows(int i, int j);
        shared_ptr<GF2Sparse<ENTRY_OBJ>> transpose();
        template <typename ENTRY_OBJ2>
        bool gf2_equal(shared_ptr<GF2Sparse<ENTRY_OBJ2>> matrix2);

};

template<class ENTRY_OBJ>
void GF2Sparse<ENTRY_OBJ>::csr_row_insert(int row_index, vector<int>& column_indices){
    for(auto col_index: column_indices){
        this->insert_entry(row_index,col_index);
    }
}

template<class ENTRY_OBJ>
void GF2Sparse<ENTRY_OBJ>::csr_insert(vector<vector<int>>& csr_matrix){
    int i = 0;
    for(auto row: csr_matrix){
        this->csr_row_insert(i, row);
        i++;
    }
}

template <class ENTRY_OBJ>
vector<uint8_t>& GF2Sparse<ENTRY_OBJ>::mulvec(vector<uint8_t>& input_vector, vector<uint8_t>& output_vector){

    for(int i = 0; i<this->m; i++) output_vector[i] = 0;
    for(int i = 0; i < this->m; i++){
        for(auto e: this->iterate_row(i)){
            output_vector[i] ^= input_vector[e->col_index];
        }
    }
    return output_vector;

}

template<class ENTRY_OBJ>
vector<uint8_t> GF2Sparse<ENTRY_OBJ>::mulvec2(vector<uint8_t>& input_vector){
    vector<uint8_t> output_vector(this->m,0);
    for(int i = 0; i < this->m; i++){
        for(auto e: this->iterate_row(i)){
            output_vector[i] ^= input_vector[e->col_index];
        }
    }
    return output_vector;
}

template<class ENTRY_OBJ>
vector<uint8_t>& GF2Sparse<ENTRY_OBJ>::mulvec_parallel(vector<uint8_t>& input_vector, vector<uint8_t>& output_vector){
    #pragma omp for
    for(int i = 0; i<this->m; i++) output_vector[i] = 0;
    
    #pragma omp for
    for(int i = 0; i < this->m; i++){
        for(auto e: this->iterate_row(i)){
            output_vector[i] ^= input_vector[e->col_index];
        }
    }
    return output_vector;
}

template<typename ENTRY_OBJ>
template<typename ENTRY_OBJ2>
shared_ptr<GF2Sparse<ENTRY_OBJ>> GF2Sparse<ENTRY_OBJ>::matmul(shared_ptr<GF2Sparse<ENTRY_OBJ2>> mat_right) {

    if( this->n!=mat_right->m){
        throw invalid_argument("Input matrices have invalid dimensions!");
    }

    auto output_mat = GF2Sparse<ENTRY_OBJ>::New(this->m,mat_right->n);

    
    for(int i = 0; i<output_mat->m; i++){
        for(int j = 0; j<output_mat->n; j++){
            int sum = 0;
            for(auto e: mat_right->iterate_column(j)){
                for(auto g: this->iterate_row(i)){
                    if(g->col_index == e->row_index) sum^=1;
                }
            }
            if(sum) output_mat->insert_entry(i,j);
        }
    }

    return output_mat;    

}

template<class ENTRY_OBJ>
void GF2Sparse<ENTRY_OBJ>::add_rows(int i, int j){
    //row i is the row that will be overwritten
    bool intersection;
    vector<ENTRY_OBJ*> entries_to_remove;

    for(auto g: this->iterate_row(j)){
        intersection=false;
        for(auto e: this->iterate_row(i)){
            if(g->col_index==e->col_index){
                entries_to_remove.push_back(e);
                intersection=true;
                break;
            }
        }
        if(!intersection){
            auto ne = this->insert_entry(i,g->col_index);
        }
    }

    for(auto e: entries_to_remove) this->remove(e); //delete all zero values

}

template<class ENTRY_OBJ>
shared_ptr<GF2Sparse<ENTRY_OBJ>> GF2Sparse<ENTRY_OBJ>::transpose(){
        shared_ptr<GF2Sparse<ENTRY_OBJ>> pcmT = GF2Sparse<ENTRY_OBJ>::New(this->n,this->m);
        for(int i = 0; i<this->m; i++){
            for(auto e: this->iterate_row(i)) pcmT->insert_entry(e->col_index,e->row_index);
        }
        return pcmT;
    }

template<class ENTRY_OBJ>
template <typename ENTRY_OBJ2>
    bool GF2Sparse<ENTRY_OBJ>::gf2_equal(shared_ptr<GF2Sparse<ENTRY_OBJ2>> matrix2){

        if(this->n!=matrix2->n || this->m!=matrix2->m) return false;
        for(int i = 0; i<this->n; i++){
            for(auto e: this->iterate_row(i)){
                auto g = matrix2->get_entry(e->row_index,e->col_index);
                if(g->at_end()) return false;
            }
            for(auto e: matrix2->iterate_row(i)){
                auto g = this->get_entry(e->row_index,e->col_index);
                if(g->at_end()) return false;
            }
        }

        return true;

    }

template <typename ENTRY_OBJ1, typename ENTRY_OBJ2>
bool operator==(shared_ptr<GF2Sparse<ENTRY_OBJ1>> matrix1, shared_ptr<GF2Sparse<ENTRY_OBJ2>> matrix2){
    return matrix1->gf2_equal(matrix2);
}

template <class ENTRY_OBJ = GF2Entry>
shared_ptr<GF2Sparse<ENTRY_OBJ>> gf2_identity(int n){
    auto matrix = GF2Sparse<ENTRY_OBJ>::New(n,n);
    for(int i = 0; i<n; i++) matrix->insert_entry(i,i);
    return matrix;
}


vector<int> NULL_INT_VECTOR = {};


template <class GF2_MATRIX>
class RowReduce{

    public:
    
        GF2_MATRIX A;
        shared_ptr<GF2Sparse<GF2Entry>> L;
        shared_ptr<GF2Sparse<GF2Entry>> U;
        vector<int> rows;
        vector<int> cols;
        vector<int> inv_rows;
        vector<int> inv_cols;
        vector<bool> pivots;
        bool FULL_REDUCE;
        int rank;

        RowReduce(GF2_MATRIX A){

            this->A = A;
            this->pivots.resize(this->A->n,false);
            this->cols.resize(this->A->n);
            this->rows.resize(this->A->m);
            this->inv_cols.resize(this->A->n);
            this->inv_rows.resize(this->A->m);

        }

        ~RowReduce(){}

        void initiliase_LU(){
            this->U = GF2Sparse<GF2Entry>::New(this->A->m,this->A->n);
            this->L = GF2Sparse<GF2Entry>::New(this->A->m,this->A->m);

            for(int i = 0; i<this->A->m; i++){
                for(auto e: this->A->iterate_row(i)){
                    this->U->insert_entry(e->row_index, e->col_index);
                }
                this->L->insert_entry(i,i);
            }

        }
        
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




        GF2_MATRIX rref(bool full_reduce = false, bool lower_triangular = false, vector<int>& cols = NULL_INT_VECTOR, vector<int>& rows = NULL_INT_VECTOR){

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
                for(auto e: this->U->iterate_column(pivot_index)){
                    int row_index = e->row_index;
                    int row_weight = this->U->get_row_degree(row_index);
                    if(row_index >= this->rank && row_weight<max_row_weight){
                        swap_index = e->row_index;
                        max_row_weight = row_weight;
                    }
                    PIVOT_FOUND=true;
                    this->pivots[j] = true;
                }
                
                if(!PIVOT_FOUND) continue;

                if(swap_index!=this->rank){
                    U->swap_rows(swap_index,this->rank);
                    if(!lower_triangular){
                        L->swap_rows(swap_index,this->rank);
                    }
                    auto temp1 = this->rows[swap_index];
                    auto temp2 = this->rows[this->rank];
                    this->rows[swap_index] = temp2;
                    this->rows[this->rank] = temp1;
                }


                vector<int> add_rows;
                for(auto e: this->U->iterate_column(pivot_index)){
                    int row_index = e->row_index;
                    if(row_index>this->rank || row_index<this->rank && full_reduce==true){
                        add_rows.push_back(row_index);
                    }
                }

                for(int row: add_rows){
                    this->U->add_rows(row,this->rank);
                    if(lower_triangular) this->L->insert_entry(row,pivot_index);
                    else this->L->add_rows(row,this->rank);
                }


                this->rank++;

            }

            int pivot_count = 0;
            int non_pivot_count = 0;
            auto orig_cols = this->cols;
            for(int i=0; i<this->U->n; i++){
                if(this->pivots[i]){
                    this->cols[pivot_count] = orig_cols[i];
                    this->inv_cols[this->cols[pivot_count]] = pivot_count;
                    pivot_count++;
                }
                else{
                    this->cols[this->rank + non_pivot_count] = orig_cols[i];
                    this->inv_cols[this->cols[this->rank + non_pivot_count]] = this->rank + non_pivot_count;
                    non_pivot_count++; 
                }
            }

            return this->U;

        }

};



}

typedef gf2sparse::GF2Entry cygf2_entry;
typedef gf2sparse::GF2Sparse<gf2sparse::GF2Entry> cygf2_sparse;

#endif