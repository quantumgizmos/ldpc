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
    for(int i = 0; i<this->m; i++){
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

}

typedef gf2sparse::GF2Entry cygf2_entry;
typedef gf2sparse::GF2Sparse<gf2sparse::GF2Entry> cygf2_sparse;

#endif