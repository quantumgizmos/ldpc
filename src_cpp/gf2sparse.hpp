#ifndef GF2SPARSE_H
#define GF2SPARSE_H

#include <iostream>
#include <vector>
#include <memory>
#include <iterator>
#include <algorithm>
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

            void csr_row_insert(int row_index, vector<int>& column_indices){
                for(int col_index: column_indices){
                    this->insert_entry(row_index,col_index);
                }
            }

            void csr_insert(vector<vector<int>>& csr_matrix){
                int i = 0;
                for(auto row: csr_matrix){
                    this->csr_row_insert(i, row);
                    i++;
                }
            }

            vector<uint8_t>& mulvec(vector<uint8_t>& input_vector, vector<uint8_t>& output_vector){
                
                for(int i = 0; i<this->m; i++) output_vector[i] = 0;
                for(int i = 0; i < this->m; i++){
                    for(auto e: this->iterate_row(i)){
                        output_vector[i] ^= input_vector[e->col_index];
                    }
                }
                return output_vector;

            }


            vector<uint8_t> mulvec2(vector<uint8_t>& input_vector){
                
                vector<uint8_t> output_vector;
                output_vector.resize(this->m,0);

                for(int i = 0; i < this->m; i++){
                    for(auto e: this->iterate_row(i)){
                        output_vector[i] ^= input_vector[e->col_index];
                    }
                }
                return std::move(output_vector);

            }



            vector<uint8_t>& mulvec_parallel(vector<uint8_t>& input_vector, vector<uint8_t>& output_vector){
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


            template<typename ENTRY_OBJ2>
            GF2Sparse<ENTRY_OBJ>* matmul(GF2Sparse<ENTRY_OBJ2>* mat_right) {

                if( this->n!=mat_right->m){
                    throw invalid_argument("Input matrices have invalid dimensions!");
                }

                auto output_mat = new GF2Sparse<ENTRY_OBJ>(this->m,mat_right->n);
  
                
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

            void add_rows(int i, int j){
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

        GF2Sparse<ENTRY_OBJ>* transpose(){
            GF2Sparse<ENTRY_OBJ>* pcmT = new GF2Sparse<ENTRY_OBJ>(this->n,this->m);
            for(int i = 0; i<this->m; i++){
                for(auto e: this->iterate_row(i)) pcmT->insert_entry(e->col_index,e->row_index);
            }
            return pcmT;
        }


        template <typename ENTRY_OBJ2>
        bool gf2_equal(GF2Sparse<ENTRY_OBJ2>* matrix2){

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

    };

    vector<int> NULL_INT_VECTOR = {};


    template <class GF2_MATRIX>
    class RowReduce{

        public:
        
            GF2_MATRIX* A;
            GF2Sparse<GF2Entry>* L;
            GF2Sparse<GF2Entry>* U;
            bool L_ALLOCATED;
            bool U_ALLOCATED;
            vector<int> rows;
            vector<int> cols;
            vector<int> inv_rows;
            vector<int> inv_cols;
            vector<bool> pivots;
            bool FULL_REDUCE;
            int rank;

            RowReduce(GF2_MATRIX* A){

                this->A = A;
                this->pivots.resize(this->A->n,false);
                this->cols.resize(this->A->n);
                this->rows.resize(this->A->m);
                this->inv_cols.resize(this->A->n);
                this->inv_rows.resize(this->A->m);

                this->L_ALLOCATED = false;
                this->U_ALLOCATED = false;
    
            }

            ~RowReduce(){
                if(this->L_ALLOCATED) delete this->L;
                if(this->U_ALLOCATED) delete this->U;
            }

            void initiliase_LU(){
                if(this->L_ALLOCATED) delete this->L;
                if(this->U_ALLOCATED) delete this->U;
                this->U = new GF2Sparse<GF2Entry>(this->A->m,this->A->n);
                this->L = new GF2Sparse<GF2Entry>(this->A->m,this->A->m);
                this->L_ALLOCATED = true;
                this->U_ALLOCATED = true;

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




            GF2_MATRIX* rref(bool full_reduce = false, bool lower_triangular = false){

                this->set_column_row_orderings(cols,rows);
                this->initiliase_LU();
                int max_rank = min(this->U->m,this->U->n);
                this->rank = 0;
                std::fill(this->pivots.begin(),this->pivots.end(), false);

                // for(int pivot_index = 0; pivot_index<this->U->n; pivot_index++){

                //     if(this->rank == max_rank) break;

                //     bool PIVOT_FOUND = false;
                //     int max_row_weight = 1e99;
                //     int swap_index;
                //     for(auto e: this->U->iterate_column(pivot_index)){
                //         if(e->row_index >= this->rank && this->U->get_row_degree(e->row_index)<max_row_weight){
                //             swap_index = e->row_index;
                //         }
                //         PIVOT_FOUND=true;
                //         this->pivots[pivot_index] = true;
                //     }
                    
                //     if(!PIVOT_FOUND) continue;

                //     if(swap_index!=this->rank){
                //         U->swap_rows(swap_index,this->rank);
                //         L->swap_rows(swap_index,this->rank);
                //     }

                //     for(auto e: this->U->iterate_column(pivot_index)){
                //         int row_index = this->e->row_index;
                //         if(row_index>this->rank || (row_index<this->rank && full_reduce==true)){
                //             this->U->add_rows(row_index,this->rank);
                //             this->L->add_rows(row_index,this->rank);
                //         }
                //     }


                //     this->rank++;

                // }

                // int pivot_count = 0;
                // int non_pivot_count = 0;
                // for(int i=0; i<this->U->n; i++){
                //     if(this->pivots[i]){
                //         this->cols[pivot_count] = i;
                //         this->inv_cols[i] = pivot_count;
                //         pivot_count++;
                //     }
                //     else{
                //         this->cols[this->rank + non_pivot_count] = i;
                //         this->inv_cols[i] = this->rank + non_pivot_count;
                //         non_pivot_count++; 
                //     }
                // }

                return this->U;



            }

    };



}

typedef gf2sparse::GF2Entry cygf2_entry;
typedef gf2sparse::GF2Sparse<gf2sparse::GF2Entry> cygf2_sparse;

#endif