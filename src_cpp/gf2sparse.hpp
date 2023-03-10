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


    class GF2Entry: public EntryBase<GF2Entry>{ 
        public: 
            uint8_t value;
            ~GF2Entry(){};
    };

    template <class ENTRY_OBJ = GF2Entry>
    class GF2Sparse: public SparseMatrixBase<ENTRY_OBJ>{
        public:
            typedef SparseMatrixBase<ENTRY_OBJ> BASE;
            using BASE::remove;
            using BASE::insert_entry;

            GF2Sparse(int m, int n): BASE::SparseMatrixBase(m,n){}
            ~GF2Sparse(){}

            ENTRY_OBJ* insert_entry(int i, int j, uint8_t val = uint8_t(1)){
                auto e = BASE::insert_entry(i,j);
                e->value = val;
                return e;
            }

            void csr_row_insert(int row_index, vector<int>& column_indices){
                for(int col_index: column_indices){
                    // cout<<row_index<<" "<<col_index<<endl;
                    this->insert_entry(row_index,col_index,1);
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
                                if(g->col_index == e->row_index) sum^=(g->value*e->value);
                            }
                        }
                        if(sum) output_mat->insert_entry(i,j,1);
                    }
                }

                return output_mat;    

            }


            // template<class ENTRY_OBJ2 = ENTRY_OBJ>
            // GF2Sparse<ENTRY_OBJ>* matmul(GF2Sparse<ENTRY_OBJ2> *mat_right){
            
            //     if( this->n!=mat_right->m){
            //         throw invalid_argument("Input matrices have invalid dimensions!");
            //     }

            //     auto output_mat = new GF2Sparse<ENTRY_OBJ>(this->m,mat_right->n);
  
                
            //     for(int i = 0; i<output_mat->m; i++){
            //         for(int j = 0; j<output_mat->n; j++){
            //             int sum = 0;
            //             for(auto e: mat_right->iterate_column(j)){
            //                 for(auto g: this->iterate_row(i)){
            //                     if(g->col_index == e->row_index) sum^=(g->value*e->value);
            //                 }
            //             }
            //             if(sum) output_mat->insert_entry(i,j,1);
            //         }
            //     }

            //     return output_mat;    

            // }

            void add_rows(int i, int j){
                //row i is the row that will be overwritten
                bool intersection;
                vector<ENTRY_OBJ*> entries_to_remove;

                for(auto g: this->iterate_row(j)){
                    if(g->value==0) continue;
                    intersection=false;
                    for(auto e: this->iterate_row(i)){
                        if(g->col_index==e->col_index){
                            e->value = (e->value + g->value)%2;
                            if(e->value == 0) entries_to_remove.push_back(e);
                            intersection=true;
                            break;
                        }
                    }
                    if(!intersection){
                        auto ne = this->insert_entry(i,g->col_index,g->value);
                    }
                }

                for(auto e: entries_to_remove) this->remove(e); //delete all zero values

            }

            void add_columns(int i, int j){
                //row i is the row that will be overwritten
                bool intersection;
                for(auto g: this->iterate_column(j)){
                    if(g->value==0) continue;
                    intersection=false;
                    for(auto e: this->iterate_column(i)){
                        if(g->row_index==e->row_index){
                            e->value = (e->value + g->value)%2;
                            intersection=true;
                            break;
                        }
                    }
                    if(!intersection){
                        auto ne = this->insert_entry(g->row_index,i,g->value);
                    }
                }
            }


        GF2Sparse<ENTRY_OBJ>* transpose(){
            GF2Sparse<ENTRY_OBJ>* pcmT = new GF2Sparse<ENTRY_OBJ>(this->n,this->m);
            for(int i = 0; i<this->m; i++){
                for(auto e: this->iterate_row(i)) pcmT->insert_entry(e->col_index,e->row_index, 1);
            }
            return pcmT;
        }




    };

typedef GF2Entry cygf2_entry;
typedef GF2Sparse<GF2Entry> cygf2_sparse;

#endif