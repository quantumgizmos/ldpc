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
        vector<uint8_t> x;
        vector<uint8_t> b;
        bool FULL_REDUCE;
        bool LU_ALLOCATED;
        bool LOWER_TRIANGULAR;
        int rank;

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

        void initiliase_LU(){
            this->U = GF2Sparse<GF2Entry>::New(this->A->m,this->A->n);
            this->L = GF2Sparse<GF2Entry>::New(this->A->m,this->A->m);

            for(int i = 0; i<this->A->m; i++){
                for(auto e: this->A->iterate_row(i)){
                    this->U->insert_entry(e->row_index, e->col_index);
                }
                if(!this->LOWER_TRIANGULAR) this->L->insert_entry(i,i);
            }

            this->LU_ALLOCATED = true;

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




        int rref(bool full_reduce = false, bool lower_triangular = false, vector<int>& cols = NULL_INT_VECTOR, vector<int>& rows = NULL_INT_VECTOR){

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
                for(auto e: this->U->iterate_column(pivot_index)){
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
                
                if(!PIVOT_FOUND) continue;

                if(swap_index!=this->rank){
                    U->swap_rows(swap_index,this->rank);
                    L->swap_rows(swap_index,this->rank);
                    auto temp1 = this->rows[swap_index];
                    auto temp2 = this->rows[this->rank];
                    this->rows[swap_index] = temp2;
                    this->rows[this->rank] = temp1;
                }

                if(this->LOWER_TRIANGULAR) this->L->insert_entry(this->rank,this->rank);

                vector<int> add_rows;
                for(auto e: this->U->iterate_column(pivot_index)){
                    int row_index = e->row_index;
                    if(row_index>this->rank || row_index<this->rank && full_reduce==true){
                        add_rows.push_back(row_index);
                    }
                }

                for(int row: add_rows){
                    this->U->add_rows(row,this->rank);
                    if(lower_triangular) this->L->insert_entry(row,this->rank);
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

            return this->rank;

        }

        vector<uint8_t>& lu_solve(vector<uint8_t>& y){

            if(y.size()!=this->U->m) throw invalid_argument("Input parameter `y` is of the incorrect length for lu_solve.");

            /*
            Equation: Ax = y
 
            We use LU decomposition to arrange the above into the form:
            LU(Qx) = PAQ^T(Qx)=Py

            We can then solve for x using forward-backward substitution:
            1. Forward substitution: Solve Lb = Py for b
            2. Backward subsitution: Solve UQx = b for x
            */


            if(!this->LU_ALLOCATED || !this->LOWER_TRIANGULAR){
                this->rref(false,true);
            }

            std::fill(this->x.begin(),this->x.end(), 0);
            std::fill(this->b.begin(),this->b.end(), 0);

            //Solves LUx=y
            int row_sum;



            //First we solve Lb = y, where b = Ux
            //Solve Lb=y with forwared substitution
            for(int row_index=0;row_index<this->L->m;row_index++){
                row_sum=0;
                for(auto e: L->iterate_row(row_index)){
                    row_sum^=b[e->col_index];
                }
                b[row_index]=row_sum^y[this->rows[row_index]];
            }





            //Solve Ux = b with backwards substitution
            for(int row_index=(rank-1);row_index>=0;row_index--){
                row_sum=0;
                for(auto e: U->iterate_row(row_index)){
                    row_sum^=x[e->col_index];
                }
                x[this->cols[row_index]] = row_sum^b[row_index];
            }

            return x;
       
        }


        auto rref_vrs(bool full_reduce = false, bool lower_triangular = false, vector<int>& cols = NULL_INT_VECTOR, vector<int>& rows = NULL_INT_VECTOR){

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
                for(auto e: this->U->iterate_column(pivot_index)){
                    // int row_index = e->row_index;

                    if(this->inv_rows[e->row_index]<this->rank) continue;

                    int row_weight = this->U->get_row_degree(e->row_index);
                    if(this->inv_rows[e->row_index] >= this->rank && row_weight<max_row_weight){
                        swap_index = this->inv_rows[e->row_index];
                        max_row_weight = row_weight;
                    }
                    PIVOT_FOUND=true;
                    this->pivots[j] = true;
                }
                
                if(!PIVOT_FOUND) continue;

                if(swap_index!=this->rank){
                    // cout<<"Swapping rows "<<swap_index<<" and "<<this->rank<<endl;
                    // U->swap_rows(swap_index,this->inv_rows[this->rank]);
                    // L->swap_rows(swap_index,this->inv_rows[this->rank]);
                    auto temp1 = this->rows[swap_index];
                    auto temp2 = this->rows[this->rank];
                    this->rows[this->rank] = temp1;
                    this->rows[swap_index] = temp2;
                    this->inv_rows[temp1] = this->rank;
                    this->inv_rows[temp2] = swap_index;
                  
                }

                if(this->LOWER_TRIANGULAR) this->L->insert_entry(this->rows[this->rank],this->rank);
                // cout<<"Lower triangular: "<<endl;;
                // print_sparse_matrix(*this->L);
                // cout<<endl;


                vector<int> add_rows;
                for(auto e: this->U->iterate_column(pivot_index)){
                    // int row_index = e->row_index;
                    if(this->inv_rows[e->row_index]>this->rank || this->inv_rows[e->row_index]<this->rank && full_reduce==true){
                        add_rows.push_back(e->row_index);
                    }
                }

                for(int row: add_rows){
                    this->U->add_rows(row,this->rows[this->rank]);
                    if(lower_triangular) this->L->insert_entry(row,this->rank);
                    else this->L->add_rows(row,this->rows[this->rank]);
                    // cout<<"Adding row "<<row<<" to row "<<this->rows[this->rank]<<endl;
                    // print_sparse_matrix(*this->U);
                    // cout<<endl;
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

        vector<uint8_t>& lu_solve_vrs(vector<uint8_t>& y){

            if(y.size()!=this->U->m) throw invalid_argument("Input parameter `y` is of the incorrect length for lu_solve.");

            /*
            Equation: Ax = y
 
            We use LU decomposition to arrange the above into the form:
            LU(Qx) = PAQ^T(Qx)=Py
            We can then solve for x using forward-backward substitution:
            1. Forward substitution: Solve Lb = Py for b
            2. Backward subsitution: Solve UQx = b for x
            */


            if(!this->LU_ALLOCATED || !this->LOWER_TRIANGULAR){
                this->rref(false,true);
            }

            std::fill(this->x.begin(),this->x.end(), 0);
            std::fill(this->b.begin(),this->b.end(), 0);

            //Solves LUx=y
            int row_sum;



            //First we solve Lb = y, where b = Ux
            //Solve Lb=y with forwared substitution
            for(int row_index=0;row_index<this->L->m;row_index++){
                row_sum=0;
                for(auto e: L->iterate_row(this->rows[row_index])){
                    row_sum^=b[e->col_index];
                }
                b[row_index]=row_sum^y[this->rows[row_index]];
            }





            //Solve Ux = b with backwards substitution
            for(int row_index=(rank-1);row_index>=0;row_index--){
                row_sum=0;
                for(auto e: U->iterate_row(this->rows[row_index])){
                    row_sum^=x[e->col_index];
                }
                x[this->cols[row_index]] = row_sum^b[row_index];
            }

            return x;
       
        }


};


// template <class GF2MATRIX>
// shared_ptr<GF2MATRIX> kernel(shared_ptr<GF2MATRIX> mat){

//     auto matT = mat->transpose();
    
//     auto rr = new RowReduce(matT);
//     rr->rref(false,false);
//     int rank = rr->rank;
//     int n = mat->n;
//     int k = n - rank;
//     auto ker = GF2MATRIX::New(k,n);

//     for(int i = rank; i<n; i++){
//         for(auto e: rr->L->iterate_row(i)){
//             ker->insert_entry(i-rank,e->col_index);
//         }
//     }

//     delete rr;
    
//     return ker;

// }

template <class GF2MATRIX>
vector<vector<uint8_t>> kernel(shared_ptr<GF2MATRIX> mat){

    auto matT = mat->transpose();
    
    auto rr = new RowReduce(matT);
    rr->rref(false,false);
    int rank = rr->rank;
    int n = mat->n;
    int k = n - rank;
    // auto ker = GF2MATRIX::New(k,n);

    auto ker = vector<vector<uint8_t>>(k,vector<uint8_t>(n,0));

    for(int i = rank; i<n; i++){
        for(auto e: rr->L->iterate_row(i)){
            ker[i-rank][e->col_index]=1;
        }
    }

    delete rr;
    
    return ker;

}

template <class GF2MATRIX>
int rank(shared_ptr<GF2MATRIX> mat){
    auto rr = new RowReduce(mat);
    rr->rref(false,false);
    int rnk = rr->rank;
    delete rr;
    return rnk;
} 



}//end namespace gf2sparse_linalg

typedef gf2sparse_linalg::RowReduce<shared_ptr<gf2sparse::GF2Sparse<gf2sparse::GF2Entry>>> cy_row_reduce;

using kernel_func = vector<vector<uint8_t>> (*)(std::shared_ptr<gf2sparse::GF2Sparse<gf2sparse::GF2Entry>>);
kernel_func cy_kernel = gf2sparse_linalg::kernel<gf2sparse::GF2Sparse<gf2sparse::GF2Entry>>;

#endif