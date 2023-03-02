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


    class gf2entry: public entry_base<gf2entry>{ 
        public: 
            uint8_t value;
            ~gf2entry(){};
    };

    template <class ENTRY_OBJ = gf2entry>
    class gf2sparse: public sparse_matrix_base<ENTRY_OBJ>{
        public:
            typedef sparse_matrix_base<ENTRY_OBJ> BASE;
            using BASE::row_heads; using BASE::column_heads; using BASE::m; using BASE::n;
            using BASE::swap_rows; using BASE::get_entry; using BASE::iterate_column;
            using BASE::iterate_row;
            gf2sparse<gf2entry>* L;
            gf2sparse<gf2entry>* U;
            vector<int> rows;
            vector<int> cols;
            vector<int> orig_cols;
            vector<int> inv_rows;
            vector<int> inv_cols;
            bool L_allocated=false;
            bool U_allocated=false;
            bool LU_indices_allocated=false;
            int rank;
            
            gf2sparse(int m, int n): BASE::sparse_matrix_base(m,n){
                L_allocated=false;
                U_allocated=false;
                cols.resize(n);
                inv_cols.resize(n);
                rows.resize(m);
                inv_rows.resize(m);
            }
            ~gf2sparse(){
                // cout<<"L allcoated "<<L_allocated<<" U_allocated "<<U_allocated<<endl;
                if(L_allocated==true){ delete L; }
                if(U_allocated==true){ delete U; }
            }

            ENTRY_OBJ* insert_entry(int i, int j, uint8_t val = uint8_t(1)){
                auto e = BASE::insert_entry(i,j);
                e->value = val;
                return e;
            }

            vector<uint8_t>& mulvec(vector<uint8_t>& input_vector, vector<uint8_t>& output_vector){
                for(int i = 0; i<m; i++) output_vector[i] = 0;
                for(int i = 0; i < n; i++){
                    if(input_vector[i]){
                        for(auto e: BASE::iterate_column(i)){
                            output_vector[e->row_index] ^= 1;
                        }
                    }
                }
                return output_vector;
            }


            vector<uint8_t>& mulvec_parallel(vector<uint8_t>& input_vector, vector<uint8_t>& output_vector){
                #pragma omp for
                for(int i = 0; i<m; i++) output_vector[i] = 0;
                
                #pragma omp for
                for(int i = 0; i < m; i++){
                    for(auto e: BASE::iterate_row(i)){
                        output_vector[i] ^= input_vector[e->col_index];
                    }
                }
                return output_vector;
            }


            gf2sparse* matmul(gf2sparse *mat_right, gf2sparse *output_mat){
            
                if( n!=mat_right->m || m!=output_mat->m || mat_right->n!=output_mat->n){
                    throw invalid_argument("Input matrices have invalid dimensions!");
                }

                delete output_mat->sparse_matrix_entries;
                delete[] output_mat->row_heads;
                delete[] output_mat->column_heads;
                output_mat->allocate();

                int sum = 0;
                for(int i = 0; i < m; i++){
                    for(int j = 0; j < n; j++){
                        sum=0;
                        for(auto e: BASE::iterate_row(i)){
                            for(auto g: mat_right->iterate_column(j)){
                                if(g->row_index==e->col_index) sum^=(e->value*g->value);
                            }
                        }
                        if(sum==1){
                            output_mat->insert_entry(i,j,1);
                        }
                    }
                }

                return output_mat;    

            }

            void add_rows(int i, int j){
                //row i is the row that will be overwritten
                bool intersection;
                for(auto g: BASE::iterate_row(j)){
                    if(g->value==0) continue;
                    intersection=false;
                    for(auto e: BASE::iterate_row(i)){
                        if(g->col_index==e->col_index){
                            e->value = (e->value + g->value)%2;
                            if(e->value == 0) this->remove(e);
                            intersection=true;
                            break;
                        }
                    }
                    if(!intersection){
                        auto ne = BASE::insert_entry(i,g->col_index);
                        ne->value = g->value;
                    }
                }
            }

            void add_columns(int i, int j){
                //row i is the row that will be overwritten
                bool intersection;
                for(auto g: BASE::iterate_column(j)){
                    if(g->value==0) continue;
                    intersection=false;
                    for(auto e: BASE::iterate_column(i)){
                        if(g->row_index==e->row_index){
                            e->value = (e->value + g->value)%2;
                            intersection=true;
                            break;
                        }
                    }
                    if(!intersection){
                        auto ne = BASE::insert_entry(g->row_index,i);
                        ne->value = g->value;
                    }
                }
            }


            int row_reduce(bool reset_cols = true, bool full_reduce = false){

                int temp1,temp2;
                int pivot_count=0;

                if(U_allocated) delete U;
                if(L_allocated) delete L;
                U=new gf2sparse<gf2entry>(m,n);
                U_allocated=true;
                L=new gf2sparse<gf2entry>(m,m);
                L_allocated=true;
                vector<int> col_tracker;
                vector<int> pivot_cols;
                vector<int> not_pivot_cols;


                if(reset_cols){
                    for(int i=0; i<n;i++) {
                        col_tracker.push_back(i);
                        cols[i] = i;
                        inv_cols[cols[i]] = i;
                    }
                }
                else{
                    for(int i=0; i<n;i++) {
                        col_tracker.push_back(i);
                        inv_cols[cols[i]] = i;
                    }
                }

                
                for(int i=0;i<m;i++){
                    rows[i] = i;
                    inv_rows[rows[i]] = i;
                }

                for(int i=0;i<m;i++){
                    for(auto e: iterate_row(i)){
                        if(e->value==1){
                            U->insert_entry(i,e->col_index,1);
                        }
                    }
                    L->insert_entry(i,i,1);
                }


                for(int i=0;i<m;i++){
                    rows[i] = i;
                    inv_rows[rows[i]]=i;
                }



                int max_rank = min(m,n);

                for(int pivot_column = 0; pivot_column < max_rank; pivot_column++){
                    auto e = U->get_entry(pivot_count,pivot_column);
                    if(!e->at_end()) goto PIVOT_FOUND;


                    PIVOT_FOUND:
                        pivot_count++;
                        pivot_cols.push_back(pivot_column);
                        int a = 0;
                }
                
                return 0;

                }


            

            int lu_decomposition(bool reset_cols = true, bool full_reduce = false){

                int temp1,temp2;
                int pivot_count=0;

                if(U_allocated) delete U;
                if(L_allocated) delete L;
                U=new gf2sparse<gf2entry>(m,n);
                U_allocated=true;
                L=new gf2sparse<gf2entry>(m,m);
                L_allocated=true;
                vector<int> col_tracker;
                vector<int> pivot_cols;
                vector<int> not_pivot_cols;


                if(reset_cols){
                    for(int i=0; i<n;i++) {
                        col_tracker.push_back(i);
                        cols[i] = i;
                        inv_cols[cols[i]] = i;
                    }
                }
                else{
                    for(int i=0; i<n;i++) {
                        col_tracker.push_back(i);
                        inv_cols[cols[i]] = i;
                    }
                }

                
                for(int i=0;i<m;i++){
                    rows[i] = i;
                    inv_rows[rows[i]] = i;
                }

                for(int i=0;i<m;i++){
                    for(auto e: iterate_row(i)){
                        if(e->value==1){
                            U->insert_entry(i,e->col_index,1);
                        }
                    }
                    if(full_reduce == true) L->insert_entry(i,i,1);
                }


                for(int i=0;i<m;i++){
                    rows[i] = i;
                    inv_rows[rows[i]]=i;
                }



                int max_rank = min(m,n);
                // int max_mn = max(m,n);

                // int sub_pivot_index = 0;
                for(int pivot_index = 0; pivot_index < n; pivot_index++){

                    if(pivot_count==max_rank) {
                        break;
                    };

                    auto e = U->get_entry(rows[pivot_count],cols[pivot_index]);
                    if(e->value==1 && !e->at_end()) {
                        // cout<<"Direct pivot found. Pivot indices: "<<e->row_index<<", "<<e->col_index<<" Entry val: "<<unsigned(e->value)<<endl;
                        // print_sparse_matrix(*U);
                        goto pivot_found;
                    } 

                    // cout<<"hello"<<endl;
                    
                    for(auto g: U->iterate_column(cols[pivot_index])){
                            // cout<<"pivot count "<<pivot_count<<" pivot index "<<pivot_index<<" Entry value "<<unsigned(g->value)<<endl;

                            // cout<<inv_rows[g->row_index]<<"pivot: " <<pivot_index<<"sub col pivot: "<<sub_pivot_index<<endl;
                            if((inv_rows[g->row_index] > pivot_count) && (g->value == 1)){
                                

                                temp1=rows[pivot_count];
                                temp2=rows[inv_rows[g->row_index]];

                                rows[pivot_count] = temp2;
                                rows[inv_rows[g->row_index]] = temp1;

                                inv_rows[temp1]=inv_rows[g->row_index];
                                inv_rows[temp2]=pivot_count;
                                // U->swap_rows(pivot_index,g->row_index);
                        
                                // pivot_count+=1;
                                goto pivot_found;
                            }


                    }

                    not_pivot_cols.push_back(cols[pivot_index]);
                    col_tracker[cols[pivot_index]]=-100;
                    continue;
                    pivot_found:
                        if(full_reduce == false){
                            for(auto e: U->iterate_column(cols[pivot_index])){
                                if(e->value==1 && inv_rows[e->row_index] > pivot_count){
                                    U->add_rows(e->row_index, rows[pivot_count]);
                                    L->insert_entry(e->row_index,pivot_count,1);
                                    // cout<<"Pivot: "<<pivot_count<<" Pivot add row: "<<inv_rows[e->row_index]<<endl;
                                    // display_L();
                                    // cout<<endl;
                                    // display_U();

                                }
                            }
                        }

                        else if(full_reduce = true){

                            for(auto e: U->iterate_column(cols[pivot_index])){
                                if(e->value==1 && inv_rows[e->row_index] != pivot_count){
                                    U->add_rows(e->row_index, rows[pivot_count]);
                                    L->add_rows(e->row_index, rows[pivot_count]);
                                    // L->add_rows(rows[pivot_count],e->row_index);
                                    // cout<<"Pivot: "<<pivot_count<<" Pivot add row: "<<inv_rows[e->row_index]<<endl;
                                    // display_L();
                                    // cout<<endl;
                                    // display_U();

                                }
                            }

                        }

                        pivot_count+=1;
                        pivot_cols.push_back(cols[pivot_index]);
                        col_tracker[cols[pivot_index]]=-100;

                    }

                    if(full_reduce == false){
                        for(int i=0; i<m; i++){
                            L->insert_entry(rows[i],i,1);
                        }
                    }

                int col_index=-1;
                for(auto column: pivot_cols){
                    col_index+=1;
                    cols[col_index] = column;
                    inv_cols[cols[col_index]] = col_index;
                }
                for(auto column: not_pivot_cols){
                    col_index+=1;
                    cols[col_index] = column;
                    inv_cols[cols[col_index]] = col_index;
                }
                for(int i=0; i<n; i++){
                    if(col_tracker[i]!=-100){
                        col_index+=1;
                        cols[col_index] = col_tracker[i]; 
                        inv_cols[cols[col_index]] = col_index;
                    }
                }


                // cout<<"Hello"<<endl;
                rank = pivot_count;
                return pivot_count;

            }


        void display_L(){
            for(int i=0; i<m; i++){
                for(int j=0; j<m; j++){
                    auto ii = rows[i];
                    auto e=L->get_entry(ii,j);
                    if(!e->at_end()){
                        cout<<unsigned(e->value)<<" ";
                    }
                    else cout<<0<<" ";
                }
                cout<<endl;
            }
        }

        void display_U(){
            for(int i=0; i<m; i++){
                for(int j=0; j<n; j++){
                    auto ii = rows[i];
                    auto jj = cols[j];
                    auto e=U->get_entry(ii,jj);
                    if(!e->at_end()){
                        cout<<unsigned(e->value)<<" ";
                    }
                    else cout<<0<<" ";
                }
                cout<<endl;
            }
        }

        vector<uint8_t>& lu_solve(vector<uint8_t>& y, vector<uint8_t>& x){

            for(int i=0;i<n;i++) x[i]=0;

            if(!L_allocated){
                lu_decomposition();
            }

            //Solves LUx=y
            int row_sum;
            vector<uint8_t> b;
            b.resize(m);

            //First we solve Lb = y, where b = Ux
            //Solve Lb=y with forwared substitution
            for(int row_index=0;row_index<m;row_index++){
                row_sum=0;
                for(auto e: L->iterate_row(rows[row_index])){
                    row_sum+=e->value*b[e->col_index];
                }
                // cout<<row_sum<<endl;
                b[row_index]=(row_sum%2)^y[rows[row_index]];
            }


            //Solve Ux = b with backwards substitution
            for(int row_index=(rank-1);row_index>=0;row_index--){
                row_sum=0;
                for(auto e: U->iterate_row(rows[row_index])){
                    row_sum+=e->value*x[e->col_index];
                }
                x[cols[row_index]] = (row_sum%2)^b[row_index];
            }

            return x;
       
        }

        gf2sparse<ENTRY_OBJ>* transpose(){
            gf2sparse<ENTRY_OBJ>* pcmT = new gf2sparse<ENTRY_OBJ>(n,m);
            for(int i = 0; i<m; i++){
                for(auto e: BASE::iterate_row(i)) pcmT->insert_entry(e->col_index,e->row_index, 1);
            }
            return pcmT;
        }

        gf2sparse<ENTRY_OBJ>* kernel(){

            auto pcmT=this;

            bool TRANSPOSE_ALLOCATED = false;

            if(n>m) {
                pcmT=this->transpose();
                TRANSPOSE_ALLOCATED = true;
            }

            rank = pcmT->lu_decomposition(true,true);

            int dimension = pcmT->m-rank;

            gf2sparse<ENTRY_OBJ>* kern = new gf2sparse<ENTRY_OBJ>(dimension,pcmT->m);

            for(int i = rank; i<pcmT->L->m; i++){
                for(auto e: pcmT->L->iterate_row(pcmT->rows[i])){
                    kern->insert_entry(i-rank,e->col_index,e->value);
                }
            }

            if(TRANSPOSE_ALLOCATED) delete pcmT;

            return kern;

        }

        vector<vector<int>> nonzero_coordinates(){

            vector<vector<int>> nonzero;

            this->node_count = 0;

            for(int i = 0; i<m; i++){
                for(auto e: iterate_row(i)){
                    if(e->value == 1){
                        this->node_count += 1;
                        vector<int> coord;
                        coord.push_back(e->row_index);
                        coord.push_back(e->col_index);
                        nonzero.push_back(coord);
                    }
                }
            }

            return nonzero;

        }


    };

    // template <class uint8_t = int, template<class> class ENTRY_OBJ = sparse_matrix_entry>
    // vector<uint8_t> gf2sparse_mulvec(gf2sparse<ENTRY_OBJ,uint8_t>* matrix, vector<uint8_t> input_vector, vector<uint8_t> output_vector){
    //     int m,row_sum;
    //     m = matrix->m;
    //     fill(output_vector,output_vector+m,0);
    //     for(int i = 0; i < n; i++){
    //         if(input_vector[i]){
    //             for(auto e: matrix->iterate_column(i)){
    //                 output_vector[e->row_index]^=e->value;
    //             }
    //         }
    //     }
    //     return output_vector;
    // }




typedef gf2entry cygf2_entry;
typedef gf2sparse<gf2entry> cygf2_sparse;

#endif