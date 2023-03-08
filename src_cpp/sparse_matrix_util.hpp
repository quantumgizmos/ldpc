#ifndef SPARSE_MATRIX_UTIL_H
#define SPARSE_MATRIX_UTIL_H

#include <iostream>
#include <vector>
#include <memory>
#include <iterator>
#include "sparse_matrix.hpp"
#include <sstream>      
#include <string>

using namespace std;
using namespace sparse_matrix;

template <class SPARSE_MATRIX_CLASS>
stringstream print_sparse_matrix(SPARSE_MATRIX_CLASS& matrix, bool SILENT = false){
    stringstream ss;
    int m = matrix.m;
    int n = matrix.n;
    for(int j=0; j<m;j++){
        for(int i=0; i<n;i++){
        //     // cout<<j<<" "<<i<<endl;
            auto e = matrix.get_entry(j,i);
            
            if(e->at_end()) {
                ss << unsigned(0);
            } 
            else {
                if(is_same<decltype(e->value), uint8_t>::value){
                    ss << unsigned(e->value);
                }
                else{
                    ss << e->value;
                }
                // cout<<e->row_index<<" "<<e->col_index<<" "<<unsigned(e->value)<<endl;
            }
            
            if(i!=(n-1)&& !is_same<decltype(e->value), uint8_t>::value) ss << " ";
        }
        if(j!=(m-1)) ss << "\n";
    }
    if(!SILENT) cout<<ss.str()<<endl;
    return ss;
}




template <class SPARSE_MATRIX_CLASS>
SPARSE_MATRIX_CLASS *copy_cols(SPARSE_MATRIX_CLASS *mat, vector<int> cols){
    int m,n,i,j;
    m = mat->m;
    n = cols.size();
    SPARSE_MATRIX_CLASS * copy_mat = new SPARSE_MATRIX_CLASS(m,n);
    int new_col_index=-1;
    for(auto col_index: cols){
        new_col_index+=1;
        for(auto e: mat->iterate_column(col_index)){
            copy_mat->insert_entry(e->row_index,new_col_index,e->value);
        }
    }
    return copy_mat;
}


template<class T>
void print_vector(const T& input){
    int length = input.size();
    cout<<"[";
    for(int i = 0; i<length; i++){
        if(is_same<T, vector<uint8_t>>::value){
            cout<<unsigned(input[i]);
            }
        else cout<<input[i];
        if(i!=(length-1)) cout<<" ";
    }
    cout<<"]"<<endl;
}


template <class T>
void print_array(T array, int length){
    for(int i=0;i<length;i++){
        if(is_same<T, uint8_t*>::value){
            cout<<unsigned(array[i])<<" ";
            }
        else cout<<array[i]<<" ";
    }
    cout<<endl;
}


#endif