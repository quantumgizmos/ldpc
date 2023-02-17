#ifndef SPARSE_MATRIX_UTIL_H
#define SPARSE_MATRIX_UTIL_H

#include <iostream>
#include <vector>
#include <memory>
#include <iterator>
#include "sparse_matrix.hpp"

using namespace std;

template <class SPARSE_MATRIX_CLASS>
void print_sparse_matrix(SPARSE_MATRIX_CLASS &matrix){
    int m = matrix.m;
    int n = matrix.n;
    for(int j=0; j<m;j++){
        for(int i=0; i<n;i++){
            auto e = matrix.get_entry(j,i);
                if(e==matrix.row_heads[j]) cout<<unsigned(0);
                else cout<<unsigned(e->value);
            cout<<"  ";
        }
        cout<<endl;
    }
        
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