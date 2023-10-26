#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <vector>
#include <iterator>
#include <string>
#include "sparse_matrix_base.hpp"

namespace ldpc{
namespace sparse_matrix{

template <class T>
class SparseMatrixEntry: public ldpc::sparse_matrix_base::EntryBase<SparseMatrixEntry<T>> {

    public:
        T value = T(0); // the value structure we are storing at each matrix location. We can define this as any object, and overload operators.
    ~SparseMatrixEntry(){};
    
    std::string str(){
        return std::to_string(this->value);
    }

    SparseMatrixEntry<T> operator+(const SparseMatrixEntry<T>& other);

};

template <class T>
SparseMatrixEntry<T> SparseMatrixEntry<T>::operator+(const SparseMatrixEntry<T>& other){
    SparseMatrixEntry<T> result;
    result.value = this->value + other.value;
    return result;
}

template <class T>
bool operator==(SparseMatrixEntry<T> const &lhs,SparseMatrixEntry<T> const &rhs){
    return lhs.value == rhs.value;
}

template <class T, template<class> class ENTRY_OBJ=SparseMatrixEntry>
class SparseMatrix: public ldpc::sparse_matrix_base::SparseMatrixBase<ENTRY_OBJ<T>> {

private:
    typedef ldpc::sparse_matrix_base::SparseMatrixBase<ENTRY_OBJ<T>> BASE;

public:

    SparseMatrix() = default;

    SparseMatrix(int m, int n, int entry_count = 0): BASE(m,n,entry_count){};

    ENTRY_OBJ<T>& insert_entry(int i, int j, T val = T(1)){
        auto& e = BASE::insert_entry(i,j);
        e.value = val;
        return e;
    }

    void allocate(int m, int n, int entry_count = 0){
        BASE::initialise_sparse_matrix(m,n,entry_count);
    }

    void insert_row(int row_index, std::vector<int>& col_indices, std::vector<T>& values){
        BASE::insert_row(row_index,col_indices);
        int i = 0;
        for(auto& e: this->iterate_row(row_index)){
            e.value = values[i];
            i++; 
        }
    }

    ~SparseMatrix() = default;
};

}//end namespace SparseMatrix
}//end namespace ldpc


#endif