#ifndef GF2DENSE_H
#define GF2DENSE_H

#include <vector>
#include <iterator>
#include "gf2sparse.hpp"
#include "sparse_matrix_util.hpp"

namespace ldpc{
namespace gf2dense{

template<class T>
int vector_find(std::vector<T> vec, T value){
    int index = 0;
    for(auto val: vec){
        if(val == value) return index;
        index++;
    }

    return -1;

}

enum PluMethod{
    SPARSE_ELIMINATION = 0,
    DENSE_ELIMINATION = 1
};


std::vector<int> NULL_INT_VECTOR = {};
typedef std::vector<std::vector<int>> CscMatrix;
typedef std::vector<std::vector<int>> CsrMatrix;


void print_csc(int row_count, int col_count, std::vector<std::vector<int>> csc_mat){

    CsrMatrix csr_matrix;
    for(int i = 0; i<row_count; i++){
        csr_matrix.push_back(std::vector<int>{});
    }

    int col_index = 0;
    for(auto col: csc_mat){
        for(auto entry: col){
            csr_matrix[entry].push_back(col_index);
        }
        col_index++;
    }

    for(auto row: csr_matrix){
        auto row_dense = std::vector<int>(col_count,0);
        for(auto entry: row){
            row_dense[entry] = 1;
        }
        ldpc::sparse_matrix_util::print_vector(row_dense);
    }

}

class PluDecomposition {
private:
    CscMatrix& csc_mat;

public:
    CscMatrix L;
    CscMatrix U;
    CscMatrix P;
    int matrix_rank;
    int row_count;
    int col_count;
    std::vector<int> rows;
    std::vector<int> swap_rows;
    std::vector<int> pivot_cols;
    std::vector<int> not_pivot_cols;
    
    PluDecomposition(int row_count, int col_count, std::vector<std::vector<int>>& csc_mat)
        : row_count(row_count), col_count(col_count), csc_mat(csc_mat) {}
    
    ~PluDecomposition() = default;

    void reset(){
        this->matrix_rank = 0;
        this->rows.clear();
        this->swap_rows.clear();
        this->pivot_cols.clear();
        this->not_pivot_cols.clear();
    
        for(auto& col: this->L){
            col.clear();
        }
        this->L.clear();

        for(auto& col: this->U){
            col.clear();
        }
        this->U.clear();

        for(auto& col: this->P){
            col.clear();
        }
        this->P.clear();

    }


    void rref(bool construct_U = true){
    
        this->reset();

        for(int i = 0; i<this->row_count; i++){
            this->rows.push_back(i);
        }

        std::vector<uint8_t> rr_col;
        rr_col.resize(this->row_count,0);
     


        int max_rank = std::min(this->row_count, this->col_count);

        for(int col = 0; col<this->col_count; col++){

     
            std::fill(rr_col.begin(),rr_col.end(),0);
            for(int row_index: this->csc_mat[col]){
                rr_col[row_index] = 1;
            }
            

            for(int i = 0; i<this->matrix_rank; i++){

                std::swap(rr_col[i],rr_col[this->swap_rows[i]]);
                if(rr_col[i] == 1){
                    for(auto it = this->L[i].begin() + 1; it != this->L[i].end(); ++it){
                        int add_row = *it;
                        rr_col[add_row] ^= 1;
                    }

                }
            
            }

            bool PIVOT_FOUND = false;

            for(int i = this->matrix_rank; i<this->row_count; i++){
                if(rr_col[i] == 1){
                    PIVOT_FOUND = true;
                    this->swap_rows.push_back(i);
                    this->pivot_cols.push_back(col);
                    break;
                }
            }

            if(!PIVOT_FOUND){
                this->not_pivot_cols.push_back(col);
                continue;
            }

            std::swap(rr_col[this->matrix_rank],rr_col[this->swap_rows[this->matrix_rank]]);
            std::swap(this->rows[this->matrix_rank],this->rows[this->swap_rows[this->matrix_rank]]);

            this->L.push_back(std::vector<int>{this->matrix_rank});

            for(int i = this->matrix_rank+1; i<this->row_count; i++){
                if(rr_col[i] == 1){
                    // rr_col[i] ^= 1; //we don't actually need to eliminate here.
                    this->L[this->matrix_rank].push_back(i);
                }
            }

            this->matrix_rank++;

            if(construct_U){
                this->U.push_back(std::vector<int>{});
                for(int i = 0; i<matrix_rank; i++){
                    if(rr_col[i] == 1){
                        this->U[col].push_back(i);
                    }
                }
            }

            if(this->matrix_rank==max_rank) break;

        }

    }

};

int rank(int row_count, int col_count, CscMatrix& csc_mat){
    auto plu = PluDecomposition(row_count,col_count,csc_mat);
    plu.rref(false);
    return plu.matrix_rank;
}

CscMatrix kernel(int row_count, int col_count, CsrMatrix& csr_mat){

    // To compute the kernel, we need to do PLU decomposition on the transpose of the matrix.
    // The CSR representation of mat is the CSC representation of mat.transpose(). 

    auto plu = PluDecomposition(col_count,row_count,csr_mat);
    plu.rref(false);
    
    std::vector<uint8_t> rr_col(col_count,0);
    std::vector<std::vector<int>> ker;

    for(int j = 0; j<col_count; j++){

        ker.push_back(std::vector<int>{});

        std::fill(rr_col.begin(),rr_col.end(),0);

        rr_col[j] = 1;

        for(int i = 0; i<plu.matrix_rank; i++){

            std::swap(rr_col[i],rr_col[plu.swap_rows[i]]);
            if(rr_col[i] == 1){
                for(auto it = plu.L[i].begin() + 1; it != plu.L[i].end(); ++it){
                    int add_row = *it;
                    rr_col[add_row] ^= 1;
                }

            }
        
        }

        for(int i = plu.matrix_rank; i<col_count; i++){
            if(rr_col[i] == 1){
                ker[j].push_back(i-plu.matrix_rank);
            }
        }
    
    }

    return ker; //returns the kernel as a csc matrix
}

std::vector<int> pivot_rows(int row_count, int col_count, CsrMatrix& csr_mat){
    auto plu = PluDecomposition(col_count,row_count,csr_mat);
    plu.rref(false);
    return plu.pivot_cols;
}

}//end namespace gf2dense
}//end namespace ldpc

#endif