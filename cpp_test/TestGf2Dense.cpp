#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <iostream>
#include <vector>

#include "sparse_matrix_util.hpp"
#include "gf2sparse.hpp"
#include "gf2sparse_linalg.hpp"
#include "gf2dense.hpp"
#include "gf2codes.hpp"
#include "bp.hpp"

using namespace std;
using namespace ldpc::gf2sparse;
using namespace ldpc::gf2sparse_linalg;
using namespace ldpc::sparse_matrix_util;
using namespace ldpc::gf2codes;

std::vector<std::vector<int>> random_csr_matrix(int m, int n, float sparsity = 0.5, unsigned int seed = 42) {
    std::vector<std::vector<int>> csr_matrix;
    
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0, 1);

    // Initialize row_ptr with 0
    std::vector<int> row_ptr = {0};

    for (int i = 0; i < m; ++i) {
        std::vector<int> row_indices;
        int nnz = 0;  // Number of non-zero elements in this row
        for (int j = 0; j < n; ++j) {
            float value = dis(gen);
            if (value < sparsity) {
                row_indices.push_back(j);
                ++nnz;
            }
        }
        csr_matrix.push_back(row_indices);  // Store the column indices of non-zero elements
        row_ptr.push_back(row_ptr.back() + nnz);  // Update row_ptr
    }
    
    // Optionally, you can also return the row_ptr
    // For this example, I'm only returning the csr_matrix containing column indices of non-zero elements
    return csr_matrix;
}

double sparsity = 0.01;
int check_count = 400;
int bit_count = 400;

TEST(gf2dense, sparse_kernel_rnd_binary_matrix){
    int m, n;
    m=check_count;
    n=bit_count;

    auto rnd_csr = random_csr_matrix(m,n,sparsity);

    auto rnd_mat = GF2Sparse<>(m,n);

    rnd_mat.csr_insert(rnd_csr);

    // print_sparse_matrix(rnd_mat);

    auto ker = ldpc::gf2dense::sparse_kernel(m,n,rnd_csr);



    auto ker_mat = GF2Sparse<>(m,n);

    for(int i=0; i<n; i++){
        for(int row : ker[i]){
            ker_mat.insert_entry(row,i);
        }
    }

    auto kerT = ker_mat.transpose();

    auto ker0 = rnd_mat.matmul(kerT);

    auto expected = GF2Sparse<>(m,n);

    ASSERT_TRUE(ker0==expected);

}

TEST(gf2dense, kernel_rnd_binary_matrix){
    int m, n;
    m=check_count;
    n=bit_count;

    auto rnd_csr = random_csr_matrix(m,n,sparsity);

    auto rnd_mat = GF2Sparse<>(m,n);

    rnd_mat.csr_insert(rnd_csr);

    // print_sparse_matrix(rnd_mat);

    auto ker = ldpc::gf2dense::kernel(m,n,rnd_csr);



    auto ker_mat = GF2Sparse<>(m,n);

    for(int i=0; i<n; i++){
        for(int row : ker[i]){
            ker_mat.insert_entry(row,i);
        }
    }

    auto kerT = ker_mat.transpose();

    auto ker0 = rnd_mat.matmul(kerT);

    auto expected = GF2Sparse<>(m,n);

    ASSERT_TRUE(ker0==expected);

}


TEST(gf2dense, rank) {

    for(int i=3; i<10;i++){
        auto pcm1 = hamming_code(i);
        auto pcm2 = hamming_code(i);
        auto pcm0 = ldpc::bp::BpSparse(pcm1.m,pcm1.n);
        auto mats = vector<decltype(pcm1)>{pcm0,pcm1,pcm1};
        auto pcm = ldpc::gf2sparse::vstack(mats);

        std::vector<std::vector<int>> mat_csc;
        for(int col = 0; col<pcm.n; col++){
            mat_csc.push_back(std::vector<int>{});
            for(auto e: pcm.iterate_column(col)){
                mat_csc[col].push_back(e.row_index);
            }
        }

        int r = ldpc::gf2dense::rank(pcm.m, pcm.n, mat_csc);

        ASSERT_EQ(r,i);

    
    }
}

TEST(gf2dense, rep_code_test){

    auto pcm1 = ring_code(3);


    auto pcm0 = ldpc::bp::BpSparse(3,3);
    auto mats = vector<decltype(pcm0)>{pcm0,pcm1};
    auto pcm = ldpc::gf2sparse::vstack(mats);


    std::vector<std::vector<int>> mat_csc;
    for(int col = 0; col<pcm.n; col++){
        mat_csc.push_back(std::vector<int>{});
        for(auto e: pcm.iterate_column(col)){
            mat_csc[col].push_back(e.row_index);
        }
    }

    int rank = ldpc::gf2dense::rank(6,3,mat_csc);



    ASSERT_EQ(rank, 2);


}