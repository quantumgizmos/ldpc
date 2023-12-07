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
#include "util.hpp"

using namespace std;
using namespace ldpc::gf2sparse;
using namespace ldpc::gf2sparse_linalg;
using namespace ldpc::sparse_matrix_util;
using namespace ldpc::gf2codes;
using namespace ldpc::gf2dense;

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

TEST(PluDecomposition, hamming_code) {

    auto pcm = hamming_code(3);
    auto pcm_csc = pcm.col_adjacency_list();
    auto plu = ldpc::gf2dense::PluDecomposition(pcm.m, pcm.n, pcm_csc);
    plu.rref();

    for(auto i = 0; i<std::pow(2,pcm.n); i++){
        auto error = ldpc::util::decimal_to_binary(i,pcm.n);
        auto synd = pcm.mulvec(error);
        auto x = plu.lu_solve(synd);
        auto x_synd = pcm.mulvec(x);
        ASSERT_EQ(x_synd, synd);
    }

}

