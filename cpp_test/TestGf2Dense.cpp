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
#include "io.hpp"
#include "rapidcsv.h"


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

    for (auto i = 0; i < std::pow(2, pcm.n); i++) {
        auto error = ldpc::util::decimal_to_binary(i, pcm.n);
        auto synd = pcm.mulvec(error);
        auto x = plu.lu_solve(synd);
        auto x_synd = pcm.mulvec(x);
        ASSERT_EQ(x_synd, synd);
    }

}

TEST(PluDecomposition, hamming_code_not_full_row_rank) {

    auto pcm0 = hamming_code(3);
    auto zero_mat = GF2Sparse<>(pcm0.m, pcm0.n);

    auto pstack_list = std::vector<decltype(pcm0)>{zero_mat, pcm0, pcm0};
    auto pcm = ldpc::gf2sparse::vstack(pstack_list);

    // ldpc::sparse_matrix_util::print_sparse_matrix(pcm);

    auto pcm_csc = pcm.col_adjacency_list();
    auto plu = ldpc::gf2dense::PluDecomposition(pcm.m, pcm.n, pcm_csc);

    plu.rref();

    for (auto i = 0; i < std::pow(2, pcm.n); i++) {
        auto error = ldpc::util::decimal_to_binary(i, pcm.n);
        auto synd = pcm.mulvec(error);
        auto x = plu.lu_solve(synd);
        auto x_synd = pcm.mulvec(x);
        ASSERT_EQ(x_synd, synd);
    }

}

TEST(PluDecomposition, hamming_code_not_full_col_rank) {

    auto pcm0 = hamming_code(3);
    auto zero_mat = GF2Sparse<>(pcm0.m, pcm0.n);

    auto pstack_list = std::vector<decltype(pcm0)>{zero_mat, pcm0, pcm0};
    auto pcm = ldpc::gf2sparse::hstack(pstack_list);

    auto pcm_csc = pcm.col_adjacency_list();
    auto plu = ldpc::gf2dense::PluDecomposition(pcm.m, pcm.n, pcm_csc);

    plu.rref();

    for (auto i = 0; i < std::pow(2, pcm.m); i++) {
        auto synd = ldpc::util::decimal_to_binary(i, pcm.m);
        auto x = plu.lu_solve(synd);
        auto x_synd = pcm.mulvec(x);
        ASSERT_EQ(x_synd, synd);
    }

}

TEST(PluDecomposition, hamming_code_not_full_col_rank5) {

    auto pcm0 = hamming_code(5);
    auto zero_mat = GF2Sparse<>(pcm0.m, pcm0.n);

    auto pstack_list = std::vector<decltype(pcm0)>{zero_mat, pcm0, pcm0};
    auto pcm = ldpc::gf2sparse::hstack(pstack_list);

    auto pcm_csc = pcm.col_adjacency_list();
    auto plu = ldpc::gf2dense::PluDecomposition(pcm.m, pcm.n, pcm_csc);

    plu.rref();

    for (auto i = 0; i < std::pow(2, pcm.m); i++) {
        auto synd = ldpc::util::decimal_to_binary(i, pcm.m);
        auto x = plu.lu_solve(synd);
        auto x_synd = pcm.mulvec(x);
        // ldpc::sparse_matrix_util::print_vector(x);
        // ldpc::sparse_matrix_util::print_vector(synd);
        // ldpc::sparse_matrix_util::print_vector(x_synd);
        // std::cout<<std::endl;
        ASSERT_EQ(x_synd, synd);
    }

}

TEST(PluDecomposition, ring_code_left_padding) {

    auto pcm0 = ring_code(3);
    auto zero_mat = GF2Sparse<>(pcm0.m, 5);

    auto pstack_list = std::vector<decltype(pcm0)>{zero_mat, pcm0};
    auto pcm = ldpc::gf2sparse::hstack(pstack_list);

    auto pcm_csc = pcm.col_adjacency_list();
    auto plu = ldpc::gf2dense::PluDecomposition(pcm.m, pcm.n, pcm_csc);

    plu.rref();

    for (auto i = 0; i < std::pow(2, pcm.n); i++) {
        auto error = ldpc::util::decimal_to_binary(i, pcm.n);
        auto synd = pcm.mulvec(error);
        auto x = plu.lu_solve(synd);
        auto x_synd = pcm.mulvec(x);
        // ldpc::sparse_matrix_util::print_vector(synd);
        // ldpc::sparse_matrix_util::print_vector(x_synd);
        // std::cout<<std::endl;
        ASSERT_EQ(x_synd, synd);
    }

}

TEST(PluDecomposition, ring_code_top_padding) {

    auto pcm0 = ring_code(3);
    auto zero_mat = GF2Sparse<>(5, pcm0.n);

    auto pstack_list = std::vector<decltype(pcm0)>{zero_mat, pcm0, pcm0};
    auto pcm = ldpc::gf2sparse::vstack(pstack_list);

    auto pcm_csc = pcm.col_adjacency_list();
    auto plu = ldpc::gf2dense::PluDecomposition(pcm.m, pcm.n, pcm_csc);

    plu.rref();

    for (auto i = 0; i < std::pow(2, pcm.n); i++) {
        auto error = ldpc::util::decimal_to_binary(i, pcm.n);
        auto synd = pcm.mulvec(error);
        auto x = plu.lu_solve(synd);
        auto x_synd = pcm.mulvec(x);
        // ldpc::sparse_matrix_util::print_vector(synd);
        // ldpc::sparse_matrix_util::print_vector(x_synd);
        // std::cout<<std::endl;
        ASSERT_EQ(x_synd, synd);
    }

}


TEST(GF2Sparse, lu_solve_batch) {

    auto csv_path = ldpc::io::getFullPath("cpp_test/test_inputs/gf2_lu_solve_test.csv");
    rapidcsv::Document doc(csv_path, rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(';'));

    int row_count = doc.GetColumn<string>(0).size();

    for (int i = 0; i < row_count; i++) {

        // std::cout<<i<<std::endl;

        std::vector<string> row = doc.GetRow<string>(i);
        int m = stoi(row[0]);
        int n = stoi(row[1]);
        auto input_csr_vector = ldpc::io::string_to_csr_vector(row[2]);
        auto synd = ldpc::io::binaryStringToVector(row[3]);

        ASSERT_EQ(synd.size(), m);

        auto pcm = GF2Sparse<>(m, n);
        pcm.csr_insert(input_csr_vector);

        auto pcm_csc = pcm.col_adjacency_list();
        auto plu = ldpc::gf2dense::PluDecomposition(pcm.m, pcm.n, pcm_csc);
        plu.rref(true, true);
        auto x = plu.lu_solve(synd);

        auto x_synd = pcm.mulvec(x);
        ASSERT_EQ(x_synd, synd);

    }

}


TEST(GF2Sparse, fast_lu_solve_batch) {

    auto csv_path = ldpc::io::getFullPath("cpp_test/test_inputs/gf2_lu_solve_test.csv");
    rapidcsv::Document doc(csv_path, rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(';'));

    int row_count = doc.GetColumn<string>(0).size();

    for (int i = 0; i < row_count; i++) {

        // std::cout<<i<<std::endl;

        std::vector<string> row = doc.GetRow<string>(i);
        int m = stoi(row[0]);
        int n = stoi(row[1]);
        auto input_csr_vector = ldpc::io::string_to_csr_vector(row[2]);
        auto synd = ldpc::io::binaryStringToVector(row[3]);
        ASSERT_EQ(synd.size(), m);

        auto pcm = GF2Sparse<>(m, n);
        pcm.csr_insert(input_csr_vector);
        auto pcm_csc = pcm.col_adjacency_list();
        // ldpc::sparse_matrix_util::print_sparse_matrix(pcm);
        // ldpc::sparse_matrix_util::print_vector(synd);


        auto plu = ldpc::gf2dense::PluDecomposition(pcm.m, pcm.n, pcm_csc);
        auto x = plu.fast_lu_solve(synd);


        auto x_synd = pcm.mulvec(x);
        ASSERT_EQ(x_synd, synd);

    }

}

TEST(PluDecomposition, fast_solve_ring_code) {

    for (int j = 3; j < 10; j++) {

        auto pcm = ring_code(j);
        auto pcm_csc = pcm.col_adjacency_list();
        auto plu = ldpc::gf2dense::PluDecomposition(pcm.m, pcm.n, pcm_csc);

        for (auto i = 0; i < std::pow(2, pcm.n); i++) {
            auto error = ldpc::util::decimal_to_binary(i, pcm.n);
            auto synd = pcm.mulvec(error);
            auto x = plu.fast_lu_solve(synd);
            auto x_synd = pcm.mulvec(x);


            ASSERT_EQ(x_synd, synd);
        }

    }

}

TEST(PluDecomposition, ring_code_test7) {

    std::vector<std::vector<int>> matrix = {
            {1, 0, 1, 0, 0, 0, 0},
            {1, 1, 0, 0, 0, 0, 0},
            {0, 1, 0, 0, 1, 0, 0},
            {0, 0, 1, 1, 0, 0, 0},
            {0, 0, 0, 1, 0, 0, 1},
            {0, 0, 0, 0, 1, 1, 0},
            {0, 0, 0, 0, 0, 1, 1}
    };

    auto pcm_csr = std::vector<std::vector<int>>{7, std::vector<int>{}};
    for (auto i = 0; i < 7; i++) {
        for (auto j = 0; j < 7; j++) {
            if (matrix[i][j] == 1) {
                pcm_csr[i].push_back(j);
            }
        }
    }

    // ldpc::gf2dense::print_csr(pcm_csr);
    auto pcm_csc = ldpc::gf2dense::csc_to_csr(pcm_csr);
    // ldpc::gf2dense::print_csc(pcm_csc);

    auto plu = ldpc::gf2dense::PluDecomposition(7, 7, pcm_csc);

    auto synd = std::vector<uint8_t>{1, 1, 0, 1, 1, 1, 1};
    auto x = plu.fast_lu_solve(synd);

    auto pcm = ldpc::gf2sparse::csc_to_gf2sparse(pcm_csc);

    auto x_synd = pcm.mulvec(x);

    ldpc::gf2dense::print_csr(plu.L);
    std::cout << std::endl;
    ldpc::gf2dense::print_csr(plu.U);

    ASSERT_EQ(x_synd, synd);


}
/**
 * Tests the merge of two plu decompositions
 * Ring code is used as simple exmaple here.
 */
TEST(PluDecomposition, merge_with_other) {
    //setup first plu decomposition
    std::vector<std::vector<int>> matrix = {
            {1, 1, 0},
            {0, 1, 1},
            {0, 0, 1},
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0},
//            {1, 1, 0, 0, 0, 0, 0},
//            {0, 1, 1, 0, 0, 0, 0},
//            {0, 0, 1, 1, 0, 0, 0},
//            {0, 0, 0, 1, 1, 0, 0},
//            {0, 0, 0, 0, 1, 1, 0},
//            {0, 0, 0, 0, 0, 1, 1},
    };

    std::vector<std::vector<int>> matrix2 = {
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0},
            {1, 0, 0},
            {1, 1, 0},
            {0, 1, 1},
    };

    auto pcm_csr1 = std::vector<std::vector<int>>{6, std::vector<int>{}};
    for (auto i = 0; i < matrix.size(); i++) {
        for (auto j = 0; j < matrix.at(i).size(); j++) {
            if (matrix[i][j] == 1) {
                pcm_csr1[i].push_back(j);
            }
        }
    }

    auto pcm_csr2 = std::vector<std::vector<int>>{6, std::vector<int>{}};
    for (auto i = 0; i < matrix2.size(); i++) {
        for (auto j = 0; j < matrix2.at(i).size(); j++) {
            if (matrix2[i][j] == 1) {
                pcm_csr2[i].push_back(j);
            }
        }
    }

    // ldpc::gf2dense::print_csr(pcm_csr);
    auto pcm_csc1 = ldpc::gf2dense::csc_to_csr(pcm_csr1);
    auto plu1 = ldpc::gf2dense::PluDecomposition(matrix.size(), matrix.at(0).size(), pcm_csc1);
    auto syndr1 = std::vector<uint8_t>{1, 0, 0, 0, 0};
    plu1.rref_with_y_image_check(syndr1);
    plu1.print();
    auto plu1_nrcols_before = plu1.col_count;
    auto plu1_piv_cols_before = plu1.pivot_cols;

    auto pcm_csc2 = ldpc::gf2dense::csc_to_csr(pcm_csr2);
    auto plu2 = ldpc::gf2dense::PluDecomposition(matrix2.size(), matrix2.at(0).size(), pcm_csc2);
    auto syndr2 = std::vector<uint8_t>{0, 0, 0, 1, 0};
    plu2.rref_with_y_image_check(syndr1);
    plu2.print();

    plu1.merge_with_decomposition(plu2, 3);
    plu1.print();
    ASSERT_TRUE(plu1.row_count == plu1.rows.size());
    ASSERT_TRUE(plu1.row_count == 6);

    ASSERT_TRUE(plu1.L.size() == plu1.row_count);
    ASSERT_TRUE(plu1.U.size() == plu1.matrix_rank);

    ASSERT_TRUE(plu1.col_count == plu1_nrcols_before + plu2.col_count);
    ASSERT_TRUE(plu1.pivot_cols.size() == plu1_piv_cols_before.size() + plu2.pivot_cols.size());
    auto pivot_cols = plu1_piv_cols_before;
    pivot_cols.insert(pivot_cols.end(), plu2.pivot_cols.begin(),
                                                  plu2.pivot_cols.end());
    ASSERT_TRUE(plu1.pivot_cols == pivot_cols);
}