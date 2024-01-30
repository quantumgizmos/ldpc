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

CscMatrix hamming_code_csc(const int d) {
    auto pcm = hamming_code(7);
    CscMatrix pcm_csc;
    for (auto i = 0; i < pcm.m; i++) {
        auto col = std::vector<int>();
        for (auto e: pcm.iterate_column(i)) {
            col.push_back(e.row_index);
        }
        pcm_csc.push_back(col);
    }
    return pcm_csc;
}

TEST(gf2dense, hamming_code){

    auto hamming_pcm = hamming_code(3);

    int m = hamming_pcm.m;
    int n = hamming_pcm.n;

    vector<vector<int>> rnd_csr;
    auto rnd_mat = GF2Sparse<>(m,n);


    for(int i = 0; i<hamming_pcm.m; i++){
        rnd_csr.push_back(vector<int>{});
        for(auto e: hamming_pcm.iterate_row(i)){
            rnd_csr[i].push_back(e.col_index);
            rnd_mat.insert_entry(i,e.col_index);
        }
    }



    // rnd_mat.csr_insert(rnd_csr);

    // print_sparse_matrix(rnd_mat);

    auto ker = ldpc::gf2dense::kernel(m,n,rnd_csr);

    int count_m = -1;
    for(int i=0; i<n; i++){
        for(int row : ker[i]){
            if(row >= count_m){
                count_m = row+1;
            }
        }
    }

    auto ker_mat = GF2Sparse<>(count_m,n);

    for(int i=0; i<n; i++){
        for(int row : ker[i]){
            ker_mat.insert_entry(row,i);
        }
    }

    print_sparse_matrix(ker_mat);

    auto kerT = ker_mat.transpose();

    auto ker0 = rnd_mat.matmul(kerT);

    print_sparse_matrix(ker0);

    auto expected = GF2Sparse<>(m, count_m);

    ASSERT_TRUE(ker0==expected);

}


double sparsity = 0.4;
int check_count = 10;
int bit_count = 10;

TEST(gf2dense, kernel_rnd_binary_matrix){
    int m, n;
    m=check_count;
    n=bit_count;

    auto rnd_csr = random_csr_matrix(m,n,sparsity, 145455);

    auto rnd_mat = GF2Sparse<>(m,n);

    rnd_mat.csr_insert(rnd_csr);

    // print_sparse_matrix(rnd_mat);

    auto ker = ldpc::gf2dense::kernel(m,n,rnd_csr);

    int count_m = -1;
    for(int i=0; i<n; i++){
        for(int row : ker[i]){
            if(row >= count_m){
                count_m = row+1;
            }
        }
    }

    auto ker_mat = GF2Sparse<>(count_m,n);

    for(int i=0; i<n; i++){
        for(int row : ker[i]){
            ker_mat.insert_entry(row,i);
        }
    }

    // print_sparse_matrix(ker_mat);

    auto kerT = ker_mat.transpose();

    auto ker0 = rnd_mat.matmul(kerT);

    // print_sparse_matrix(ker0);

    auto expected = GF2Sparse<>(n, count_m);

    ASSERT_TRUE(ker0==expected);

}


TEST(gf2dense, rank) {

    for(int i=3; i<10;i++){
        auto pcm1 = hamming_code(i);
        auto pcm2 = hamming_code(i);
        auto pcm0 = ldpc::gf2sparse::GF2Sparse(pcm1.m,pcm1.n);
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


    auto pcm0 = ldpc::gf2sparse::GF2Sparse(3,3);
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

TEST(pivot_rows, hamming_code_test) {

    for(int i=3; i<10;i++){
        auto pcm1 = hamming_code(i);
        auto pcm2 = hamming_code(i);
        auto pcm0 = ldpc::gf2sparse::GF2Sparse(pcm1.m,pcm1.n);
        auto mats = vector<decltype(pcm1)>{pcm0,pcm1,pcm1};
        auto pcm = ldpc::gf2sparse::vstack(mats);

        ldpc::gf2dense::CsrMatrix pcm_csr;

        for(int i = 0; i<pcm.m; i++){
            pcm_csr.push_back(vector<int>{});
            for(auto e: pcm.iterate_row(i)){
                pcm_csr[i].push_back(e.col_index);
            }
        }

        auto pivot_rows = ldpc::gf2dense::pivot_rows(pcm.m, pcm.n, pcm_csr);
        // print_vector(pivot_rows);
        ASSERT_EQ(pivot_rows.size(),i);
    }
}

TEST(Gf2DenseTest, RowSpan_SingleRow) {
    int row_count = 1;
    int col_count = 3;
    ldpc::gf2dense::CsrMatrix csr_mat = {{0, 1, 2}};
    auto result = ldpc::gf2dense::row_span(row_count, col_count, csr_mat);
    ldpc::gf2dense::CsrMatrix expected = {{}, {0, 1, 2}};
    ASSERT_EQ(result, expected);
}

TEST(Gf2DenseTest, RowSpan_MultipleRows) {
    int row_count = 2;
    int col_count = 3;
    ldpc::gf2dense::CsrMatrix csr_mat = {{0, 1}, {1, 2}};
    auto result = ldpc::gf2dense::row_span(row_count, col_count, csr_mat);
    ldpc::gf2dense::CsrMatrix expected = {{}, {0, 1}, {1, 2}, {0, 2}};
    ASSERT_EQ(result, expected);
}

TEST(Gf2Dense, estimate_code_distance) {

    auto pcm = hamming_code(7);
    CsrMatrix pcm_csr;
    for (int i = 0; i < pcm.m; i++) {
        pcm_csr.push_back(vector<int>{});
        for (auto e: pcm.iterate_row(i)) {
            pcm_csr[i].push_back(e.col_index);
        }
    }
    auto distance = ldpc::gf2dense::estimate_code_distance(pcm.m, pcm.n, pcm_csr, 0.025, 10);

    ASSERT_EQ(distance.min_distance, 3);

}

TEST(Gf2Dense, test_case_1) {

    // [[0 0 1 0]
    //  [0 0 1 1]
    //  [1 1 0 0]
    //  [0 1 0 0]]

    CscMatrix mat = {{2},
                     {2, 3},
                     {0, 1},
                     {1}};

    int rank = ldpc::gf2dense::rank(4, 4, mat);

    ASSERT_EQ(rank, 4);

}

TEST(Gf2Dense, row_span_rep_code) {

    // [[0 0 1 0]
    //  [0 0 1 1]
    //  [1 1 0 0]
    //  [0 1 0 0]]

    auto pcm = ldpc::gf2codes::rep_code(3);

    std::vector<std::vector<int>> pcm_csr;
    for (int i = 0; i < pcm.m; i++) {
        pcm_csr.push_back(vector<int>{});
        for (auto e: pcm.iterate_row(i)) {
            pcm_csr[i].push_back(e.col_index);
        }
    }

    auto span = ldpc::gf2dense::row_span(pcm.m, pcm.n, pcm_csr);

    cout << span.size() << endl;

    auto span_mat = ldpc::gf2sparse::GF2Sparse(span.size(), pcm.n);
    for (int i = 0; i < span.size(); i++) {
        for (int j: span[i]) {
            // cout<<i<<","<<j<<endl;
            span_mat.insert_entry(i, j);
        }
    }

    ldpc::sparse_matrix_util::print_sparse_matrix(span_mat);

}

TEST(ConvertMatrixTest, CscToCsr) {
    // Create a CscMatrix
    CscMatrix csc_mat = {{0, 1},
                         {2},
                         {1, 2}};

    // Convert to CsrMatrix
    CsrMatrix csr_mat = csc_to_csr(csc_mat);

    // Create the expected CsrMatrix
    CsrMatrix expected_csr_mat = {{0},
                                  {0, 2},
                                  {1, 2}};

    // Check that the converted matrix is as expected
    EXPECT_EQ(csr_mat, expected_csr_mat);
}

TEST(Gf2Dense, exact_code_distance) {

    // [[0 0 1 0]
    //  [0 0 1 1]
    //  [1 1 0 0]
    //  [0 1 0 0]]

    auto pcm = ldpc::gf2codes::hamming_code(3);

    std::vector<std::vector<int>> pcm_csr;
    for (int i = 0; i < pcm.m; i++) {
        pcm_csr.push_back(vector<int>{});
        for (auto e: pcm.iterate_row(i)) {
            pcm_csr[i].push_back(e.col_index);
        }
    }

    int d = ldpc::gf2dense::compute_exact_code_distance(pcm.m, pcm.n, pcm_csr);

    ASSERT_EQ(d, 3);

}

// TEST(Gf2Dense, partial_rref) {
//     auto ham_code = hamming_code_csc(3);

//     auto plu_decomp = PluDecomposition(3,7, ham_code);
//     plu_decomp.rref(true);
// }