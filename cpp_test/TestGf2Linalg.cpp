#include <gtest/gtest.h>
#include "sparse_matrix_util.hpp"
#include "gf2sparse.hpp"
#include "gf2sparse_linalg.hpp"
#include <cmath>

#include <random>

using namespace std;
using namespace ldpc::gf2sparse;
using namespace ldpc::gf2sparse_linalg;
using namespace ldpc::sparse_matrix_util;

#include <iostream>
#include <vector>
#include <random>

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

/**
 * Creates the parity check matrix of a repetition code of length n.
 *
 * @tparam T The type of the entries in the sparse matrix. Default is ldpc::gf2sparse::GF2Entry.
 * @param n The length of the repetition code.
 * @return A shared pointer to a GF2Sparse<T> matrix representing the parity check matrix.
 */
template <typename T = GF2Entry>
ldpc::gf2sparse::GF2Sparse<T> rep_code(int n){
    // Create a shared pointer to a new GF2Sparse<T> matrix with n-1 rows and n columns.
    auto pcm = ldpc::gf2sparse::GF2Sparse<T>(n-1, n);
    // Fill in the entries of the matrix corresponding to the repetition code.
    for(int i=0; i<n-1; i++){
        pcm.insert_entry(i, i);    // Insert a 1 in the diagonal position.
        pcm.insert_entry(i, i+1);  // Insert a 1 in the position to the right of the diagonal.
    }
    // Return the shared pointer to the matrix.
    return pcm;
}

/**
 * Creates the parity check matrix of a cyclic repetition code of length n.
 *
 * @tparam T The type of the entries in the sparse matrix. Default is ldpc::gf2sparse::GF2Entry.
 * @param n The length of the cyclic repetition code.
 * @return A shared pointer to a GF2Sparse<T> matrix representing the parity check matrix.
 */
template <typename T = GF2Entry>
ldpc::gf2sparse::GF2Sparse<T> ring_code(int n) {
    // Create a shared pointer to a new GF2Sparse<T> matrix with n-1 rows and n columns.
    auto pcm = ldpc::gf2sparse::GF2Sparse<T>(n, n);
    // Fill in the entries of the matrix corresponding to the cyclic repetition code.
    for (int i = 0; i < n; i++) {
        pcm.insert_entry(i, i);    // Insert a 1 in the diagonal position.
        pcm.insert_entry(i, (i + 1) % n);  // Insert a 1 in the position to the right of the diagonal, with wraparound.
    }
    // Return the shared pointer to the matrix.
    return pcm;
}

/**
 * Creates the parity check matrix of a Hamming code with given rank.
 *
 * @tparam T The type of the entries in the sparse matrix. Default is ldpc::gf2sparse::GF2Entry.
 * @param r The rank of the Hamming code, where the block length is 2^r - 1.
 * @return A shared pointer to a GF2Sparse<T> matrix representing the parity check matrix.
 */
template <typename T = GF2Entry>
ldpc::gf2sparse::GF2Sparse<T> hamming_code(int r) {
    // Calculate the block length and the number of data bits.
    int n = (1 << r) - 1;
    int k = n - r;

    // Create a shared pointer to a new GF2Sparse<T> matrix with r rows and n columns.
    auto pcm = ldpc::gf2sparse::GF2Sparse<T>(r, n);

    // Fill in the entries of the matrix corresponding to the Hamming code.
    for (int i = 0; i < n; i++) {
        int binary = i + 1;
        for (int j = 0; j < r; j++) {
            if (binary & (1 << j)) {
                pcm.insert_entry(j, i);
            }
        }
    }

    // Return the shared pointer to the matrix.
    return pcm;
}
TEST(kernel, hamming_code_test) {

    for(int i=2; i<8;i++){

        auto pcm = hamming_code(i);
        auto ker = ldpc::gf2sparse_linalg::kernel(pcm);
        // print_sparse_matrix(ker);
        auto kerT = ker.transpose();
        auto ker0 = pcm.matmul(kerT);
        ASSERT_EQ(ker0.entry_count(),0);
    
    }
}

TEST(rank, hamming_code_test) {

    for(int i=2; i<10;i++){
        auto pcm1 = hamming_code(i);
        auto pcm2 = hamming_code(i);
        auto pcm0 = ldpc::gf2sparse::GF2Sparse<ldpc::gf2sparse::GF2Entry>(pcm1.m,pcm1.n);
        auto mats = vector<decltype(pcm1)>{pcm0,pcm1,pcm1};
        auto pcm = ldpc::gf2sparse::vstack(mats);
            
        int rank = ldpc::gf2sparse_linalg::rank(pcm);

        ASSERT_EQ(rank,i);

    
    }
}

TEST(row_complement_basis, identity_test){

    for(int i = 0; i<5; i++){
        auto pcm = ldpc::gf2sparse::identity(5);
        pcm.remove_entry(i,i);
        auto complement = ldpc::gf2sparse_linalg::row_complement_basis(pcm);
        auto expected = ldpc::gf2sparse::GF2Sparse<>(1,5);
        expected.insert_entry(0,i);
        ASSERT_TRUE(complement==expected);    
    }

    // cout<<endl;

    auto pcm = ldpc::gf2sparse::GF2Sparse<>(1,4);
    for(int i = 0; i<4; i++) pcm.insert_entry(0,i);

    // print_sparse_matrix(pcm);

    auto complement = ldpc::gf2sparse_linalg::row_complement_basis(pcm);  

    // print_sparse_matrix(complement);

}


TEST(pivot_cols, hamming_code_test) {

    for(int i=3; i<10;i++){
        auto pcm1 = hamming_code(i);
        auto pcm2 = hamming_code(i);
        auto pcm0 = ldpc::gf2sparse::GF2Sparse<ldpc::gf2sparse::GF2Entry>(pcm1.m,pcm1.n);


        auto mats = vector<decltype(pcm1)>{pcm0,pcm1,pcm1};

        auto pcm = ldpc::gf2sparse::vstack(mats);
            
        auto pivot_columns = ldpc::gf2sparse_linalg::pivot_columns(pcm);
        // print_vector(pivot_columns);
        ASSERT_EQ(pivot_columns.size(),i);
    }
}


TEST(pivot_rows, hamming_code_test) {

    for(int i=3; i<10;i++){
        auto pcm1 = hamming_code(i);
        auto pcm2 = hamming_code(i);
        auto pcm0 = ldpc::gf2sparse::GF2Sparse<ldpc::gf2sparse::GF2Entry>(pcm1.m,pcm1.n);
        auto mats = vector<decltype(pcm1)>{pcm0,pcm1,pcm1};
        auto pcm = ldpc::gf2sparse::vstack(mats);
            
        auto pivot_rows = ldpc::gf2sparse_linalg::pivot_rows(pcm);
        // print_vector(pivot_rows);
        ASSERT_EQ(pivot_rows.size(),i);
    }
}

TEST(kernel2, hamming_code_test) {

    auto pcm = hamming_code<GF2Entry>(3);

    auto ker = ldpc::gf2sparse_linalg::kernel2(pcm);

    auto ker_mat = GF2Sparse<>(4,7);

    for(int i=0; i<7; i++){
        for(int row : ker[i]){
            ker_mat.insert_entry(row,i);
        }
    }

    auto kerT = ker_mat.transpose();

    auto ker0 = pcm.matmul(kerT);

    // ldpc::sparse_matrix_util::print_sparse_matrix(ker0);

}

TEST(kernel, rnd_binary_matrix){
    int m, n;
    m=1000;
    n=1000;

    auto rnd_csr = random_csr_matrix(m,n,0.01);

    auto rnd_mat = GF2Sparse<>(m,n);

    rnd_mat.csr_insert(rnd_csr);

    // print_sparse_matrix(rnd_mat);

    auto ker = ldpc::gf2sparse_linalg::kernel2(rnd_mat);



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

double sparsity = 0.01;
int check_count = 400;
int bit_count = 400;


TEST(gf2sparse, kernel_sparse){
    int m, n;
    m=check_count;
    n=bit_count;

    auto rnd_csr = random_csr_matrix(m,n,sparsity);

    auto rnd_mat = GF2Sparse<>(m,n);

    rnd_mat.csr_insert(rnd_csr);

    // print_sparse_matrix(rnd_mat);

    auto ker_mat = ldpc::gf2sparse_linalg::kernel(rnd_mat);



    // auto ker_mat = GF2Sparse<>(m,n);

    // for(int i=0; i<n; i++){
    //     for(int row : ker[i]){
    //         ker_mat.insert_entry(row,i);
    //     }
    // }

    auto kerT = ker_mat.transpose();

    auto ker0 = rnd_mat.matmul(kerT);

    auto expected = GF2Sparse<>(m,n);

    // ASSERT_TRUE(ker0==expected);

}



int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}