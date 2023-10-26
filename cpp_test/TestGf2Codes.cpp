#include <gtest/gtest.h>
#include "ldpc.hpp"
#include "gf2codes.hpp"

TEST(TestRepCode, init) {

    int n = 10;
    auto pcm = ldpc::gf2codes::rep_code(n);
    ldpc::sparse_matrix_util::print_sparse_matrix(pcm);

    ASSERT_EQ(pcm.n,n);
    ASSERT_EQ(pcm.m,n-1);

    for(int i = 0; i<n-1; i++){
        auto& e = pcm.get_entry(i,i);
        ASSERT_EQ(e.at_end(),false);
    }
    for(int i = 0; i<n-1; i++){
        auto& e = pcm.get_entry(i,i+1);
        ASSERT_EQ(e.at_end(),false);
    }
    
}

TEST(TestRingCode, init) {

    int n = 10;
    auto pcm = ldpc::gf2codes::ring_code(n);
    ldpc::sparse_matrix_util::print_sparse_matrix(pcm);

    ASSERT_EQ(pcm.n,n);
    ASSERT_EQ(pcm.m,n);
    for(int i = 0; i<n; i++){
        auto& e = pcm.get_entry(i,i);
        ASSERT_EQ(e.at_end(),false);
    }
    for(int i = 0; i<n; i++){
        auto& e = pcm.get_entry(i,(i+1)%n);
        ASSERT_EQ(e.at_end(),false);
    }
    
}

// Utility function to count the number of set bits (1s) in an integer
int count_set_bits(int n) {
    int count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}

TEST(TestHammingCode, init) {
    int r = 4;
    int n = (1 << r) - 1;
    int k = n - r;

    auto pcm = ldpc::gf2codes::hamming_code(r);
    ldpc::sparse_matrix_util::print_sparse_matrix(pcm);

    ASSERT_EQ(pcm.n, n);
    ASSERT_EQ(pcm.m, r);

    for (int i = 0; i < n; i++) {
        int row_sum = 0;
        for (int j = 0; j < r; j++) {
            auto e = pcm.get_entry(j, i);
            if (!e.at_end()) {
                row_sum++;
            }
        }
        ASSERT_EQ(row_sum, count_set_bits(i + 1));
    }
}






int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}