#include <gtest/gtest.h>
#include "sparse_matrix_util.hpp"
#include "gf2sparse.hpp"
#include "gf2sparse_linalg.hpp"
#include "gf2codes.hpp"
#include <cmath>

// #include "sparse_matrix.hpp"

TEST(kernel, hamming_code_test) {

    for(int i=2; i<8;i++){

        auto pcm = gf2codes::hamming_code(i);
        auto ker = gf2sparse_linalg::kernel(pcm);
        print_sparse_matrix(ker);
        auto kerT = ker.transpose();
        auto ker0 = pcm.matmul(kerT);
        ASSERT_EQ(ker0.entry_count(),0);
    
    }
}

TEST(rank, hamming_code_test) {

    for(int i=2; i<6;i++){
        auto pcm1 = gf2codes::hamming_code(i);
        auto pcm2 = gf2codes::hamming_code(i);
        auto pcm0 = gf2sparse::GF2Sparse<bp::BpEntry>(pcm1.m,pcm1.n);


        auto mats = vector<decltype(pcm1)>{pcm1,pcm0,pcm1};

        auto pcm = gf2sparse::vstack(mats);
            
        int rank = gf2sparse_linalg::rank(pcm1);

        ASSERT_EQ(rank,i);

    
    }
}







int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}