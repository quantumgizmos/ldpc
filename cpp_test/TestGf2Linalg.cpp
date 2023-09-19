#include <gtest/gtest.h>
#include "sparse_matrix_util.hpp"
#include "gf2sparse.hpp"
#include "gf2sparse_linalg.hpp"
#include "gf2codes.hpp"
#include <cmath>

using namespace std;

// #include "sparse_matrix.hpp"

TEST(kernel, hamming_code_test) {

    for(int i=2; i<8;i++){

        auto pcm = gf2codes::hamming_code(i);
        auto ker = gf2sparse_linalg::kernel(pcm);
        // print_sparse_matrix(ker);
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

TEST(row_complement_basis, identity_test){

    for(int i = 0; i<5; i++){
        auto pcm = gf2sparse::identity(5);
        pcm.remove_entry(i,i);
        auto complement = gf2sparse_linalg::row_complement_basis(pcm);
        auto expected = gf2sparse::GF2Sparse<>(1,5);
        expected.insert_entry(0,i);
        ASSERT_TRUE(complement==expected);    
    }

    // cout<<endl;

    auto pcm = gf2sparse::GF2Sparse<>(1,4);
    for(int i = 0; i<4; i++) pcm.insert_entry(0,i);

    // print_sparse_matrix(pcm);

    auto complement = gf2sparse_linalg::row_complement_basis(pcm);  

    print_sparse_matrix(complement);

}







int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}