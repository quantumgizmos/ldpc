#include <gtest/gtest.h>
#include "bp.hpp"
#include "gf2sparse.hpp"
#include "gf2sparse_linalg.hpp"
#include "sparse_matrix_util.hpp"
#include "osd.hpp"
#include "gf2codes.hpp"
#include "sparse_matrix.hpp"

// TEST(kernel, hamming_code) {

//     for(int i=2; i<8;i++){
//         auto pcm1 = gf2codes::hamming_code(i);
//         auto pcm2 = gf2codes::hamming_code(i);

//         auto mats = vector<decltype(pcm1)>{pcm1,pcm2};
//         auto pcm = gf2sparse::vstack(mats);

//         auto kernel = gf2sparse_linalg::kernel(pcm);
//         // print_sparse_matrix(*kernel);
//         auto kernelT = kernel->transpose();
//         auto mm = pcm->matmul(kernelT);

//         // print_sparse_matrix(*mm);

//         ASSERT_EQ(mm->entry_count(),0);
    
//     }
// }

TEST(rank, hamming_code_test) {

    for(int i=2; i<8;i++){
        auto pcm1 = gf2codes::hamming_code(i);
        auto pcm2 = gf2codes::hamming_code(i);
        auto pcm0 = gf2sparse::GF2Sparse<bp::BpEntry>::New(pcm1->m,pcm1->n);

        auto mats = vector<decltype(pcm1)>{pcm1,pcm0,pcm2};
        auto pcm = gf2sparse::vstack(mats);

        int rank = gf2sparse_linalg::rank(pcm);

        ASSERT_EQ(rank,i);
    
    }
}







int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}