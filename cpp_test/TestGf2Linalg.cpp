#include <gtest/gtest.h>
#include "bp.hpp"
#include "gf2sparse.hpp"
#include "gf2sparse_linalg.hpp"
#include "sparse_matrix_util.hpp"
#include "osd.hpp"
#include "gf2codes.hpp"
#include "sparse_matrix.hpp"

TEST(kernel, hamming_code) {

    for(int i=2; i<8;i++){
        auto pcm1 = gf2codes::hamming_code(i);
        auto pcm2 = gf2codes::hamming_code(i);

        auto mats = vector<decltype(pcm1)>{pcm1,pcm2};
        auto pcm = gf2sparse::vstack(mats);

        auto kernel = gf2sparse_linalg::kernel(pcm);
        // print_sparse_matrix(*kernel);
        auto kernelT = kernel->transpose();
        auto mm = pcm->matmul(kernelT);

        // print_sparse_matrix(*mm);

        ASSERT_EQ(mm->entry_count(),0);
    
    }
}







int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}