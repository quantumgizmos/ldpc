#include <gtest/gtest.h>
#include "bp.hpp"
#include "gf2sparse.hpp"
#include "gf2sparse_linalg.hpp"
#include "sparse_matrix_util.hpp"
#include "osd.hpp"
#include "gf2codes.hpp"
#include "sparse_matrix.hpp"

TEST(kernel, hamming_code) {

    for(int i=2; i<10;i++){
        auto pcm = gf2codes::hamming_code(i);
        auto kernel = gf2sparse_linalg::kernel(pcm);
        // print_sparse_matrix(*kernel);
        auto kernelT = kernel->transpose();
        auto mm = pcm->matmul(kernelT);

        // print_sparse_matrix(*mm);

        ASSERT_EQ(mm->entry_count(),0);
    
    }
}

// TEST(GF2Sparse,hstack){
//     auto m1 = gf2codes::hamming_code(3);
//     auto m2 = gf2codes::hamming_code(3);

//     auto m3 = gf2sparse::hstack(vector<shared_ptr<sparse_matrix::SparseMatrixBase>>{m1,m2});
//     print_sparse_matrix(*m3);
// }





int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}