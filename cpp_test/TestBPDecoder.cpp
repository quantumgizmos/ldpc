#include <gtest/gtest.h>
// #include "sparse_matrix_util.hpp"
#include "gf2sparse.hpp"
#include "bp.hpp"
#include "sparse_matrix_util.hpp"

TEST(BpEntry, init) {

    auto e = new bp::BpEntry();
    ASSERT_EQ(e->row_index,-100);
    ASSERT_EQ(e->col_index,-100);
    ASSERT_EQ(e->at_end(),true);
    ASSERT_EQ(e->bit_to_check_msg,0.0);
    ASSERT_EQ(e->bit_to_check_msg,0.0);
    delete e;

}

TEST(BpSparse, init){

    int n = 3;
    auto pcm = bp::BpSparse::New(n-1,n);
    for(int i = 0; i<(n-1); i++){
        pcm->insert_entry(i,i);
        pcm->insert_entry(i,(i+1)%n);
    }

    auto expected = "1 1 0\n0 1 1";
    ASSERT_EQ(print_sparse_matrix(*pcm,false).str(), expected);
}






int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}