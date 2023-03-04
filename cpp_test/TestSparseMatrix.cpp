#include <gtest/gtest.h>
#include "sparse_matrix.hpp"

TEST(SparseMatrixTest, SparseMatrix){
    auto pcm = new SparseMatrix<int>(3,4);

    // Test constructor and allocation.
    int m,n;
    m = 3;
    n = 4;
    ASSERT_EQ(pcm->m, m);
    ASSERT_EQ(pcm->n, n);
    ASSERT_EQ(pcm->released_entry_count, m+n);
    ASSERT_EQ(pcm->entry_count(), 0);

    // Test `insert_entry(i,j,value)
    auto e = pcm->insert_entry(1,2,1);
    ASSERT_EQ(e->value,1);
    ASSERT_EQ(e->row_index,1);
    ASSERT_EQ(e->col_index,2);
    ASSERT_EQ(pcm->released_entry_count, m+n+1);

    // Test `get_entry(i,j,value)
    auto g = pcm->get_entry(1,2);
    ASSERT_EQ(g,e);
    ASSERT_EQ(g->value,1);
    ASSERT_EQ(g->row_index,1);
    ASSERT_EQ(g->col_index,2);

    //Test `remove_entry(i,j)`
    pcm->remove_entry(1,2);
    auto f = pcm->get_entry(1,2);
    ASSERT_NE(f,g);
    /*the removed entry is stored in the `removed_entries buffer`. The total
    number of released entries should therefore remain the same. */
    ASSERT_EQ(pcm->released_entry_count, m+n+1);
    ASSERT_EQ(pcm->entry_count(),0);

    /*Test allocation buffer New entries should preferentially come from the
    `removed entries buffer`. We therefore expect the number of release 
    entries to the stay the same below:  */

    pcm->insert_entry(2,2,1);
    ASSERT_EQ(pcm->entry_count(),1);
    ASSERT_EQ(pcm->released_entry_count,m+n+1);
    
    delete pcm;
    
};

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}