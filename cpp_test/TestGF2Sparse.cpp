#include <gtest/gtest.h>
#include "gf2sparse.hpp"
#include "sparse_matrix_util.hpp"
#include <set>

bool TEST_WITH_CSR(GF2Sparse<GF2Entry> matrix, vector<vector<int>>& csr_matrix){

    int i = 0;
    for(vector<int> row: csr_matrix){
        int j = 0;
        set<int> row_set;
        for(auto col : row) row_set.insert(col);
        for(auto e: matrix.iterate_row(i)){

            if(!e->value==1) return false;
            if(!e->row_index==i) return false;
            if(row_set.contains(e->col_index)){
                row_set.erase(e->col_index); //this ensures there a no duplicates.
            }
            else return false;

            j++;
        }
    
        if(row_set.size()!=0) return false; //this checks that there no entries in the CSR matrix that aren't in the matrix.
        i++;
    }

    return true; 

}


TEST(GF2Sparse, csr_insert){

    auto matrix = GF2Sparse(3,3);
    vector<vector<int>> csr_mat = {{0},{1,2},{2}};
    matrix.csr_insert(csr_mat);
    // print_sparse_matrix(matrix);

    ASSERT_EQ(matrix.get_entry(0,0)->value, 1);
    ASSERT_EQ(matrix.get_entry(1,1)->value, 1);
    ASSERT_EQ(matrix.get_entry(1,2)->value, 1);
    ASSERT_EQ(matrix.get_entry(2,2)->value, 1);
    ASSERT_EQ(matrix.get_entry(2,0)->value, 0);

    //test using the TEST_WITH_CSR function
    ASSERT_EQ(TEST_WITH_CSR(matrix,csr_mat),true);

    vector<vector<int>> csr_mat2 = {{0,4},{1,2},{2}};
    ASSERT_EQ(TEST_WITH_CSR(matrix,csr_mat2),false);

    //check the entry counts
    ASSERT_EQ(matrix.entry_count(),4);

}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}