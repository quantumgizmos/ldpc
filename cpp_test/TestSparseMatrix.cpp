#include <gtest/gtest.h>
#include <vector>
#include "sparse_matrix.hpp"
#include "sparse_matrix_util.hpp"


TEST(TestSparseMatrix, InitialisationAndAllocationStack)
{
    /* Testing initialization, entry insert, entry removal. */

    ldpc::sparse_matrix::SparseMatrix<int> pcm(3, 4); // Stack-allocated SparseMatrix

    // Test constructor and allocation.
    int m = 3;
    int n = 4;
    ASSERT_EQ(pcm.m, m);
    ASSERT_EQ(pcm.n, n);
    ASSERT_EQ(pcm.released_entry_count, m + n);
    ASSERT_EQ(pcm.entry_count(), 0);

    // Test `insert_entry(i,j,value)
    auto& e = pcm.insert_entry(1, 2, 1);
    ASSERT_EQ(e.value, 1);
    ASSERT_EQ(e.row_index, 1);
    ASSERT_EQ(e.col_index, 2);
    ASSERT_EQ(pcm.released_entry_count, m + n + 1);

    // Test `get_entry(i,j,value)
    auto& g = pcm.get_entry(1, 2);
    ASSERT_EQ(&g, &e);
    ASSERT_EQ(g.value, 1);
    ASSERT_EQ(g.row_index, 1);
    ASSERT_EQ(g.col_index, 2);

    // Test `remove_entry(i,j)`
    pcm.remove_entry(1, 2);
    auto& f = pcm.get_entry(1, 2);
    ASSERT_NE(&f, &g);
    /* The removed entry is stored in the `removed_entries buffer`. The total
    number of released entries should therefore remain the same. */
    ASSERT_EQ(pcm.released_entry_count, m + n + 1);
    ASSERT_EQ(pcm.entry_count(), 0);

    /* Test allocation buffer New entries should preferentially come from the
    `removed entries buffer`. We therefore expect the number of release
    entries to stay the same below:  */

    auto& k = pcm.insert_entry(2, 2, 1);
    ASSERT_EQ(pcm.entry_count(), 1);
    ASSERT_EQ(pcm.released_entry_count, m + n + 1);

    /* If we insert an entry that already exists, we use the memory that has
    already been allocated. In the test below, the value of the this entry is
    reassigned. */
    auto& l = pcm.insert_entry(2, 2, 3435);
    ASSERT_EQ(l.value, 3435);
    ASSERT_EQ(&k, &l);                                // Memory location should be the same
    ASSERT_EQ(pcm.entry_count(), 1);                // The entry count should be unchanged
    ASSERT_EQ(pcm.released_entry_count, m + n + 1); // The total number of allocated entries is unchanged.
}


TEST(SparseMatrix, row_and_column_weights){

    ldpc::sparse_matrix::SparseMatrix<int> matrix(3, 3);
    for(int i = 0; i<3; i++){
        ASSERT_EQ(matrix.get_row_degree(i),0);
        ASSERT_EQ(matrix.get_col_degree(i),0);
    }

    matrix.insert_entry(0,0,10);
    matrix.insert_entry(0,1,10);
    ASSERT_EQ(matrix.get_row_degree(0),2);
    ASSERT_EQ(matrix.get_col_degree(0),1);
    ASSERT_EQ(matrix.get_col_degree(1),1);

    matrix.remove_entry(0,0);

    ASSERT_EQ(matrix.get_row_degree(0),1);
    ASSERT_EQ(matrix.get_col_degree(0),0);
    ASSERT_EQ(matrix.get_col_degree(1),1);

    matrix.insert_entry(0, 0, 5);
    matrix.insert_entry(0, 1, 3);
    matrix.insert_entry(0, 2, 2);
    matrix.insert_entry(1, 0, 8);
    matrix.insert_entry(1, 1, 1);
    matrix.insert_entry(2, 0, 9);
    matrix.insert_entry(2, 2, 4);

    int rc = 0;
    int cc = 0;
    for(int i = 0; i<3; i++){
        rc+=matrix.get_row_degree(i);
        cc+=matrix.get_col_degree(i);
    }
    ASSERT_EQ(rc,cc);
    ASSERT_EQ(rc,matrix.entry_count());

}

TEST(SparseMatrix, sparsity){

    auto matrix = ldpc::sparse_matrix::SparseMatrix<int>(5,5);
    auto e = matrix.insert_entry(1,1,5);
    ASSERT_EQ(matrix.entry_count(),1);
    ASSERT_EQ(matrix.sparsity(),1.0/25);

    matrix.remove(e);
    ASSERT_EQ(matrix.entry_count(),0);
    ASSERT_EQ(matrix.sparsity(),0.0);

    for(int j = 0; j<5; j++){
        for(int i = 0; i<5; i++){
            matrix.insert_entry(i,j,1);
        }
    }

    ASSERT_EQ(matrix.sparsity(),1.0);

}


TEST(SparseMatrix, RowIterateTest)
{
    ldpc::sparse_matrix::SparseMatrix<int> matrix(3, 3);
    matrix.insert_entry(0, 0, 5);
    matrix.insert_entry(0, 1, 3);
    matrix.insert_entry(0, 2, 2);
    matrix.insert_entry(1, 0, 8);
    matrix.insert_entry(1, 1, 1);
    matrix.insert_entry(1, 2, 7);
    matrix.insert_entry(2, 0, 9);
    matrix.insert_entry(2, 1, 6);
    matrix.insert_entry(2, 2, 4);

    // Expected values for each row
    int expected_values[3][3] = {{5, 3, 2}, {8, 1, 7}, {9, 6, 4}};

    // Check each row
    for (int row = 0; row < 3; row++)
    {
        int actual_values[3];
        int i = 0;

        // Iterate through the entries in the row of the matrix
        for (auto entry : matrix.iterate_row(row))
        {
            actual_values[i] = entry.value;
            entry.value+=1; //this is not saved in the matrix (passing by value)
            i++;
        }

        // Check that the values match the expected values
        for (int j = 0; j < 3; j++)
        {
            ASSERT_EQ(expected_values[row][j], actual_values[j]);
        }


        // Iterate through the entries in the row of the matrix
        i = 0;
        for (auto& entry : matrix.iterate_row(row))
        {
            actual_values[i] = entry.value;
            entry.value+=1; //this should now change the matrix values (passing by reference)
            i++;
        }

        i = 0;
        for (auto entry : matrix.iterate_row(row))
        {
            actual_values[i] = entry.value;
            ASSERT_EQ(expected_values[row][i]+1, entry.value);
            i++;
        }

    }
}

TEST(SparseMatrix, ReverseRowIterateTest)
{
    ldpc::sparse_matrix::SparseMatrix<int> matrix(3, 3);
    matrix.insert_entry(0, 0, 5);
    matrix.insert_entry(0, 1, 3);
    matrix.insert_entry(0, 2, 2);
    matrix.insert_entry(1, 0, 8);
    matrix.insert_entry(1, 1, 1);
    matrix.insert_entry(1, 2, 7);
    matrix.insert_entry(2, 0, 9);
    matrix.insert_entry(2, 1, 6);
    matrix.insert_entry(2, 2, 4);

    // Expected values for each row in reverse order
    int expected_values[3][3] = {{2, 3, 5}, {7, 1, 8}, {4, 6, 9}};

    // Check each row
    for (int row = 0; row < 3; row++)
    {
        int actual_values[3];
        int i = 0;

        // Iterate through the entries in the row of the matrix in reverse order
        for (auto& entry : matrix.reverse_iterate_row(row))
        {
            actual_values[i] = entry.value;
            i++;
        }

        // Check that the values match the expected values
        for (int j = 0; j < 3; j++)
        {
            ASSERT_EQ(expected_values[row][j], actual_values[j]);
        }
    }
}

TEST(SparseMatrix, IterateColTest)
{
    ldpc::sparse_matrix::SparseMatrix<int> matrix(3, 3);
    matrix.insert_entry(0, 0, 5);
    matrix.insert_entry(0, 1, 3);
    matrix.insert_entry(0, 2, 2);
    matrix.insert_entry(1, 0, 8);
    matrix.insert_entry(1, 1, 1);
    matrix.insert_entry(1, 2, 7);
    matrix.insert_entry(2, 0, 9);
    matrix.insert_entry(2, 1, 6);
    matrix.insert_entry(2, 2, 4);

    //   print_sparse_matrix(matrix);

    // Expected values for each column
    int expected_values[3][3] = {{5, 8, 9}, {3, 1, 6}, {2, 7, 4}};

    // Check each column
    for (int col = 0; col < 3; col++)
    {
        int actual_values[3];
        int i = 0;

        // Iterate through the entries in the column of the matrix
        for (auto& entry : matrix.iterate_column(col))
        {
            //   std::cout<<entry.value<<endl;
            actual_values[i] = entry.value;
            i++;
        }

        // Check that the values match the expected values
        for (int j = 0; j < 3; j++)
        {
            ASSERT_EQ(expected_values[col][j], actual_values[j]);
        }
    }
}

TEST(SparseMatrix, ReverseIterateColTest)
{
    ldpc::sparse_matrix::SparseMatrix<int> matrix(3, 3);
    matrix.insert_entry(0, 0, 5);
    matrix.insert_entry(0, 1, 3);
    matrix.insert_entry(0, 2, 2);
    matrix.insert_entry(1, 0, 8);
    matrix.insert_entry(1, 1, 1);
    matrix.insert_entry(1, 2, 7);
    matrix.insert_entry(2, 0, 9);
    matrix.insert_entry(2, 1, 6);
    matrix.insert_entry(2, 2, 4);

    // Expected values for each column in reverse order
    int expected_values[3][3] = {{9, 8, 5}, {6, 1, 3}, {4, 7, 2}};

    // Check each column
    for (int col = 0; col < 3; col++)
    {
        int actual_values[3] ={0,0,0};
        int i = 0;

        // Iterate through the entries in the column of the matrix in reverse order
        for (auto& entry : matrix.reverse_iterate_column(col))
        {
            actual_values[i] = entry.value;
            i++;
        }

        // Check that the values match the expected values
        for (int j = 0; j < 3; j++)
        {
            ASSERT_EQ(expected_values[col][j], actual_values[j]);
        }
    }
}

TEST(SparseMatrix, SwapRowsTest)
{
    ldpc::sparse_matrix::SparseMatrix<int> matrix(3, 3);
    matrix.insert_entry(0, 0, 1);
    matrix.insert_entry(0, 1, 2);
    matrix.insert_entry(0, 2, 3);
    matrix.insert_entry(1, 0, 4);
    matrix.insert_entry(1, 1, 5);
    matrix.insert_entry(1, 2, 6);
    matrix.insert_entry(2, 0, 7);
    matrix.insert_entry(2, 1, 8);
    matrix.insert_entry(2, 2, 9);

    matrix.swap_rows(0, 2);

    EXPECT_EQ(matrix.get_entry(0, 0).value, 7);
    EXPECT_EQ(matrix.get_entry(0, 1).value, 8);
    EXPECT_EQ(matrix.get_entry(0, 2).value, 9);
    EXPECT_EQ(matrix.get_entry(1, 0).value, 4);
    EXPECT_EQ(matrix.get_entry(1, 1).value, 5);
    EXPECT_EQ(matrix.get_entry(1, 2).value, 6);
    EXPECT_EQ(matrix.get_entry(2, 0).value, 1);
    EXPECT_EQ(matrix.get_entry(2, 1).value, 2);
    EXPECT_EQ(matrix.get_entry(2, 2).value, 3);
}

// Test for the swap_cols method
TEST(SparseMatrix, SwapColsTest) {
    ldpc::sparse_matrix::SparseMatrix<int> matrix(3, 3);
    matrix.insert_entry(0, 0, 1);
    matrix.insert_entry(0, 1, 2);
    matrix.insert_entry(0, 2, 3);
    matrix.insert_entry(1, 0, 4);
    matrix.insert_entry(1, 1, 5);
    matrix.insert_entry(1, 2, 6);
    matrix.insert_entry(2, 0, 7);
    matrix.insert_entry(2, 1, 8);
    matrix.insert_entry(2, 2, 9);

    matrix.swap_columns(0, 2);

    EXPECT_EQ(matrix.get_entry(0, 0).value, 3);
    EXPECT_EQ(matrix.get_entry(0, 1).value, 2);
    EXPECT_EQ(matrix.get_entry(0, 2).value, 1);
    EXPECT_EQ(matrix.get_entry(1, 0).value, 6);
    EXPECT_EQ(matrix.get_entry(1, 1).value, 5);
    EXPECT_EQ(matrix.get_entry(1, 2).value, 4);
    EXPECT_EQ(matrix.get_entry(2, 0).value, 9);
    EXPECT_EQ(matrix.get_entry(2, 1).value, 8);
    EXPECT_EQ(matrix.get_entry(2, 2).value, 7);
}

TEST(SparseMatrix, InsertRowTest)
{
    ldpc::sparse_matrix::SparseMatrix<int> matrix(8, 7);
    std::vector<int> col_indices = {1, 3, 4, 6};
    std::vector<int> values = {10, 20, 30, 40};
    matrix.insert_row(2, col_indices, values);

    // Check that only the specified entries are inserted
    EXPECT_EQ(matrix.entry_count(), 4);

    // Check the values of the inserted entries
    EXPECT_EQ(matrix.get_entry(2, 1).value, 10);
    EXPECT_EQ(matrix.get_entry(2, 3).value, 20);
    EXPECT_EQ(matrix.get_entry(2, 4).value, 30);
    EXPECT_EQ(matrix.get_entry(2, 6).value, 40);

    // Check that all other entries are still zero
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 7; j++)
        {
            if (i == 2 && (j == 1 || j == 3 || j == 4 || j == 6))
                continue;
            EXPECT_EQ(matrix.get_entry(i, j).value, 0);
        }
    }
}

TEST(SparseMatrix,reorder_rows){
    ldpc::sparse_matrix::SparseMatrix<int> matrix(3, 3);
    matrix.insert_entry(0, 0, 1);
    matrix.insert_entry(0, 1, 2);
    matrix.insert_entry(0, 2, 3);
    matrix.insert_entry(1, 0, 4);
    matrix.insert_entry(1, 1, 5);
    matrix.insert_entry(1, 2, 6);
    matrix.insert_entry(2, 0, 7);
    matrix.insert_entry(2, 1, 8);
    matrix.insert_entry(2, 2, 9);

    std::vector<int> new_order = {2,0,1};
    matrix.reorder_rows(new_order);

    EXPECT_EQ(matrix.get_entry(0, 0).value, 7);
    EXPECT_EQ(matrix.get_entry(0, 1).value, 8);
    EXPECT_EQ(matrix.get_entry(0, 2).value, 9);
    EXPECT_EQ(matrix.get_entry(1, 0).value, 1);
    EXPECT_EQ(matrix.get_entry(1, 1).value, 2);
    EXPECT_EQ(matrix.get_entry(1, 2).value, 3);
    EXPECT_EQ(matrix.get_entry(2, 0).value, 4);
    EXPECT_EQ(matrix.get_entry(2, 1).value, 5);
    EXPECT_EQ(matrix.get_entry(2, 2).value, 6);
}

TEST(SparseMatrix, block_allocate_STACK) {
    ldpc::sparse_matrix::SparseMatrix<int> pcm(3, 3, 5);
    ASSERT_EQ(pcm.entry_count(), 0);
    ASSERT_EQ(pcm.allocated_entry_count, 11);
    ASSERT_EQ(pcm.released_entry_count, 6);
    ASSERT_EQ(pcm.block_idx, 0);
    ASSERT_EQ(pcm.block_position, 6);

    pcm.insert_entry(0, 0, 1);
    ASSERT_EQ(pcm.entry_count(), 1);
    ASSERT_EQ(pcm.allocated_entry_count, 11);
    ASSERT_EQ(pcm.released_entry_count, 7);
    ASSERT_EQ(pcm.block_idx, 0);
    ASSERT_EQ(pcm.block_position, 7);

    pcm.remove_entry(0, 0);

    ASSERT_EQ(pcm.entry_count(), 0);
    ASSERT_EQ(pcm.allocated_entry_count, 11);
    ASSERT_EQ(pcm.released_entry_count, 7);
    ASSERT_EQ(pcm.block_idx, 0);
    ASSERT_EQ(pcm.block_position, 7);

    pcm.insert_entry(1, 1, 99);

    ASSERT_EQ(pcm.entry_count(), 1);
    ASSERT_EQ(pcm.allocated_entry_count, 11);
    ASSERT_EQ(pcm.released_entry_count, 7);
    ASSERT_EQ(pcm.block_idx, 0);
    ASSERT_EQ(pcm.block_position, 7);

    pcm.insert_entry(1, 2, 99);

    ASSERT_EQ(pcm.entry_count(), 2);
    ASSERT_EQ(pcm.allocated_entry_count, 11);
    ASSERT_EQ(pcm.released_entry_count, 8);
    ASSERT_EQ(pcm.block_idx, 0);
    ASSERT_EQ(pcm.block_position, 8);

    pcm.insert_entry(2, 0, 99);
    pcm.insert_entry(2, 1, 99);
    pcm.insert_entry(2, 2, 99);

    ASSERT_EQ(pcm.entry_count(), 5);
    ASSERT_EQ(pcm.allocated_entry_count, 11);
    ASSERT_EQ(pcm.released_entry_count, 11);
    ASSERT_EQ(pcm.block_idx, 0);
    ASSERT_EQ(pcm.block_position, 11);

    pcm.insert_entry(0, 0, 99);

    ASSERT_EQ(pcm.entry_count(), 6);
    ASSERT_EQ(pcm.allocated_entry_count, 17);
    ASSERT_EQ(pcm.released_entry_count, 12);
    ASSERT_EQ(pcm.block_idx, 1);
    ASSERT_EQ(pcm.block_position, 1);
}


TEST(SparseMatrix, MoveSemantics)
{
    ldpc::sparse_matrix::SparseMatrix<int> matrix(8, 7);
    std::vector<int> col_indices = {1, 3, 4, 6};
    std::vector<int> values = {10, 20, 30, 40};
    matrix.insert_row(2, col_indices, values);

    ldpc::sparse_matrix::SparseMatrix<int> matrix2(8,7);
    std::vector<ldpc::sparse_matrix::SparseMatrix<int>> matrices = {matrix,matrix2};

    ASSERT_TRUE(true);


}

// Test for the nonzero_rows method
TEST(SparseMatrix, NonzeroRowsTest) {
    ldpc::sparse_matrix::SparseMatrix<int> matrix(3, 3);
    matrix.insert_entry(0, 0, 1);
    matrix.insert_entry(0, 2, 3);
    matrix.insert_entry(1, 1, 5);
    matrix.insert_entry(2, 0, 7);
    matrix.insert_entry(2, 2, 9);

    std::vector<std::vector<int>> expected_output = {
        {0, 2},
        {1},
        {0, 2}
    };

    std::vector<std::vector<int>> nonzero_rows = matrix.nonzero_rows();

    ASSERT_EQ(nonzero_rows, expected_output);
}

TEST(SparseMatrix, nonzero_coordinates){

    auto mat = ldpc::sparse_matrix::SparseMatrix<int>(2,3);
    mat.insert_entry(0,0,1);
    mat.insert_entry(0,1,1);
    mat.insert_entry(1,1,1);
    mat.insert_entry(1,2,1);

    auto nonzero_coordinates = mat.nonzero_coordinates();

    ASSERT_TRUE(nonzero_coordinates.size()==4);

    auto expected_coordinates = std::vector<std::vector<int>>{{0,0},{0,1},{1,1},{1,2}};

    ASSERT_TRUE(nonzero_coordinates==expected_coordinates);

}

TEST(SparseMatrix, row_adjacency_list){
    
        auto mat = ldpc::sparse_matrix::SparseMatrix<int>(2,3);
        mat.insert_entry(0,0,1);
        mat.insert_entry(0,1,1);
        mat.insert_entry(1,1,1);
        mat.insert_entry(1,2,1);
    
        auto row_adjacency_list = mat.row_adjacency_list();
    
        ASSERT_TRUE(row_adjacency_list.size()==2);
    
        auto expected_row_adjacency_list = std::vector<std::vector<int>>{{0,1},{1,2}};
    
        ASSERT_TRUE(row_adjacency_list==expected_row_adjacency_list);
    
}

TEST(SparseMatrix, col_adjacency_list){
        
            auto mat = ldpc::sparse_matrix::SparseMatrix<int>(2,3);
            mat.insert_entry(0,0,1);
            mat.insert_entry(0,1,1);
            mat.insert_entry(1,1,1);
            mat.insert_entry(1,2,1);
        
            auto col_adjacency_list = mat.col_adjacency_list();
        
            ASSERT_TRUE(col_adjacency_list.size()==3);
        
            auto expected_col_adjacency_list = std::vector<std::vector<int>>{{0},{0,1},{1}};
        
            ASSERT_TRUE(col_adjacency_list==expected_col_adjacency_list);
        
}

TEST(SparseMatrix, to_csr_matrix){
    ASSERT_TRUE(true);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}