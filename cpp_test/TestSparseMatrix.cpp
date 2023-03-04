#include <gtest/gtest.h>
#include "sparse_matrix.hpp"
#include "sparse_matrix_util.hpp"

TEST(TestSparseMatrix, InitialisationAndAllocation)
{
    /*Testing initialisation, entry insert, entry removal.*/

    auto pcm = new SparseMatrix<int>(3, 4);

    // Test constructor and allocation.
    int m, n;
    m = 3;
    n = 4;
    ASSERT_EQ(pcm->m, m);
    ASSERT_EQ(pcm->n, n);
    ASSERT_EQ(pcm->released_entry_count, m + n);
    ASSERT_EQ(pcm->entry_count(), 0);

    // Test `insert_entry(i,j,value)
    auto e = pcm->insert_entry(1, 2, 1);
    ASSERT_EQ(e->value, 1);
    ASSERT_EQ(e->row_index, 1);
    ASSERT_EQ(e->col_index, 2);
    ASSERT_EQ(pcm->released_entry_count, m + n + 1);

    // Test `get_entry(i,j,value)
    auto g = pcm->get_entry(1, 2);
    ASSERT_EQ(g, e);
    ASSERT_EQ(g->value, 1);
    ASSERT_EQ(g->row_index, 1);
    ASSERT_EQ(g->col_index, 2);

    // Test `remove_entry(i,j)`
    pcm->remove_entry(1, 2);
    auto f = pcm->get_entry(1, 2);
    ASSERT_NE(f, g);
    /*the removed entry is stored in the `removed_entries buffer`. The total
    number of released entries should therefore remain the same. */
    ASSERT_EQ(pcm->released_entry_count, m + n + 1);
    ASSERT_EQ(pcm->entry_count(), 0);

    /*Test allocation buffer New entries should preferentially come from the
    `removed entries buffer`. We therefore expect the number of release
    entries to the stay the same below:  */

    auto k = pcm->insert_entry(2, 2, 1);
    ASSERT_EQ(pcm->entry_count(), 1);
    ASSERT_EQ(pcm->released_entry_count, m + n + 1);

    /*If we insert an entry that already exists, we use the memory that has
    already been allocated. In the test below, the value of the this entry is
    reassigned.*/
    auto l = pcm->insert_entry(2, 2, 3435);
    ASSERT_EQ(l->value, 3435);
    ASSERT_EQ(k, l);                                 // memory location should be the same
    ASSERT_EQ(pcm->entry_count(), 1);                // the entry count should be unchanged
    ASSERT_EQ(pcm->released_entry_count, m + n + 1); // the total number of allocated entries is unchanged.

    delete pcm;
}


TEST(SparseMatrixTest, RowIterateTest)
{
    SparseMatrix<int> matrix(3, 3);
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
            actual_values[i] = entry->value;
            i++;
        }

        // Check that the values match the expected values
        for (int j = 0; j < 3; j++)
        {
            ASSERT_EQ(expected_values[row][j], actual_values[j]);
        }
    }
}

TEST(SparseMatrixTest, ReverseRowIterateTest)
{
    SparseMatrix<int> matrix(3, 3);
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
        for (auto entry : matrix.reverse_iterate_row(row))
        {
            actual_values[i] = entry->value;
            i++;
        }

        // Check that the values match the expected values
        for (int j = 0; j < 3; j++)
        {
            ASSERT_EQ(expected_values[row][j], actual_values[j]);
        }
    }
}

TEST(SparseMatrixTest, IterateColTest)
{
    SparseMatrix<int> matrix(3, 3);
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
        for (auto entry : matrix.iterate_column(col))
        {
            //   std::cout<<entry->value<<endl;
            actual_values[i] = entry->value;
            i++;
        }

        // Check that the values match the expected values
        for (int j = 0; j < 3; j++)
        {
            ASSERT_EQ(expected_values[col][j], actual_values[j]);
        }
    }
}

TEST(SparseMatrixTest, ReverseIterateColTest)
{
    SparseMatrix<int> matrix(3, 3);
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
        int actual_values[3];
        int i = 0;

        // Iterate through the entries in the column of the matrix in reverse order
        for (auto entry : matrix.reverse_iterate_column(col))
        {
            actual_values[i] = entry->value;
            i++;
        }

        // Check that the values match the expected values
        for (int j = 0; j < 3; j++)
        {
            ASSERT_EQ(expected_values[col][j], actual_values[j]);
        }
    }
}

TEST(SparseMatrixTest, SwapRowsTest)
{
    SparseMatrix<int> matrix(3, 3);
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

    EXPECT_EQ(matrix.get_entry(0, 0)->value, 7);
    EXPECT_EQ(matrix.get_entry(0, 1)->value, 8);
    EXPECT_EQ(matrix.get_entry(0, 2)->value, 9);
    EXPECT_EQ(matrix.get_entry(1, 0)->value, 4);
    EXPECT_EQ(matrix.get_entry(1, 1)->value, 5);
    EXPECT_EQ(matrix.get_entry(1, 2)->value, 6);
    EXPECT_EQ(matrix.get_entry(2, 0)->value, 1);
    EXPECT_EQ(matrix.get_entry(2, 1)->value, 2);
    EXPECT_EQ(matrix.get_entry(2, 2)->value, 3);
}

TEST(SparseMatrixTest, InsertRowTest)
{
    SparseMatrix<int> matrix(8, 7);
    vector<int> col_indices = {1, 3, 4, 6};
    vector<int> values = {10, 20, 30, 40};
    matrix.insert_row(2, col_indices, values);

    // Check that only the specified entries are inserted
    EXPECT_EQ(matrix.entry_count(), 4);

    // Check the values of the inserted entries
    EXPECT_EQ(matrix.get_entry(2, 1)->value, 10);
    EXPECT_EQ(matrix.get_entry(2, 3)->value, 20);
    EXPECT_EQ(matrix.get_entry(2, 4)->value, 30);
    EXPECT_EQ(matrix.get_entry(2, 6)->value, 40);

    // Check that all other entries are still zero
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 7; j++)
        {
            if (i == 2 && (j == 1 || j == 3 || j == 4 || j == 6))
                continue;
            EXPECT_EQ(matrix.get_entry(i, j)->value, 0);
        }
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}