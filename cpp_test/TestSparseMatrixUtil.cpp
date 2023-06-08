#include <gtest/gtest.h>
#include "sparse_matrix_util.hpp"

using namespace sparse_matrix;

TEST(PrintSparseMatrixTest, PrintsCorrectly) {
    SparseMatrix<int> matrix(3, 4);
    matrix.insert_entry(0, 0, 1);
    matrix.insert_entry(0, 2, 2);
    matrix.insert_entry(1, 1, 3);
    matrix.insert_entry(1, 3, 4);
    matrix.insert_entry(2, 0, 5);
    matrix.insert_entry(2, 1, 6);

    std::stringstream expected_output;
    expected_output << "1 0 2 0" << std::endl
                    << "0 3 0 4" << std::endl
                    << "5 6 0 0";

    std::stringstream actual_output;
    actual_output=print_sparse_matrix(matrix,true);

    EXPECT_EQ(actual_output.str(), expected_output.str());
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}