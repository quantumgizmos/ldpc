#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "sparse_matrix_util.hpp"

#include "util.hpp"

TEST(UtilTest, DecimalToBinarySparse) {
    int i = 16;
    auto sdec = ldpc::util::decimal_to_binary_sparse(i, 6);
    auto expected = std::vector<int>{4};
    ASSERT_EQ(sdec, expected);
}

TEST(UtilTest, DecimalToBinarySparse2) {
    int i = 17;
    auto sdec = ldpc::util::decimal_to_binary_sparse(i, 6);
    auto expected = std::vector<int>{0,4};
    ASSERT_EQ(sdec, expected);
}

TEST(UtilTest, DecimalToBinarySparse_Zero) {
    int i = 0;
    auto sdec = ldpc::util::decimal_to_binary_sparse(i, 6);
    auto expected = std::vector<int>{};
    ASSERT_EQ(sdec, expected);
}

TEST(UtilTest, DecimalToBinarySparse_Max) {
    int i = 63;
    auto sdec = ldpc::util::decimal_to_binary_sparse(i, 6);
    auto expected = std::vector<int>{0, 1, 2, 3, 4, 5};
    ASSERT_EQ(sdec, expected);
}

TEST(UtilTest, DecimalToBinarySparse_Negative) {
    int i = -17;
    auto sdec = ldpc::util::decimal_to_binary_sparse(i, 6);
    auto expected = std::vector<int>{};
    ASSERT_EQ(sdec, expected);
}

TEST(UtilTest, DecimalToBinarySparse_LargeNumber) {
    int i = 1024;
    auto sdec = ldpc::util::decimal_to_binary_sparse(i, 6);
    auto expected = std::vector<int>{};
    ASSERT_EQ(sdec, expected);
}