// FILEPATH: /home/joschka/github/ldpc/cpp_test/TestRng.cpp
#include <gtest/gtest.h>
#include <climits>
#include "rng.hpp"

// Using the namespaces for convenience
using namespace ldpc::rng;
using namespace std;

// Test that a random double is within the range [0, 1]
TEST(RandomNumberGeneratorTest, RandomDouble) {
    RandomNumberGenerator rng;  // Create a RandomNumberGenerator object
    double random_num = rng.random_double();  // Generate a random double
    ASSERT_GE(random_num, 0.0);  // Check that the random number is greater than or equal to 0
    ASSERT_LE(random_num, 1.0);  // Check that the random number is less than or equal to 1
}

// Test that a random integer is within the range [0, max_int]
TEST(RandomNumberGeneratorTest, RandomInt) {
    RandomNumberGenerator rng;  // Create a RandomNumberGenerator object
    int max_int = 10;
    int random_num = rng.random_int(max_int);  // Generate a random integer
    ASSERT_GE(random_num, 0);  // Check that the random number is greater than or equal to 0
    ASSERT_LE(random_num, max_int);  // Check that the random number is less than or equal to max_int
}

// Test that a random integer is within the range [0, INT_MAX]
TEST(RandomNumberGeneratorTest, RandomInt_LargeMaxInt) {
    RandomNumberGenerator rng;  // Create a RandomNumberGenerator object
    int max_int = INT_MAX;  // Set the maximum integer value
    int random_num = rng.random_int(max_int);  // Generate a random integer
    ASSERT_GE(random_num, 0);  // Check that the random number is greater than or equal to 0
    ASSERT_LE(random_num, max_int);  // Check that the random number is less than or equal to max_int
}

// Test that a random integer is 0 when max_int is 0
TEST(RandomNumberGeneratorTest, RandomInt_ZeroMaxInt) {
    RandomNumberGenerator rng;  // Create a RandomNumberGenerator object
    int max_int = 0;  // Set the maximum integer value to 0
    int random_num = rng.random_int(max_int);  // Generate a random integer
    ASSERT_EQ(random_num, 0);  // Check that the random number is equal to 0
}


// Test that two calls to random_int produce different results
TEST(RandomNumberGeneratorTest, RandomInt_MultipleCalls) {
    RandomNumberGenerator rng;  // Create a RandomNumberGenerator object
    int max_int = 10;  // Set the maximum integer value
    int random_num1 = rng.random_int(max_int);  // Generate a random integer
    int random_num2 = rng.random_int(max_int);  // Generate another random integer
    ASSERT_NE(random_num1, random_num2);  // Check that the two random numbers are not equal
}