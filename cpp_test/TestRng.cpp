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

// Test that RandomListShuffle shuffles a list correctly
TEST(RandomListShuffleTest, Shuffle_ShufflesList) {
    ldpc::rng::RandomListShuffle<int> shuffler;  // Create a RandomListShuffle object
    vector<int> original_list = {1, 2, 3, 4, 5};  // Define a list
    vector<int> shuffled_list = original_list;  // Copy the list
    shuffler.shuffle(shuffled_list);  // Shuffle the list

    // Check that the shuffled list contains the same elements as the original
    ASSERT_EQ(original_list.size(), shuffled_list.size());
    ASSERT_TRUE(is_permutation(original_list.begin(), original_list.end(), shuffled_list.begin()));

    // Check that the shuffled list is not in the same order as the original
    ASSERT_NE(original_list, shuffled_list);
}

// Test that RandomListShuffle works with an empty list
TEST(RandomListShuffleTest, Shuffle_EmptyList) {
    ldpc::rng::RandomListShuffle<int> shuffler;  // Create a RandomListShuffle object
    vector<int> empty_list;  // Define an empty list
    shuffler.shuffle(empty_list);  // Shuffle the empty list

    // Check that the shuffled list is still empty
    ASSERT_TRUE(empty_list.empty());
}

// Test that RandomListShuffle works with a single-element list
TEST(RandomListShuffleTest, Shuffle_SingleElementList) {
    ldpc::rng::RandomListShuffle<int> shuffler;  // Create a RandomListShuffle object
    vector<int> single_element_list = {42};  // Define a single-element list
    shuffler.shuffle(single_element_list);  // Shuffle the list

    // Check that the shuffled list is unchanged
    ASSERT_EQ(single_element_list.size(), 1);
    ASSERT_EQ(single_element_list[0], 42);
}

// Test that setting the seed for RandomListShuffle produces consistent results
TEST(RandomListShuffleTest, Shuffle_ConsistentWithSeed) {
    ldpc::rng::RandomListShuffle<int> shuffler1;  // Create the first RandomListShuffle object
    ldpc::rng::RandomListShuffle<int> shuffler2;  // Create the second RandomListShuffle object

    unsigned int seed = 12345;  // Define a seed
    shuffler1.seed(seed);  // Set the seed for the first shuffler
    shuffler2.seed(seed);  // Set the same seed for the second shuffler

    vector<int> list1 = {1, 2, 3, 4, 5};  // Define the first list
    vector<int> list2 = list1;  // Copy the list for the second shuffler

    shuffler1.shuffle(list1);  // Shuffle the first list
    shuffler2.shuffle(list2);  // Shuffle the second list

    // Check that the shuffled lists are identical
    ASSERT_EQ(list1, list2);
}

// Test that setting seed=0 initializes the RNG from the system clock
TEST(RandomListShuffleTest, Shuffle_SeedZeroUsesSystemClock) {
    ldpc::rng::RandomListShuffle<int> shuffler1;  // Create the first RandomListShuffle object
    ldpc::rng::RandomListShuffle<int> shuffler2;  // Create the second RandomListShuffle object

    shuffler1.seed(0);  // Set the seed to 0 for the first shuffler
    shuffler2.seed(0);  // Set the seed to 0 for the second shuffler

    vector<int> list1 = {1, 2, 3, 4, 5};  // Define the first list
    vector<int> list2 = list1;  // Copy the list for the second shuffler

    shuffler1.shuffle(list1);  // Shuffle the first list
    shuffler2.shuffle(list2);  // Shuffle the second list

    // Check that the shuffled lists are not identical, as they should use different seeds from the system clock
    ASSERT_NE(list1, list2);
}