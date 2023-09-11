#include <gtest/gtest.h>
// #include "sparse_matrix_util.hpp"
#include "gf2sparse.hpp"
#include "bp.hpp"
#include "sparse_matrix_util.hpp"

using namespace std;

TEST(BpEntry, init)
{
    auto e = bp::BpEntry();
    ASSERT_EQ(e.row_index, -100);
    ASSERT_EQ(e.col_index, -100);
    ASSERT_EQ(e.at_end(), true);
    ASSERT_EQ(e.bit_to_check_msg, 0.0);
    ASSERT_EQ(e.bit_to_check_msg, 0.0);
}

TEST(BpSparse, init)
{

    int n = 3;
    auto pcm = bp::BpSparse(n - 1, n);
    for (int i = 0; i < (n - 1); i++)
    {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }

    auto expected = "1 1 0\n0 1 1";
    ASSERT_EQ(print_sparse_matrix(pcm, true).str(), expected);
}

TEST(BpDecoderTest, InitializationTest)
{
    // Define input arguments for initialization
    int n = 3;
    auto pcm = bp::BpSparse(n - 1, n);
    for (int i = 0; i < (n - 1); i++)
    {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }
    int maximum_iterationsations = pcm.n;
    auto channel_probabilities = vector<double>(pcm.n, 0.1);

    // // Initialize decoder using input arguments
    auto decoder = bp::BpDecoder(pcm, channel_probabilities, maximum_iterationsations);

    // Check if member variables are set correctly
    EXPECT_TRUE(pcm.gf2_equal(pcm));
    EXPECT_TRUE(pcm == decoder.pcm);
    EXPECT_EQ(pcm.m, decoder.check_count);
    EXPECT_EQ(pcm.n, decoder.bit_count);
    EXPECT_EQ(channel_probabilities, decoder.channel_probabilities);
    EXPECT_EQ(maximum_iterationsations, decoder.maximum_iterations);
    EXPECT_EQ(0.625, decoder.ms_scaling_factor);
    EXPECT_EQ(bp::PRODUCT_SUM, decoder.bp_method);
    EXPECT_EQ(bp::PARALLEL, decoder.schedule);
    EXPECT_EQ(1, decoder.omp_thread_count);
    EXPECT_EQ(0, decoder.random_schedule_seed);
}

TEST(BpDecoderTest, InitializationWithOptionalParametersTest)
{
    // Define input arguments for initialization
    int n = 4;
    auto pcm = bp::BpSparse(n - 1, n);
    for (int i = 0; i < (n - 1); i++)
    {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }
    int maximum_iterations = 10;
    auto channel_probabilities = vector<double>{0.1, 0.2, 0.3, 0.4};

    // Define optional input parameters
    bp::BpMethod bp_method = bp::MINIMUM_SUM;
    double min_sum_scaling_factor = 0.5;
    bp::BpSchedule bp_schedule = bp::SERIAL;
    int omp_threads = 4;
    vector<int> serial_schedule{1, 3, 0, 2};
    int random_schedule = 0;

    // Initialize decoder using input arguments and optional parameters
    auto decoder = bp::BpDecoder(pcm, channel_probabilities, maximum_iterations, bp_method, bp_schedule, min_sum_scaling_factor, omp_threads, serial_schedule, random_schedule);

    // Check if member variables are set correctly
    EXPECT_TRUE(pcm == decoder.pcm);
    EXPECT_EQ(pcm.m, decoder.check_count);
    EXPECT_EQ(pcm.n, decoder.bit_count);
    EXPECT_EQ(channel_probabilities, decoder.channel_probabilities);
    EXPECT_EQ(maximum_iterations, decoder.maximum_iterations);
    EXPECT_EQ(min_sum_scaling_factor, decoder.ms_scaling_factor);
    EXPECT_EQ(bp_method, decoder.bp_method);
    EXPECT_EQ(bp_schedule, decoder.schedule);
    EXPECT_EQ(omp_threads, decoder.omp_thread_count);
    EXPECT_EQ(serial_schedule, decoder.serial_schedule_order);
    EXPECT_EQ(random_schedule, decoder.random_schedule_seed);
    EXPECT_EQ(omp_threads, omp_get_max_threads());
}

TEST(BpDecoderTest, InitialiseLogDomainBpTest)
{
    // Define input arguments for initialization
    int n = 3;
    auto pcm = bp::BpSparse(n - 1, n);
    for (int i = 0; i < (n - 1); i++)
    {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }
    int maximum_iterations = 100;
    auto channel_probabilities = vector<double>{0.1, 0.2, 0.3};

    // Initialize decoder using input arguments
    auto decoder = bp::BpDecoder(pcm, channel_probabilities, maximum_iterations);

    // Call initialise_log_domain_bp() function
    decoder.initialise_log_domain_bp();

    // Check if member variables are set correctly
    for (int i = 0; i < decoder.bit_count; i++)
    {
        EXPECT_EQ(log((1 - channel_probabilities[i]) / channel_probabilities[i]), decoder.initial_log_prob_ratios[i]);

        for (auto& e : decoder.pcm.iterate_column(i))
        {
            EXPECT_EQ(decoder.initial_log_prob_ratios[i], e.bit_to_check_msg);
        }
    }
}


TEST(BpDecoder, product_sum_parallel){
    
    int n = 3;
    auto pcm = bp::BpSparse(n - 1, n);
    for (int i = 0; i < (n - 1); i++)
    {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }
    int maximum_iterations = pcm.n;
    auto channel_probabilities = vector<double>(pcm.n, 0.1);

    // Initialize decoder using input arguments
    auto decoder = bp::BpDecoder(pcm, channel_probabilities, maximum_iterations,bp::PRODUCT_SUM,bp::PARALLEL,79879879);

    // Check if member variables are set correctly
    EXPECT_TRUE(pcm == decoder.pcm);
    EXPECT_EQ(pcm.m, decoder.check_count);
    EXPECT_EQ(pcm.n, decoder.bit_count);
    EXPECT_EQ(channel_probabilities, decoder.channel_probabilities);
    EXPECT_EQ(maximum_iterations, decoder.maximum_iterations);
    EXPECT_EQ(79879879, decoder.ms_scaling_factor);
    EXPECT_EQ(0, decoder.bp_method);
    EXPECT_EQ(bp::PARALLEL, decoder.schedule);
    EXPECT_EQ(1, decoder.omp_thread_count);
    EXPECT_EQ(0, decoder.random_schedule_seed);

    auto syndromes = vector<vector<uint8_t>>{{0,0},{0,1},{1,0},{1,1}};
    auto expected_decoding = vector<vector<uint8_t>>{{0,0,0},{0,0,1},{1,0,0},{0,1,0}};

    auto count = 0;
    for(auto syndrome: syndromes){
        auto decoding = decoder.decode(syndrome);
        ASSERT_EQ(expected_decoding[count], decoding);
        count++;
    }
    
}

TEST(BpDecoder, ProdSumParallel_RepetitionCode5) {
    int n = 5;
    auto pcm = bp::BpSparse(n - 1, n);
    for (int i = 0; i < (n - 1); i++) {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }
    int maximum_iterations = pcm.n;
    auto channel_probabilities = vector<double>(pcm.n, 0.1);

    // Initialize decoder using input arguments
    auto decoder = bp::BpDecoder(pcm, channel_probabilities, maximum_iterations, bp::PRODUCT_SUM, bp::PARALLEL,4324234);

    auto syndromes = vector<vector<uint8_t>>{{0, 0, 0, 0}, {0, 0, 0, 1}, {0, 1, 0, 1}, {1, 0, 1, 0}, {1, 1, 1, 1}};
    auto expected_decoding = vector<vector<uint8_t>>{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 1}, {0, 0, 1, 1, 0}, {0, 1, 1, 0, 0}, {0, 1, 0, 1, 0}};

    auto count = 0;
    for (auto syndrome : syndromes) {
        auto decoding = decoder.decode(syndrome);
        ASSERT_EQ(expected_decoding[count], decoding);
        count++;
    }
}


TEST(BpDecoder, MinSum_RepetitionCode5) {
    int n = 5;
    auto pcm = bp::BpSparse(n - 1, n);
    for (int i = 0; i < (n - 1); i++) {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }
    int maximum_iterations = pcm.n;
    auto channel_probabilities = vector<double>(pcm.n, 0.1);

    // Initialize decoder using input arguments
    auto decoder = bp::BpDecoder(pcm, channel_probabilities, maximum_iterations, bp::MINIMUM_SUM, bp::PARALLEL, 1);

    auto syndromes = vector<vector<uint8_t>>{{0, 0, 0, 0}, {0, 0, 0, 1}, {0, 1, 0, 1}, {1, 0, 1, 0}, {1, 1, 1, 1}};
    auto expected_decoding = vector<vector<uint8_t>>{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 1}, {0, 0, 1, 1, 0}, {0, 1, 1, 0, 0}, {0, 1, 0, 1, 0}};

    auto count = 0;
    for (auto syndrome : syndromes) {
        auto decoding = decoder.decode(syndrome);
        ASSERT_EQ(expected_decoding[count], decoding);
        count++;
    }
}


TEST(BpDecoder, ProdSumSerial_RepetitionCode5) {
    int n = 5;
    auto pcm = bp::BpSparse(n - 1, n);
    for (int i = 0; i < (n - 1); i++) {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }
    int maximum_iterations = pcm.n;
    auto channel_probabilities = vector<double>(pcm.n, 0.1);

    // Initialize decoder using input arguments
    auto decoder = bp::BpDecoder(pcm, channel_probabilities, maximum_iterations, bp::PRODUCT_SUM, bp::SERIAL, 4324234);

    auto syndromes = vector<vector<uint8_t>>{{0, 0, 0, 0}, {0, 0, 0, 1}, {0, 1, 0, 1}, {1, 0, 1, 0}, {1, 1, 1, 1}};
    auto expected_decoding = vector<vector<uint8_t>>{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 1}, {0, 0, 1, 1, 0}, {0, 1, 1, 0, 0}, {0, 1, 0, 1, 0}};

    auto count = 0;
    for (auto syndrome : syndromes) {
        auto decoding = decoder.decode(syndrome);
        ASSERT_EQ(expected_decoding[count], decoding);
        count++;
    }
}


TEST(BpDecoder, MinSum_Serial_RepetitionCode5) {
    int n = 5;
    auto pcm = bp::BpSparse(n - 1, n);
    for (int i = 0; i < (n - 1); i++) {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }
    int maximum_iterations = pcm.n;
    auto channel_probabilities = vector<double>(pcm.n, 0.1);

    // Initialize decoder using input arguments
    auto decoder = bp::BpDecoder(pcm, channel_probabilities, maximum_iterations, bp::MINIMUM_SUM, bp::SERIAL,1);

    auto syndromes = vector<vector<uint8_t>>{{0, 0, 0, 0}, {0, 0, 0, 1}, {0, 1, 0, 1}, {1, 0, 1, 0}, {1, 1, 1, 1}};
    auto expected_decoding = vector<vector<uint8_t>>{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 1}, {0, 0, 1, 1, 0}, {0, 1, 1, 0, 0}, {0, 1, 0, 1, 0}};

    auto count = 0;
    for (auto syndrome : syndromes) {
        auto decoding = decoder.decode(syndrome);
        ASSERT_EQ(expected_decoding[count], decoding);
        count++;
    }
}

TEST(BpDecoder, min_sum_parallel){
    
    int n = 3;
    auto pcm = bp::BpSparse(n - 1, n);
    for (int i = 0; i < (n - 1); i++)
    {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }
    int maximum_iterations = pcm.n;
    auto channel_probabilities = vector<double>(pcm.n, 0.1);

    // Initialize decoder using input arguments
    auto decoder = bp::BpDecoder(pcm, channel_probabilities, maximum_iterations,bp::MINIMUM_SUM, bp::PARALLEL,0.625);

    // Check if member variables are set correctly
    EXPECT_TRUE(pcm == decoder.pcm);
    EXPECT_EQ(pcm.m, decoder.check_count);
    EXPECT_EQ(pcm.n, decoder.bit_count);
    EXPECT_EQ(channel_probabilities, decoder.channel_probabilities);
    EXPECT_EQ(maximum_iterations, decoder.maximum_iterations);
    EXPECT_EQ(0.625, decoder.ms_scaling_factor);
    EXPECT_EQ(1, decoder.bp_method);
    EXPECT_EQ(bp::PARALLEL, decoder.schedule);
    EXPECT_EQ(1, decoder.omp_thread_count);
    EXPECT_EQ(0, decoder.random_schedule_seed);

    auto syndromes = vector<vector<uint8_t>>{{0,0},{0,1},{1,0},{1,1}};
    auto expected_decoding = vector<vector<uint8_t>>{{0,0,0},{0,0,1},{1,0,0},{0,1,0}};

    auto count = 0;
    for(auto syndrome: syndromes){
        auto decoding = decoder.decode(syndrome);
        print_vector(decoder.log_prob_ratios);
        ASSERT_EQ(expected_decoding[count], decoding);
        count++;
    }
    
}

TEST(BpDecoder, min_sum_single_scan){
    
    int n = 3;
    auto pcm = bp::BpSparse(n - 1, n);
    for (int i = 0; i < (n - 1); i++)
    {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }
    int maximum_iterations = pcm.n;
    auto channel_probabilities = vector<double>(pcm.n, 0.1);

    // Initialize decoder using input arguments
    auto decoder = bp::BpDecoder(pcm, channel_probabilities, maximum_iterations,bp::MINIMUM_SUM, bp::PARALLEL,0.625);

    // Check if member variables are set correctly
    EXPECT_TRUE(pcm == decoder.pcm);
    EXPECT_EQ(pcm.m, decoder.check_count);
    EXPECT_EQ(pcm.n, decoder.bit_count);
    EXPECT_EQ(channel_probabilities, decoder.channel_probabilities);
    EXPECT_EQ(maximum_iterations, decoder.maximum_iterations);
    EXPECT_EQ(0.625, decoder.ms_scaling_factor);
    EXPECT_EQ(1, decoder.bp_method);
    EXPECT_EQ(bp::PARALLEL, decoder.schedule);
    EXPECT_EQ(1, decoder.omp_thread_count);
    EXPECT_EQ(0, decoder.random_schedule_seed);

    auto syndromes = vector<vector<uint8_t>>{{0,0},{0,1},{1,0},{1,1}};
    auto expected_decoding = vector<vector<uint8_t>>{{0,0,0},{0,0,1},{1,0,0},{0,1,0}};

    auto count = 0;
    for(auto syndrome: syndromes){
        auto decoding = decoder.bp_decode_single_scan(syndrome);
        print_vector(decoder.log_prob_ratios);
        ASSERT_EQ(expected_decoding[count], decoding);
        count++;
    }
    
}


int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}