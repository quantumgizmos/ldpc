#include <gtest/gtest.h>
// #include "sparse_matrix_util.hpp"
#include "gf2sparse.hpp"
#include "bp.hpp"
#include "sparse_matrix_util.hpp"

TEST(BpEntry, init)
{

    auto e = new bp::BpEntry();
    ASSERT_EQ(e->row_index, -100);
    ASSERT_EQ(e->col_index, -100);
    ASSERT_EQ(e->at_end(), true);
    ASSERT_EQ(e->bit_to_check_msg, 0.0);
    ASSERT_EQ(e->bit_to_check_msg, 0.0);
    delete e;
}

TEST(BpSparse, init)
{

    int n = 3;
    auto pcm = bp::BpSparse::New(n - 1, n);
    for (int i = 0; i < (n - 1); i++)
    {
        pcm->insert_entry(i, i);
        pcm->insert_entry(i, (i + 1) % n);
    }

    auto expected = "1 1 0\n0 1 1";
    ASSERT_EQ(print_sparse_matrix(*pcm, true).str(), expected);
}

TEST(BpDecoderTest, InitializationTest)
{
    // Define input arguments for initialization
    int n = 3;
    auto pcm = bp::BpSparse::New(n - 1, n);
    for (int i = 0; i < (n - 1); i++)
    {
        pcm->insert_entry(i, i);
        pcm->insert_entry(i, (i + 1) % n);
    }
    int max_iter = pcm->n;
    auto channel_probabilities = vector<double>(pcm->n, 0.1);

    // Initialize decoder using input arguments
    auto decoder = make_shared<bp::BpDecoder>(pcm, channel_probabilities, max_iter);

    // Check if member variables are set correctly
    EXPECT_EQ(pcm, decoder->pcm);
    EXPECT_EQ(pcm->m, decoder->check_count);
    EXPECT_EQ(pcm->n, decoder->bit_count);
    EXPECT_EQ(channel_probabilities, decoder->channel_probs);
    EXPECT_EQ(max_iter, decoder->max_iter);
    EXPECT_EQ(0.625, decoder->ms_scaling_factor);
    EXPECT_EQ(1, decoder->decoding_method);
    EXPECT_EQ(0, decoder->schedule);
    EXPECT_EQ(1, decoder->omp_thread_count);
    EXPECT_EQ(0, decoder->random_serial_schedule);
}

TEST(BpDecoderTest, InitializationWithOptionalParametersTest)
{
    // Define input arguments for initialization
    int n = 4;
    auto pcm = bp::BpSparse::New(n - 1, n);
    for (int i = 0; i < (n - 1); i++)
    {
        pcm->insert_entry(i, i);
        pcm->insert_entry(i, (i + 1) % n);
    }
    int max_iter = 10;
    auto channel_probabilities = vector<double>{0.1, 0.2, 0.3, 0.4};

    // Define optional input parameters
    int bp_method = 2;
    double min_sum_scaling_factor = 0.5;
    int bp_schedule = 2;
    int omp_threads = 4;
    vector<int> serial_schedule{1, 3, 0, 2};
    int random_schedule = 0;

    // Initialize decoder using input arguments and optional parameters
    auto decoder = make_shared<bp::BpDecoder>(pcm, channel_probabilities, max_iter, bp_method, min_sum_scaling_factor, bp_schedule, omp_threads, serial_schedule, random_schedule);

    // Check if member variables are set correctly
    EXPECT_EQ(pcm, decoder->pcm);
    EXPECT_EQ(pcm->m, decoder->check_count);
    EXPECT_EQ(pcm->n, decoder->bit_count);
    EXPECT_EQ(channel_probabilities, decoder->channel_probs);
    EXPECT_EQ(max_iter, decoder->max_iter);
    EXPECT_EQ(min_sum_scaling_factor, decoder->ms_scaling_factor);
    EXPECT_EQ(bp_method, decoder->decoding_method);
    EXPECT_EQ(bp_schedule, decoder->schedule);
    EXPECT_EQ(omp_threads, decoder->omp_thread_count);
    EXPECT_EQ(serial_schedule, decoder->serial_schedule_order);
    EXPECT_EQ(random_schedule, decoder->random_serial_schedule);
    EXPECT_EQ(omp_threads, omp_get_max_threads());
}

TEST(BpDecoderTest, InitialiseLogDomainBpTest)
{
    // Define input arguments for initialization
    int n = 3;
    auto pcm = bp::BpSparse::New(n - 1, n);
    for (int i = 0; i < (n - 1); i++)
    {
        pcm->insert_entry(i, i);
        pcm->insert_entry(i, (i + 1) % n);
    }
    int max_iter = 100;
    auto channel_probabilities = vector<double>{0.1, 0.2, 0.3};

    // Initialize decoder using input arguments
    auto decoder = make_shared<bp::BpDecoder>(pcm, channel_probabilities, max_iter);

    // Call initialise_log_domain_bp() function
    decoder->initialise_log_domain_bp();

    // Check if member variables are set correctly
    for (int i = 0; i < decoder->bit_count; i++)
    {
        EXPECT_EQ(log((1 - channel_probabilities[i]) / channel_probabilities[i]), decoder->initial_log_prob_ratios[i]);

        for (auto e : decoder->pcm->iterate_column(i))
        {
            EXPECT_EQ(decoder->initial_log_prob_ratios[i], e->bit_to_check_msg);
        }
    }
}


TEST(BpDecoder, product_sum_parallel){
    
    int n = 3;
    auto pcm = bp::BpSparse::New(n - 1, n);
    for (int i = 0; i < (n - 1); i++)
    {
        pcm->insert_entry(i, i);
        pcm->insert_entry(i, (i + 1) % n);
    }
    int max_iter = pcm->n;
    auto channel_probabilities = vector<double>(pcm->n, 0.1);

    // Initialize decoder using input arguments
    auto decoder = make_shared<bp::BpDecoder>(pcm, channel_probabilities, max_iter,0,79879879,0);

    // Check if member variables are set correctly
    EXPECT_EQ(pcm, decoder->pcm);
    EXPECT_EQ(pcm->m, decoder->check_count);
    EXPECT_EQ(pcm->n, decoder->bit_count);
    EXPECT_EQ(channel_probabilities, decoder->channel_probs);
    EXPECT_EQ(max_iter, decoder->max_iter);
    EXPECT_EQ(79879879, decoder->ms_scaling_factor);
    EXPECT_EQ(0, decoder->decoding_method);
    EXPECT_EQ(0, decoder->schedule);
    EXPECT_EQ(1, decoder->omp_thread_count);
    EXPECT_EQ(0, decoder->random_serial_schedule);

    auto syndromes = vector<vector<uint8_t>>{{0,0},{0,1},{1,0},{1,1}};
    auto expected_decoding = vector<vector<uint8_t>>{{0,0,0},{0,0,1},{1,0,0},{0,1,0}};

    auto count = 0;
    for(auto syndrome: syndromes){
        auto decoding = decoder->decode(syndrome);
        ASSERT_EQ(expected_decoding[count], decoding);
        count++;
    }
    
}

TEST(BpDecoder, min_sum_parallel){
    
    int n = 3;
    auto pcm = bp::BpSparse::New(n - 1, n);
    for (int i = 0; i < (n - 1); i++)
    {
        pcm->insert_entry(i, i);
        pcm->insert_entry(i, (i + 1) % n);
    }
    int max_iter = pcm->n;
    auto channel_probabilities = vector<double>(pcm->n, 0.1);

    // Initialize decoder using input arguments
    auto decoder = make_shared<bp::BpDecoder>(pcm, channel_probabilities, max_iter,1,0.625,0);

    // Check if member variables are set correctly
    EXPECT_EQ(pcm, decoder->pcm);
    EXPECT_EQ(pcm->m, decoder->check_count);
    EXPECT_EQ(pcm->n, decoder->bit_count);
    EXPECT_EQ(channel_probabilities, decoder->channel_probs);
    EXPECT_EQ(max_iter, decoder->max_iter);
    EXPECT_EQ(0.625, decoder->ms_scaling_factor);
    EXPECT_EQ(1, decoder->decoding_method);
    EXPECT_EQ(0, decoder->schedule);
    EXPECT_EQ(1, decoder->omp_thread_count);
    EXPECT_EQ(0, decoder->random_serial_schedule);

    auto syndromes = vector<vector<uint8_t>>{{0,0},{0,1},{1,0},{1,1}};
    auto expected_decoding = vector<vector<uint8_t>>{{0,0,0},{0,0,1},{1,0,0},{0,1,0}};

    auto count = 0;
    for(auto syndrome: syndromes){
        auto decoding = decoder->decode(syndrome);
        ASSERT_EQ(expected_decoding[count], decoding);
        count++;
    }
    
}

TEST(BpDecoder, ProdSumParallel_RepetitionCode5) {
    int n = 5;
    auto pcm = bp::BpSparse::New(n - 1, n);
    for (int i = 0; i < (n - 1); i++) {
        pcm->insert_entry(i, i);
        pcm->insert_entry(i, (i + 1) % n);
    }
    int max_iter = pcm->n;
    auto channel_probabilities = vector<double>(pcm->n, 0.1);

    // Initialize decoder using input arguments
    auto decoder = make_shared<bp::BpDecoder>(pcm, channel_probabilities, max_iter, 0, 4324234, 0);

    auto syndromes = vector<vector<uint8_t>>{{0, 0, 0, 0}, {0, 0, 0, 1}, {0, 1, 0, 1}, {1, 0, 1, 0}, {1, 1, 1, 1}};
    auto expected_decoding = vector<vector<uint8_t>>{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 1}, {0, 0, 1, 1, 0}, {0, 1, 1, 0, 0}, {0, 1, 0, 1, 0}};

    auto count = 0;
    for (auto syndrome : syndromes) {
        auto decoding = decoder->decode(syndrome);
        ASSERT_EQ(expected_decoding[count], decoding);
        count++;
    }
}


TEST(BpDecoder, MinSum_RepetitionCode5) {
    int n = 5;
    auto pcm = bp::BpSparse::New(n - 1, n);
    for (int i = 0; i < (n - 1); i++) {
        pcm->insert_entry(i, i);
        pcm->insert_entry(i, (i + 1) % n);
    }
    int max_iter = pcm->n;
    auto channel_probabilities = vector<double>(pcm->n, 0.1);

    // Initialize decoder using input arguments
    auto decoder = make_shared<bp::BpDecoder>(pcm, channel_probabilities, max_iter, 1, 1, 0);

    auto syndromes = vector<vector<uint8_t>>{{0, 0, 0, 0}, {0, 0, 0, 1}, {0, 1, 0, 1}, {1, 0, 1, 0}, {1, 1, 1, 1}};
    auto expected_decoding = vector<vector<uint8_t>>{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 1}, {0, 0, 1, 1, 0}, {0, 1, 1, 0, 0}, {0, 1, 0, 1, 0}};

    auto count = 0;
    for (auto syndrome : syndromes) {
        auto decoding = decoder->decode(syndrome);
        ASSERT_EQ(expected_decoding[count], decoding);
        count++;
    }
}


TEST(BpDecoder, ProdSumSerial_RepetitionCode5) {
    int n = 5;
    auto pcm = bp::BpSparse::New(n - 1, n);
    for (int i = 0; i < (n - 1); i++) {
        pcm->insert_entry(i, i);
        pcm->insert_entry(i, (i + 1) % n);
    }
    int max_iter = pcm->n;
    auto channel_probabilities = vector<double>(pcm->n, 0.1);

    // Initialize decoder using input arguments
    auto decoder = make_shared<bp::BpDecoder>(pcm, channel_probabilities, max_iter, 0, 4324234, 1);

    auto syndromes = vector<vector<uint8_t>>{{0, 0, 0, 0}, {0, 0, 0, 1}, {0, 1, 0, 1}, {1, 0, 1, 0}, {1, 1, 1, 1}};
    auto expected_decoding = vector<vector<uint8_t>>{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 1}, {0, 0, 1, 1, 0}, {0, 1, 1, 0, 0}, {0, 1, 0, 1, 0}};

    auto count = 0;
    for (auto syndrome : syndromes) {
        auto decoding = decoder->decode(syndrome);
        ASSERT_EQ(expected_decoding[count], decoding);
        count++;
    }
}


TEST(BpDecoder, MinSum_Serial_RepetitionCode5) {
    int n = 5;
    auto pcm = bp::BpSparse::New(n - 1, n);
    for (int i = 0; i < (n - 1); i++) {
        pcm->insert_entry(i, i);
        pcm->insert_entry(i, (i + 1) % n);
    }
    int max_iter = pcm->n;
    auto channel_probabilities = vector<double>(pcm->n, 0.1);

    // Initialize decoder using input arguments
    auto decoder = make_shared<bp::BpDecoder>(pcm, channel_probabilities, max_iter, 1, 1, 1);

    auto syndromes = vector<vector<uint8_t>>{{0, 0, 0, 0}, {0, 0, 0, 1}, {0, 1, 0, 1}, {1, 0, 1, 0}, {1, 1, 1, 1}};
    auto expected_decoding = vector<vector<uint8_t>>{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 1}, {0, 0, 1, 1, 0}, {0, 1, 1, 0, 0}, {0, 1, 0, 1, 0}};

    auto count = 0;
    for (auto syndrome : syndromes) {
        auto decoding = decoder->decode(syndrome);
        ASSERT_EQ(expected_decoding[count], decoding);
        count++;
    }
}


int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}