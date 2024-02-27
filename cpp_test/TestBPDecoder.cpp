#include <gtest/gtest.h>
#include "ldpc.hpp"
#include "bp.hpp"
#include "gf2codes.hpp"

using namespace std;

TEST(BpEntry, init) {
    auto e = ldpc::bp::BpEntry();
    ASSERT_EQ(e.row_index, -100);
    ASSERT_EQ(e.col_index, -100);
    ASSERT_EQ(e.at_end(), true);
    ASSERT_EQ(e.bit_to_check_msg, 0.0);
    ASSERT_EQ(e.bit_to_check_msg, 0.0);
}

TEST(BpSparse, init) {

    int n = 3;
    auto pcm = ldpc::bp::BpSparse(n - 1, n);
    for (int i = 0; i < (n - 1); i++) {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }

    auto expected = "1 1 0\n0 1 1";
    ASSERT_EQ(ldpc::sparse_matrix_util::print_sparse_matrix(pcm, true).str(), expected);
}

TEST(BpDecoderTest, InitializationTest) {
    // Define input arguments for initialization
    int n = 3;
    auto pcm = ldpc::bp::BpSparse(n - 1, n);
    for (int i = 0; i < (n - 1); i++) {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }
    int maximum_iterationsations = pcm.n;
    auto channel_probabilities = vector<double>(pcm.n, 0.1);

    // // Initialize decoder using input arguments
    auto decoder = ldpc::bp::BpDecoder(pcm, channel_probabilities, maximum_iterationsations);

    // Check if member variables are set correctly
    EXPECT_TRUE(pcm.gf2_equal(pcm));
    EXPECT_TRUE(pcm == decoder.pcm);
    EXPECT_EQ(pcm.m, decoder.check_count);
    EXPECT_EQ(pcm.n, decoder.bit_count);
    EXPECT_EQ(channel_probabilities, decoder.channel_probabilities);
    EXPECT_EQ(maximum_iterationsations, decoder.maximum_iterations);
    EXPECT_EQ(0.625, decoder.ms_scaling_factor);
    EXPECT_EQ(ldpc::bp::PRODUCT_SUM, decoder.bp_method);
    EXPECT_EQ(ldpc::bp::PARALLEL, decoder.schedule);
    EXPECT_EQ(1, decoder.omp_thread_count);
}

TEST(BpDecoderTest, InitializationWithOptionalParametersTest) {
    // Define input arguments for initialization
    int n = 4;
    auto pcm = ldpc::bp::BpSparse(n - 1, n);
    for (int i = 0; i < (n - 1); i++) {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }
    int maximum_iterations = 10;
    auto channel_probabilities = vector<double>{0.1, 0.2, 0.3, 0.4};

    // Define optional input parameters
    ldpc::bp::BpMethod bp_method = ldpc::bp::MINIMUM_SUM;
    double min_sum_scaling_factor = 0.5;
    ldpc::bp::BpSchedule bp_schedule = ldpc::bp::SERIAL;
    int omp_threads = 4;
    vector<int> serial_schedule{1, 3, 0, 2};
    int random_schedule = 0;

    // Initialize decoder using input arguments and optional parameters
    auto decoder = ldpc::bp::BpDecoder(pcm, channel_probabilities, maximum_iterations, bp_method, bp_schedule,
                                       min_sum_scaling_factor, omp_threads, serial_schedule, random_schedule);

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
    // EXPECT_EQ(omp_threads, omp_get_max_threads());
}

TEST(BpDecoderTest, InitialiseLogDomainBpTest) {
    // Define input arguments for initialization
    int n = 3;
    auto pcm = ldpc::bp::BpSparse(n - 1, n);
    for (int i = 0; i < (n - 1); i++) {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }
    int maximum_iterations = 100;
    auto channel_probabilities = vector<double>{0.1, 0.2, 0.3};

    // Initialize decoder using input arguments
    auto decoder = ldpc::bp::BpDecoder(pcm, channel_probabilities, maximum_iterations);

    // Call initialise_log_domain_bp() function
    decoder.initialise_log_domain_bp();

    // Check if member variables are set correctly
    for (int i = 0; i < decoder.bit_count; i++) {
        EXPECT_EQ(log((1 - channel_probabilities[i]) / channel_probabilities[i]), decoder.initial_log_prob_ratios[i]);

        for (auto &e: decoder.pcm.iterate_column(i)) {
            EXPECT_EQ(decoder.initial_log_prob_ratios[i], e.bit_to_check_msg);
        }
    }
}


TEST(BpDecoder, product_sum_parallel) {

    int n = 3;
    auto pcm = ldpc::bp::BpSparse(n - 1, n);
    for (int i = 0; i < (n - 1); i++) {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }
    int maximum_iterations = pcm.n;
    auto channel_probabilities = vector<double>(pcm.n, 0.1);

    // Initialize decoder using input arguments
    auto decoder = ldpc::bp::BpDecoder(pcm, channel_probabilities, maximum_iterations, ldpc::bp::PRODUCT_SUM,
                                       ldpc::bp::PARALLEL, 79879879);

    // Check if member variables are set correctly
    EXPECT_TRUE(pcm == decoder.pcm);
    EXPECT_EQ(pcm.m, decoder.check_count);
    EXPECT_EQ(pcm.n, decoder.bit_count);
    EXPECT_EQ(channel_probabilities, decoder.channel_probabilities);
    EXPECT_EQ(maximum_iterations, decoder.maximum_iterations);
    EXPECT_EQ(79879879, decoder.ms_scaling_factor);
    EXPECT_EQ(0, decoder.bp_method);
    EXPECT_EQ(ldpc::bp::PARALLEL, decoder.schedule);
    EXPECT_EQ(1, decoder.omp_thread_count);

    auto syndromes = vector<vector<uint8_t>>{{0, 0},
                                             {0, 1},
                                             {1, 0},
                                             {1, 1}};
    auto expected_decoding = vector<vector<uint8_t>>{{0, 0, 0},
                                                     {0, 0, 1},
                                                     {1, 0, 0},
                                                     {0, 1, 0}};

    auto count = 0;
    for (auto syndrome: syndromes) {
        auto decoding = decoder.decode(syndrome);
        ASSERT_EQ(expected_decoding[count], decoding);
        count++;
    }

}

TEST(BpDecoder, ProdSumParallel_RepetitionCode5) {
    int n = 5;
    auto pcm = ldpc::bp::BpSparse(n - 1, n);
    for (int i = 0; i < (n - 1); i++) {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }
    int maximum_iterations = pcm.n;
    auto channel_probabilities = vector<double>(pcm.n, 0.1);

    // Initialize decoder using input arguments
    auto decoder = ldpc::bp::BpDecoder(pcm, channel_probabilities, maximum_iterations, ldpc::bp::PRODUCT_SUM,
                                       ldpc::bp::PARALLEL, 4324234);

    auto syndromes = vector<vector<uint8_t>>{{0, 0, 0, 0},
                                             {0, 0, 0, 1},
                                             {0, 1, 0, 1},
                                             {1, 0, 1, 0},
                                             {1, 1, 1, 1}};
    auto expected_decoding = vector<vector<uint8_t>>{{0, 0, 0, 0, 0},
                                                     {0, 0, 0, 0, 1},
                                                     {0, 0, 1, 1, 0},
                                                     {0, 1, 1, 0, 0},
                                                     {0, 1, 0, 1, 0}};

    auto count = 0;
    for (auto syndrome: syndromes) {
        auto decoding = decoder.decode(syndrome);
        ASSERT_EQ(expected_decoding[count], decoding);
        count++;
    }
}


TEST(BpDecoder, MinSum_RepetitionCode5) {
    int n = 5;
    auto pcm = ldpc::bp::BpSparse(n - 1, n);
    for (int i = 0; i < (n - 1); i++) {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }
    int maximum_iterations = pcm.n;
    auto channel_probabilities = vector<double>(pcm.n, 0.1);

    // Initialize decoder using input arguments
    auto decoder = ldpc::bp::BpDecoder(pcm, channel_probabilities, maximum_iterations, ldpc::bp::MINIMUM_SUM,
                                       ldpc::bp::PARALLEL, 1);

    auto syndromes = vector<vector<uint8_t>>{{0, 0, 0, 0},
                                             {0, 0, 0, 1},
                                             {0, 1, 0, 1},
                                             {1, 0, 1, 0},
                                             {1, 1, 1, 1}};
    auto expected_decoding = vector<vector<uint8_t>>{{0, 0, 0, 0, 0},
                                                     {0, 0, 0, 0, 1},
                                                     {0, 0, 1, 1, 0},
                                                     {0, 1, 1, 0, 0},
                                                     {0, 1, 0, 1, 0}};

    auto count = 0;
    for (auto syndrome: syndromes) {
        auto decoding = decoder.decode(syndrome);
        ASSERT_EQ(expected_decoding[count], decoding);
        count++;
    }
}


TEST(BpDecoder, ProdSumSerial_RepetitionCode5) {
    int n = 5;
    auto pcm = ldpc::bp::BpSparse(n - 1, n);
    for (int i = 0; i < (n - 1); i++) {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }
    int maximum_iterations = pcm.n;
    auto channel_probabilities = vector<double>(pcm.n, 0.1);

    // Initialize decoder using input arguments
    auto decoder = ldpc::bp::BpDecoder(pcm, channel_probabilities, maximum_iterations, ldpc::bp::PRODUCT_SUM,
                                       ldpc::bp::SERIAL, 4324234);

    auto syndromes = vector<vector<uint8_t>>{{0, 0, 0, 0},
                                             {0, 0, 0, 1},
                                             {0, 1, 0, 1},
                                             {1, 0, 1, 0},
                                             {1, 1, 1, 1}};
    auto expected_decoding = vector<vector<uint8_t>>{{0, 0, 0, 0, 0},
                                                     {0, 0, 0, 0, 1},
                                                     {0, 0, 1, 1, 0},
                                                     {0, 1, 1, 0, 0},
                                                     {0, 1, 0, 1, 0}};

    auto count = 0;
    for (auto syndrome: syndromes) {
        auto decoding = decoder.decode(syndrome);
        ASSERT_EQ(expected_decoding[count], decoding);
        count++;
    }
}


TEST(BpDecoder, MinSum_Serial_RepetitionCode5) {
    int n = 5;
    auto pcm = ldpc::bp::BpSparse(n - 1, n);
    for (int i = 0; i < (n - 1); i++) {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }
    int maximum_iterations = pcm.n;
    auto channel_probabilities = vector<double>(pcm.n, 0.1);

    // Initialize decoder using input arguments
    auto decoder = ldpc::bp::BpDecoder(pcm, channel_probabilities, maximum_iterations, ldpc::bp::MINIMUM_SUM,
                                       ldpc::bp::SERIAL, 1);

    auto syndromes = vector<vector<uint8_t>>{{0, 0, 0, 0},
                                             {0, 0, 0, 1},
                                             {0, 1, 0, 1},
                                             {1, 0, 1, 0},
                                             {1, 1, 1, 1}};
    auto expected_decoding = vector<vector<uint8_t>>{{0, 0, 0, 0, 0},
                                                     {0, 0, 0, 0, 1},
                                                     {0, 0, 1, 1, 0},
                                                     {0, 1, 1, 0, 0},
                                                     {0, 1, 0, 1, 0}};

    auto count = 0;
    for (auto syndrome: syndromes) {
        auto decoding = decoder.decode(syndrome);
        ASSERT_EQ(expected_decoding[count], decoding);
        count++;
    }
}

TEST(BpDecoder, min_sum_parallel) {

    int n = 3;
    auto pcm = ldpc::bp::BpSparse(n - 1, n);
    for (int i = 0; i < (n - 1); i++) {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }
    int maximum_iterations = pcm.n;
    auto channel_probabilities = vector<double>(pcm.n, 0.1);

    // Initialize decoder using input arguments
    auto decoder = ldpc::bp::BpDecoder(pcm, channel_probabilities, maximum_iterations, ldpc::bp::MINIMUM_SUM,
                                       ldpc::bp::PARALLEL, 0.625);

    // Check if member variables are set correctly
    EXPECT_TRUE(pcm == decoder.pcm);
    EXPECT_EQ(pcm.m, decoder.check_count);
    EXPECT_EQ(pcm.n, decoder.bit_count);
    EXPECT_EQ(channel_probabilities, decoder.channel_probabilities);
    EXPECT_EQ(maximum_iterations, decoder.maximum_iterations);
    EXPECT_EQ(0.625, decoder.ms_scaling_factor);
    EXPECT_EQ(1, decoder.bp_method);
    EXPECT_EQ(ldpc::bp::PARALLEL, decoder.schedule);
    EXPECT_EQ(1, decoder.omp_thread_count);

    auto syndromes = vector<vector<uint8_t>>{{0, 0},
                                             {0, 1},
                                             {1, 0},
                                             {1, 1}};
    auto expected_decoding = vector<vector<uint8_t>>{{0, 0, 0},
                                                     {0, 0, 1},
                                                     {1, 0, 0},
                                                     {0, 1, 0}};

    auto count = 0;
    for (auto syndrome: syndromes) {
        auto decoding = decoder.decode(syndrome);
        ldpc::sparse_matrix_util::print_vector(decoder.log_prob_ratios);
        ASSERT_EQ(expected_decoding[count], decoding);
        count++;
    }

}

TEST(BpDecoder, min_sum_single_scan) {

    int n = 3;
    auto pcm = ldpc::bp::BpSparse(n - 1, n);
    for (int i = 0; i < (n - 1); i++) {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }
    int maximum_iterations = pcm.n;
    auto channel_probabilities = vector<double>(pcm.n, 0.1);

    // Initialize decoder using input arguments
    auto decoder = ldpc::bp::BpDecoder(pcm, channel_probabilities, maximum_iterations, ldpc::bp::MINIMUM_SUM,
                                       ldpc::bp::PARALLEL, 0.625);

    // Check if member variables are set correctly
    EXPECT_TRUE(pcm == decoder.pcm);
    EXPECT_EQ(pcm.m, decoder.check_count);
    EXPECT_EQ(pcm.n, decoder.bit_count);
    EXPECT_EQ(channel_probabilities, decoder.channel_probabilities);
    EXPECT_EQ(maximum_iterations, decoder.maximum_iterations);
    EXPECT_EQ(0.625, decoder.ms_scaling_factor);
    EXPECT_EQ(1, decoder.bp_method);
    EXPECT_EQ(ldpc::bp::PARALLEL, decoder.schedule);
    EXPECT_EQ(1, decoder.omp_thread_count);

    auto syndromes = vector<vector<uint8_t>>{{0, 0},
                                             {0, 1},
                                             {1, 0},
                                             {1, 1}};
    auto expected_decoding = vector<vector<uint8_t>>{{0, 0, 0},
                                                     {0, 0, 1},
                                                     {1, 0, 0},
                                                     {0, 1, 0}};

    auto count = 0;
    for (auto syndrome: syndromes) {
        auto decoding = decoder.bp_decode_single_scan(syndrome);
        ldpc::sparse_matrix_util::print_vector(decoder.log_prob_ratios);
        ASSERT_EQ(expected_decoding[count], decoding);
        count++;
    }

}

TEST(BpDecoder, min_sum_relative_schedule) {
    int n = 3;
    auto pcm = ldpc::bp::BpSparse(n - 1, n);
    for (int i = 0; i < (n - 1); i++) {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }
    int maximum_iterations = pcm.n;
    auto channel_probabilities = vector<double>(pcm.n, 0.1);

    // Initialize decoder using input arguments
    auto decoder = ldpc::bp::BpDecoder(pcm, channel_probabilities, maximum_iterations, ldpc::bp::MINIMUM_SUM,
                                       ldpc::bp::SERIAL_RELATIVE, 0.625);

    // Check if member variables are set correctly
    EXPECT_TRUE(pcm == decoder.pcm);
    EXPECT_EQ(pcm.m, decoder.check_count);
    EXPECT_EQ(pcm.n, decoder.bit_count);
    EXPECT_EQ(channel_probabilities, decoder.channel_probabilities);
    EXPECT_EQ(maximum_iterations, decoder.maximum_iterations);
    EXPECT_EQ(0.625, decoder.ms_scaling_factor);
    EXPECT_EQ(1, decoder.bp_method);
    EXPECT_EQ(ldpc::bp::SERIAL_RELATIVE, decoder.schedule);
    EXPECT_EQ(1, decoder.omp_thread_count);

    auto syndromes = vector<vector<uint8_t>>{{0, 0},
                                             {0, 1},
                                             {1, 0},
                                             {1, 1}};
    auto expected_decoding = vector<vector<uint8_t>>{{0, 0, 0},
                                                     {0, 0, 1},
                                                     {1, 0, 0},
                                                     {0, 1, 0}};
    auto count = 0;
    for (auto syndrome: syndromes) {
        auto decoding = decoder.bp_decode_serial(syndrome);
        ldpc::sparse_matrix_util::print_vector(decoder.log_prob_ratios);
        ASSERT_EQ(expected_decoding[count], decoding);
        count++;
    }
}

TEST(BpDecoder, random_schedule_seed) {
    {// todo why are there 3 bits but channel probabilities has length 4?
        auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(3);
        auto bpd = ldpc::bp::BpDecoder(pcm, vector<double>{0.1, 0.2, 0.3, 0.4},
                                       100, ldpc::bp::MINIMUM_SUM, ldpc::bp::SERIAL,
                                       0.625, 1, vector<int>{2, 3, 1},
                                       1234);
        auto expected_order = vector<int>{2, 3, 1};
        ASSERT_EQ(bpd.random_schedule_seed, -1);
        ASSERT_EQ(expected_order, bpd.serial_schedule_order);

    }

    {
        auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(3);
        auto bpd = ldpc::bp::BpDecoder(pcm, vector<double>{0.1, 0.2, 0.3, 0.4}, 100, ldpc::bp::MINIMUM_SUM,
                                       ldpc::bp::SERIAL, 0.625, 1, ldpc::bp::NULL_INT_VECTOR, 0);
        auto expected_order = vector<int>{0, 1, 2};
        ASSERT_EQ(bpd.random_schedule_seed, 0);
        ASSERT_EQ(expected_order, bpd.serial_schedule_order);

    }

    {
        auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(3);
        auto bpd = ldpc::bp::BpDecoder(pcm, vector<double>{0.1, 0.2, 0.3, 0.4},
                                       100, ldpc::bp::MINIMUM_SUM, ldpc::bp::SERIAL,
                                       0.625, 1, ldpc::bp::NULL_INT_VECTOR, 4);
        auto expected_order = vector<int>{0, 1, 2};
        ASSERT_EQ(bpd.random_schedule_seed, 4);
        // ASSERT_EQ(expected_order, bpd.serial_schedule_order);

    }

}

TEST(BpDecoder, relative_serial_schedule_order) {
    { // should order correctly
        auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(3);
        auto bpd = ldpc::bp::BpDecoder(pcm, vector<double>{0.2, 0.3, 0.1},
                                       1, ldpc::bp::MINIMUM_SUM, ldpc::bp::SERIAL_RELATIVE,
                                       0.625, 1, ldpc::bp::NULL_INT_VECTOR,
                                       -1); // todo what should be default here?
        auto dummy_syndr = std::vector<uint8_t>{0, 0};
        bpd.decode(dummy_syndr);
        auto expected_order = vector<int>{2, 0, 1};
        ASSERT_EQ(bpd.random_schedule_seed, -1);
        ASSERT_EQ(expected_order, bpd.serial_schedule_order);

    }
    {// should overwrite initial schedule
        auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(3);
        auto bpd = ldpc::bp::BpDecoder(pcm, vector<double>{0.3, 0.2, 0.1}, 1,
                                       ldpc::bp::MINIMUM_SUM, ldpc::bp::SERIAL_RELATIVE,
                                       0.625, 1, vector<int>{0, 1, 2},
                                       -1);
        auto dummy_syndr = std::vector<uint8_t>{0, 0};
        bpd.decode(dummy_syndr);
        auto expected_order = vector<int>{2, 1, 0};
        ASSERT_EQ(bpd.random_schedule_seed, -1);
        ASSERT_EQ(expected_order, bpd.serial_schedule_order);

    }
    {// should order according to LLRs
        auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(3);
        auto bpd = ldpc::bp::BpDecoder(pcm, vector<double>{0.1, 0.01, 0.01}, 2,
                                       ldpc::bp::MINIMUM_SUM, ldpc::bp::SERIAL_RELATIVE,
                                       0.625, 1, vector<int>{0, 1, 2},
                                       -1);
        auto dummy_syndr = std::vector<uint8_t>{1, 0};
        bpd.decode(dummy_syndr);
        auto expected_order = vector<int>{1, 2, 0};
        ASSERT_EQ(bpd.random_schedule_seed, -1);
        ASSERT_EQ(expected_order, bpd.serial_schedule_order);

    }
}

//received vector decoding
TEST(BpDecoder, ProdSumSerial_RepCode5) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    int maximum_iterations = pcm.n;
    auto channel_probabilities = vector<double>(pcm.n, 0.1);

    // Initialize decoder using input arguments
    auto decoder = ldpc::bp::BpDecoder(pcm, channel_probabilities, maximum_iterations, ldpc::bp::PRODUCT_SUM,
                                       ldpc::bp::SERIAL, 4324234, ldpc::bp::AUTO);

    auto received_vectors = vector<vector<uint8_t>>{{0, 0, 0, 0, 1}, {0, 1, 1, 0, 0}, {1, 0, 0, 1, 1}};
    auto expected_decoding = vector<vector<uint8_t>>{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}}; // todo I'm not sure I understand these cases

    auto count = 0;
    for (auto received_vector: received_vectors) {
        auto decoding = decoder.decode(received_vector);
        ASSERT_EQ(expected_decoding[count], decoding);
        count++;
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}