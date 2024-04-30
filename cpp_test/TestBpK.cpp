#include <gtest/gtest.h>
#include "bp.hpp"
#include "ldpc.hpp"
#include "osd.hpp"
#include "gf2codes.hpp"
#include "bp_k.hpp"
#include "sparse_matrix_util.hpp"

using namespace std;

TEST(Kruskal, ring_code){
    {
        auto rep_code = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(3);
        auto bit_weights = std::vector<int>{0,1,2};
        auto stb = ldpc::bpk::find_weighted_spanning_tree(rep_code, bit_weights);
        auto spanning_tree_bits = stb.spanning_tree_bits;
        auto not_spanning_tree_bits = stb.not_spanning_tree_bits;
        // ldpc::sparse_matrix_util::print_vector(stb);
        auto expected = std::vector<int>{0,1};
        ASSERT_EQ(spanning_tree_bits, expected);
        expected = std::vector<int>{2};
        ASSERT_EQ(not_spanning_tree_bits, expected);
    }

    {
        auto rep_code = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(3);
        auto bit_weights = std::vector<int>{1,2,0};
        auto stb = ldpc::bpk::find_weighted_spanning_tree(rep_code, bit_weights).spanning_tree_bits;
        auto expected = std::vector<int>{1,2};
        ASSERT_EQ(stb, expected);
    }
}

TEST(Kruskal, hamming_code){
    {
        auto hamming_code = ldpc::gf2codes::hamming_code<ldpc::bp::BpEntry>(3);
        auto bit_weights = std::vector<int>{0,1,2,3,4,5,6};
        auto stb = ldpc::bpk::find_weighted_spanning_tree(hamming_code, bit_weights);
        auto spanning_tree_bits = stb.spanning_tree_bits;
        auto not_spanning_tree_bits = stb.not_spanning_tree_bits;
        ldpc::sparse_matrix_util::print_sparse_matrix(hamming_code);
        ldpc::sparse_matrix_util::print_vector(spanning_tree_bits);
        ldpc::sparse_matrix_util::print_vector(not_spanning_tree_bits);
        // auto expected = std::vector<int>{0,1};
        // ASSERT_EQ(spanning_tree_bits, expected);
        // expected = std::vector<int>{2};
        // ASSERT_EQ(not_spanning_tree_bits, expected);
    }

}




TEST(Kruskal, double_ring_code){
    {
        auto double_ring_code = ldpc::gf2sparse::GF2Sparse<ldpc::bp::BpEntry>(4, 4);
        double_ring_code.insert_entry(0, 0);
        double_ring_code.insert_entry(0, 1);
        double_ring_code.insert_entry(1, 1);
        double_ring_code.insert_entry(1, 2);
        double_ring_code.insert_entry(2, 0);
        double_ring_code.insert_entry(2, 2);
        double_ring_code.insert_entry(3, 0);
        double_ring_code.insert_entry(3, 3);

        ldpc::sparse_matrix_util::print_sparse_matrix(double_ring_code);
        auto bit_weights = std::vector<int>{0,1,2,3};
        auto stb = ldpc::bpk::find_weighted_spanning_tree(double_ring_code, bit_weights).spanning_tree_bits;
        // ldpc::sparse_matrix_util::print_vector(stb);
        auto expected = std::vector<int>{0,1,3};
        ASSERT_EQ(stb, expected);
    }

    {
        auto double_ring_code = ldpc::gf2sparse::GF2Sparse<ldpc::bp::BpEntry>(4, 4);
        double_ring_code.insert_entry(0, 0);
        double_ring_code.insert_entry(0, 1);
        double_ring_code.insert_entry(1, 1);
        double_ring_code.insert_entry(1, 2);
        double_ring_code.insert_entry(1, 3);
        double_ring_code.insert_entry(2, 0);
        double_ring_code.insert_entry(2, 2);
        double_ring_code.insert_entry(3, 1);
        double_ring_code.insert_entry(3, 3);

        ldpc::sparse_matrix_util::print_sparse_matrix(double_ring_code);
        auto bit_weights = std::vector<int>{0,1,2,3};
        auto stb = ldpc::bpk::find_weighted_spanning_tree(double_ring_code, bit_weights).spanning_tree_bits;
        // ldpc::sparse_matrix_util::print_vector(stb);
        auto expected = std::vector<int>{0,1};
        ASSERT_EQ(stb, expected);
    }


}


TEST(BpKDecoder, min_sum_parallel_hamming_code) {

    int n = 3;
    auto pcm = ldpc::gf2codes::hamming_code<ldpc::bp::BpEntry>(n);

    int maximum_iterations = 1;
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

    auto syndrome = vector<uint8_t>{0, 0, 1, 1, 1, 1};
    auto decoding = decoder.decode(syndrome);
    cout<<decoder.converge<<endl;
    ldpc::sparse_matrix_util::print_vector(decoder.log_prob_ratios);

    auto bpk_decoding = ldpc::bpk::bp_k_decode(decoder, syndrome);

    cout<<decoder.converge<<endl;
    ldpc::sparse_matrix_util::print_vector(decoder.log_prob_ratios);

    // auto count = 0;
    // for (auto syndrome: syndromes) {
    //     auto decoding = decoder.decode(syndrome);
    //     ldpc::sparse_matrix_util::print_vector(decoder.log_prob_ratios);
    //     ASSERT_EQ(expected_decoding[count], decoding);
    //     count++;
    // }

}

// TEST(OsdDecoder, rep_code_test1){

//     auto rep_code = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(3);
//     auto error = vector<uint8_t>{0, 0, 1};
//     auto syndrome = rep_code.mulvec(error);
//     auto error_channel = vector<double>{0.1,0.1,0.1};
//     auto lbr = vector<double>{0, 0, 0};
//     auto decoder = ldpc::osd::OsdDecoder(rep_code,ldpc::osd::OSD_0,0,error_channel);
//     auto decoding = decoder.decode(syndrome,lbr);
//     auto syndrome2 = rep_code.mulvec(decoding);
//     ASSERT_TRUE(syndrome2 == syndrome);

// }



int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}