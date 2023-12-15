#include <gtest/gtest.h>
#include "ldpc.hpp"
#include "gf2codes.hpp"
#include "union_find.hpp"
#include "util.hpp"
#include "bp.hpp"

using namespace std;
using namespace ldpc::uf;
using namespace ldpc::sparse_matrix_util;

// TEST(UfDecoder, single_bit_error) {

//     auto pcm1 = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(10);
//     // ldpc::sparse_matrix_util::print_sparse_matrix(pcm1);

//     auto ufd = UfDecoder(pcm1);

//     for(int i=0; i<pcm1.n; i++){

//         auto syndrome = vector<uint8_t>(pcm1.n,0);
//         syndrome[i % pcm1.n] = 1;
//         syndrome[(i+1) % pcm1.n] = 1;

//         auto decoding = ufd.peel_decode(syndrome);

//         auto expected_decoding = vector<uint8_t>(pcm1.n,0);
//         expected_decoding[(i+1) % pcm1.n] = 1;

//         ASSERT_EQ(decoding,expected_decoding);

//     }

// }


// TEST(UfDecoder, weighted_cluster_growth) {

//     auto pcm1 = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(7);

//     auto ufd = UfDecoder(pcm1);

//     auto syndrome = vector<uint8_t>(pcm1.n,0);
//     syndrome[0] = 1;
//     syndrome[1] = 1;

//     auto bit_weights = vector<double>{-1,100,-1,-1,-1,-1,-1};
//     auto decoding = ufd.peel_decode(syndrome, bit_weights);
//     auto expected_decoding = vector<uint8_t>{1,0,1,1,1,1,1};
//     ASSERT_EQ(decoding,expected_decoding);


// }

// // TEST(UfDecoder, HammingCode){

// //     int m = 3;

// //     auto pcm = ldpc::gf2codes::hamming_code(m);

// //     auto ufd = UfDecoder(pcm);
// //     auto error_channel = std::vector<double>(pcm.n,0.1);
// //     auto bpd = ldpc::bp::BpDecoder(pcm,error_channel,pcm.n,ldpc::bp::MINIMUM_SUM,ldpc::bp::PARALLEL,0.9);

// //     // auto syndrome = vector<uint8_t>(pcm.n,0);

// //     for(int i = 0; i < std::pow(2,m); i++){

// //         ldpc::sparse_matrix_util::print_vector(ldpc::util::decimal_to_binary(i,m));

// //         auto syndrome = ldpc::util::decimal_to_binary(i,m);
// //         bpd.decode(syndrome);
// //         auto decoding = ufd.bit_cluster_decode(syndrome,bpd.log_prob_ratios,1,3);

// //         auto decoding_syndrome = pcm.mulvec(decoding);
// //         ASSERT_EQ(decoding_syndrome,syndrome);
// //         std::cout<<"HEllo55"<<std::endl;


// //     }

// // }


// TEST(UfDecoder, HammingCode2){

//     int m = 5;

//     auto pcm = ldpc::gf2codes::hamming_code<ldpc::bp::BpEntry>(m);

//     auto ufd = UfDecoder(pcm);

//     // auto syndrome = vector<uint8_t>(pcm.n,0);

//     for(int i = 0; i < std::pow(2,m); i++){

//         // ldpc::sparse_matrix_util::print_vector(ldpc::util::decimal_to_binary(i,m));

//         auto syndrome = ldpc::util::decimal_to_binary(i,m);
//         auto decoding = ufd.matrix_decode(syndrome);
//         auto decoding_syndrome = pcm.mulvec(decoding);
//         ASSERT_EQ(decoding_syndrome,syndrome);

//     }

// }

TEST(UfDecoder, ring_code3) {

    auto pcm = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(5);
    auto bpd = UfDecoder(pcm);

    auto syndrome = vector<uint8_t>{1, 0, 1};

    auto decoding = bpd.peel_decode(syndrome, ldpc::uf::NULL_DOUBLE_VECTOR, 3);


}

TEST(UfDecoder, on_the_fly_small_hamming) {
    int m = 5;

    // todo this is a mess and should be replaced with parameterized tests
    for (int i = 0; i < std::pow(2, m); i++) {
        auto pcm = ldpc::gf2codes::hamming_code<ldpc::bp::BpEntry>(m);
        auto bp = ldpc::bp::BpDecoder(pcm, std::vector<double>(pcm.n, 0.1));
        bp.maximum_iterations = 2;
        auto ufd = UfDecoder(pcm);
        auto syndrome = ldpc::util::decimal_to_binary(i, m);
        bp.decode(syndrome);
        auto decoding = ufd.on_the_fly_decode(syndrome, bp.log_prob_ratios);
        auto decoding_syndrome = pcm.mulvec(decoding);
        ASSERT_EQ(decoding_syndrome, syndrome);
    }
}


TEST(UfDecoder, on_the_fly_hamming_higher_weight_syndrome) {
    int m = 5;

    // todo this is a mess and should be replaced with parameterized tests
    for (int i = 0; i < std::pow(2, m); i++) {
        auto pcm = ldpc::gf2codes::hamming_code<ldpc::bp::BpEntry>(m);
        auto bp = ldpc::bp::BpDecoder(pcm, std::vector<double>(pcm.n, 0.1));
        bp.maximum_iterations = 2;
        auto ufd = UfDecoder(pcm);
        auto syndrome = ldpc::util::decimal_to_binary(i + 1, m);
        bp.decode(syndrome);
        auto decoding = ufd.on_the_fly_decode(syndrome, bp.log_prob_ratios);
        auto decoding_syndrome = pcm.mulvec(decoding);
        ASSERT_EQ(decoding_syndrome, syndrome);
    }
}

TEST(UfDecoder, on_the_fly_ring_code) {
    auto size = 5;
    auto pcm = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(size);
    auto bp = ldpc::bp::BpDecoder(pcm, std::vector<double>(pcm.n, 0.1));
    bp.maximum_iterations = 2;
    auto ufd = UfDecoder(pcm);

    // todo check these cases/construct correct ones
    auto received_vectors = vector<vector<uint8_t>>{
            {0, 1, 1, 0, 0},
            {1, 1, 1, 0, 0},
            {1, 0, 0, 1, 0}
    };
    auto expected_decoding = vector<vector<uint8_t>>{
            {0, 0, 1, 0, 0},
            {0, 1, 0, 1, 0},
            {1, 0, 0, 1, 0}
    };

    auto count = 0;
    for (auto received_vector: received_vectors) {
        bp.decode(received_vector);
        auto decoding = ufd.on_the_fly_decode(received_vector, bp.log_prob_ratios);
        ASSERT_EQ(expected_decoding[count++], decoding);
    }
}

TEST(UfDecoder, otf_hamming_code_rank9) {

    auto hamming_code_rank = 9;

    auto pcm = ldpc::gf2codes::hamming_code<ldpc::bp::BpEntry>(hamming_code_rank);
    auto bp = ldpc::bp::BpDecoder(pcm, std::vector<double>(pcm.n, 0.1));
    bp.maximum_iterations = 2;
    auto ufd = UfDecoder(pcm);

    for (int i = 1; i < std::pow(2, hamming_code_rank); i++) {
        auto syndrome = ldpc::util::decimal_to_binary(i, hamming_code_rank);
        bp.decode(syndrome);
        auto decoding = ufd.on_the_fly_decode(syndrome, bp.log_prob_ratios);

        auto decoding_syndrome = pcm.mulvec(decoding);

        cout<<i<<endl;
        print_vector(syndrome);
        ASSERT_TRUE(syndrome == decoding_syndrome);
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}