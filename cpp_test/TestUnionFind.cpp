#include <gtest/gtest.h>
#include "ldpc.hpp"
#include "gf2codes.hpp"
#include "union_find.hpp"
#include "util.hpp"
#include "bp.hpp"
#include "robin_set.h"

using namespace std;
using namespace ldpc::uf;

TEST(UfDecoder, single_bit_error) {

    auto pcm1 = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(10);
    // ldpc::sparse_matrix_util::print_sparse_matrix(pcm1);

    auto ufd = UfDecoder(pcm1);

    for(int i=0; i<pcm1.n; i++){

        auto syndrome = vector<uint8_t>(pcm1.n,0);
        syndrome[i % pcm1.n] = 1;
        syndrome[(i+1) % pcm1.n] = 1;

        auto decoding = ufd.peel_decode(syndrome);

        auto expected_decoding = vector<uint8_t>(pcm1.n,0);
        expected_decoding[(i+1) % pcm1.n] = 1;

        ASSERT_EQ(decoding,expected_decoding);

    }

}


TEST(UfDecoder, weighted_cluster_growth) {

    auto pcm1 = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(7);

    auto ufd = UfDecoder(pcm1);

    auto syndrome = vector<uint8_t>(pcm1.n,0);
    syndrome[0] = 1;
    syndrome[1] = 1;

    auto bit_weights = vector<double>{-1,100,-1,-1,-1,-1,-1};
    auto decoding = ufd.peel_decode(syndrome, bit_weights);
    auto expected_decoding = vector<uint8_t>{1,0,1,1,1,1,1};
    ASSERT_EQ(decoding,expected_decoding);
  

}

// TEST(UfDecoder, HammingCode){

//     int m = 3;

//     auto pcm = ldpc::gf2codes::hamming_code(m);

//     auto ufd = UfDecoder(pcm);
//     auto error_channel = std::vector<double>(pcm.n,0.1);
//     auto bpd = ldpc::bp::BpDecoder(pcm,error_channel,pcm.n,ldpc::bp::MINIMUM_SUM,ldpc::bp::PARALLEL,0.9);

//     // auto syndrome = vector<uint8_t>(pcm.n,0);

//     for(int i = 0; i < std::pow(2,m); i++){

//         ldpc::sparse_matrix_util::print_vector(ldpc::util::decimal_to_binary(i,m));

//         auto syndrome = ldpc::util::decimal_to_binary(i,m);
//         bpd.decode(syndrome);
//         auto decoding = ufd.bit_cluster_decode(syndrome,bpd.log_prob_ratios,1,3);

//         auto decoding_syndrome = pcm.mulvec(decoding);
//         ASSERT_EQ(decoding_syndrome,syndrome);
//         std::cout<<"HEllo55"<<std::endl;


//     }

// }


TEST(UfDecoder, HammingCode2){

    int m = 5;

    auto pcm = ldpc::gf2codes::hamming_code<ldpc::bp::BpEntry>(m);

    auto ufd = UfDecoder(pcm);

    // auto syndrome = vector<uint8_t>(pcm.n,0);

    for(int i = 0; i < std::pow(2,m); i++){

        // ldpc::sparse_matrix_util::print_vector(ldpc::util::decimal_to_binary(i,m));

        auto syndrome = ldpc::util::decimal_to_binary(i,m);
        auto decoding = ufd.matrix_decode(syndrome);
        auto decoding_syndrome = pcm.mulvec(decoding);
        ASSERT_EQ(decoding_syndrome,syndrome);

    }

}

TEST(UfDecoder, ring_code3){

    auto pcm = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(5);
    auto bpd = UfDecoder(pcm);

    auto syndrome = vector<uint8_t>{1, 0, 1};

    auto decoding = bpd.peel_decode(syndrome, ldpc::uf::EMPTY_DOUBLE_VECTOR, 3);

    ASSERT_TRUE(bpd.pcm_max_bit_degree_2);
    tsl::robin_set<int> boundary_bits = {};
    ASSERT_EQ(bpd.planar_code_boundary_bits,boundary_bits);

}


TEST(UfDecoder, rep_code){

    for(int n = 5; n < 20; n++){

        auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
        auto bpd = UfDecoder(pcm);
        ASSERT_TRUE(bpd.pcm_max_bit_degree_2);
        tsl::robin_set<int> boundary_bits = {0,n-1};
        ASSERT_EQ(bpd.planar_code_boundary_bits,boundary_bits);
        auto syndrome = vector<uint8_t>(n,0);
        syndrome[0] = 1;
        syndrome[n-2] = 1;
        auto decoding = bpd.peel_decode(syndrome, ldpc::uf::EMPTY_DOUBLE_VECTOR, 1);
        auto expected_decoding = vector<uint8_t>(n,0);
        expected_decoding[0] = 1;
        expected_decoding[n-1] = 1;
        ASSERT_EQ(decoding,expected_decoding);

    }

}

TEST(UfDecoder, peeling_with_boundaries){


    auto pcm = ldpc::gf2sparse::GF2Sparse<ldpc::bp::BpEntry>(2,7);
    pcm.insert_entry(0,0);
    pcm.insert_entry(0,1);
    pcm.insert_entry(0,2);
    pcm.insert_entry(0,3);
    pcm.insert_entry(1,3);
    pcm.insert_entry(1,4);
    pcm.insert_entry(1,5);
    pcm.insert_entry(1,6);

    // ldpc::sparse_matrix_util::print_sparse_matrix(pcm);

    auto bpd = UfDecoder(pcm);
    ASSERT_TRUE(bpd.pcm_max_bit_degree_2);
    tsl::robin_set<int> boundary_bits = {0,1,2,4,5,6};
    ASSERT_EQ(bpd.planar_code_boundary_bits,boundary_bits);
    auto syndrome = vector<uint8_t>{1,1};
    auto weights = vector<double>{-1,0,0,10,0,0,-1};
    auto decoding = bpd.peel_decode(syndrome, weights,1);
    
    // ldpc::sparse_matrix_util::print_vector(decoding);
    auto expected_decoding = vector<uint8_t>(7,0);
    expected_decoding[0] = 1;
    expected_decoding[6] = 1;
    ASSERT_EQ(decoding,expected_decoding);

    

}

TEST(UfDecoder, peeling_with_boundaries_edge_case){

    auto pcm = ldpc::gf2sparse::GF2Sparse<ldpc::bp::BpEntry>(20,41);
    pcm.insert_entry(0,0);pcm.insert_entry(0,5);pcm.insert_entry(0,25);pcm.insert_entry(1,1);pcm.insert_entry(1,6);pcm.insert_entry(1,25);pcm.insert_entry(1,26);pcm.insert_entry(2,2);pcm.insert_entry(2,7);pcm.insert_entry(2,26);pcm.insert_entry(2,27);pcm.insert_entry(3,3);pcm.insert_entry(3,8);pcm.insert_entry(3,27);pcm.insert_entry(3,28);pcm.insert_entry(4,4);pcm.insert_entry(4,9);pcm.insert_entry(4,28);pcm.insert_entry(5,5);pcm.insert_entry(5,10);pcm.insert_entry(5,29);pcm.insert_entry(6,6);pcm.insert_entry(6,11);pcm.insert_entry(6,29);pcm.insert_entry(6,30);pcm.insert_entry(7,7);pcm.insert_entry(7,12);pcm.insert_entry(7,30);pcm.insert_entry(7,31);pcm.insert_entry(8,8);pcm.insert_entry(8,13);pcm.insert_entry(8,31);pcm.insert_entry(8,32);pcm.insert_entry(9,9);pcm.insert_entry(9,14);pcm.insert_entry(9,32);pcm.insert_entry(10,10);pcm.insert_entry(10,15);pcm.insert_entry(10,33);pcm.insert_entry(11,11);pcm.insert_entry(11,16);pcm.insert_entry(11,33);pcm.insert_entry(11,34);pcm.insert_entry(12,12);pcm.insert_entry(12,17);pcm.insert_entry(12,34);pcm.insert_entry(12,35);pcm.insert_entry(13,13);pcm.insert_entry(13,18);pcm.insert_entry(13,35);pcm.insert_entry(13,36);pcm.insert_entry(14,14);pcm.insert_entry(14,19);pcm.insert_entry(14,36);pcm.insert_entry(15,15);pcm.insert_entry(15,20);pcm.insert_entry(15,37);pcm.insert_entry(16,16);pcm.insert_entry(16,21);pcm.insert_entry(16,37);pcm.insert_entry(16,38);pcm.insert_entry(17,17);pcm.insert_entry(17,22);pcm.insert_entry(17,38);pcm.insert_entry(17,39);pcm.insert_entry(18,18);pcm.insert_entry(18,23);pcm.insert_entry(18,39);pcm.insert_entry(18,40);pcm.insert_entry(19,19);pcm.insert_entry(19,24);pcm.insert_entry(19,40);

    auto error_rate = std::vector<double>(41,0.18);
    auto bpd = ldpc::bp::BpDecoder(pcm, error_rate, 1, ldpc::bp::MINIMUM_SUM, ldpc::bp::PARALLEL, 0.625);

    auto syndrome_sparse = std::vector<int>{3,  5,  7,  8, 12, 17, 18, 19};

    auto syndrome = std::vector<uint8_t>(pcm.m,0);
    for(int i : syndrome_sparse){
        syndrome[i] = 1;
    }

    bpd.decode(syndrome);

    ASSERT_FALSE(bpd.converge);

    auto ufd = UfDecoder(pcm);

    // ldpc::sparse_matrix_util::print_vector(bpd.log_prob_ratios);

    auto decoding = ufd.peel_decode(syndrome, bpd.log_prob_ratios,1);

    auto decoding_syndrome = pcm.mulvec(decoding);

    ASSERT_EQ(decoding_syndrome,syndrome);



}



int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}