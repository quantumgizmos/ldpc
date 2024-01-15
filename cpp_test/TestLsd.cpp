#include <gtest/gtest.h>
#include "ldpc.hpp"
#include "gf2codes.hpp"
#include "lsd.hpp"
#include "util.hpp"
#include "bp.hpp"

using namespace std;
using namespace ldpc::lsd;
using namespace ldpc::sparse_matrix_util;


TEST(LsdCluster, init1){

    auto pcm = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(10);
    auto gbm = new ldpc::lsd::LsdCluster *[pcm.n](); //global bit dictionary
    auto gcm = new ldpc::lsd::LsdCluster *[pcm.m](); //global check dictionary

    auto syndrome_index = 0;
    auto cl = ldpc::lsd::LsdCluster(pcm, syndrome_index, gcm, gbm);
    
    ASSERT_TRUE(cl.active);
    ASSERT_FALSE(cl.valid);

    auto expected_bit_nodes = tsl::robin_set<int>{};
    auto expected_check_nodes = tsl::robin_set<int>{syndrome_index};
    auto expected_boundary_check_nodes = tsl::robin_set<int>{syndrome_index};
    auto expected_enclosed_syndromes = tsl::robin_set<int>{syndrome_index};
    auto expected_cluster_check_idx_to_pcm_check_idx = std::vector<int>{syndrome_index};
    auto expected_pcm_check_idx_to_cluster_check_idx = tsl::robin_map<int, int>{{syndrome_index, 0}};

    ASSERT_EQ(expected_bit_nodes, cl.bit_nodes);
    ASSERT_EQ(expected_check_nodes, cl.check_nodes);
    ASSERT_EQ(expected_boundary_check_nodes, cl.boundary_check_nodes);
    ASSERT_EQ(expected_enclosed_syndromes, cl.enclosed_syndromes);
    ASSERT_EQ(gcm[syndrome_index], &cl);
    ASSERT_EQ(expected_cluster_check_idx_to_pcm_check_idx, cl.cluster_check_idx_to_pcm_check_idx);
    ASSERT_EQ(expected_pcm_check_idx_to_cluster_check_idx, cl.pcm_check_idx_to_cluster_check_idx);

    delete gbm;
    delete gcm;

}


TEST(LsdCluster, add_bitANDadd_check_add){

    auto pcm = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(10);
    auto gbm = new ldpc::lsd::LsdCluster *[pcm.n]; //global bit dictionary
    auto gcm = new ldpc::lsd::LsdCluster *[pcm.m]; //global check dictionary

    auto syndrome_index = 1;
    auto cl = ldpc::lsd::LsdCluster(pcm, syndrome_index, gcm, gbm);
    
    cl.compute_growth_candidate_bit_nodes();
    auto expected_candidate_bit_nodes = std::vector<int>{1,2};
    ASSERT_EQ(expected_candidate_bit_nodes, cl.candidate_bit_nodes);


    cl.add_bit(expected_candidate_bit_nodes[1]);
    cl.add_check(2,true);
    
    
    auto expected_bit_nodes = tsl::robin_set<int>{2};
    auto expected_check_nodes = tsl::robin_set<int>{syndrome_index,2};
    auto expected_cluster_check_idx_to_pcm_check_idx = std::vector<int>{1,2};
    auto expected_pcm_check_idx_to_cluster_check_idx = tsl::robin_map<int, int>{{1, 0},{2,1}};
    
    ASSERT_EQ(expected_bit_nodes, cl.bit_nodes);
    ASSERT_EQ(cl.global_bit_membership[2], &cl);
    ASSERT_EQ(cl.cluster_bit_idx_to_pcm_bit_idx[0], 2);    
    ASSERT_EQ(expected_check_nodes, cl.check_nodes);
    ASSERT_EQ(expected_cluster_check_idx_to_pcm_check_idx, cl.cluster_check_idx_to_pcm_check_idx);
    ASSERT_EQ(expected_pcm_check_idx_to_cluster_check_idx, cl.pcm_check_idx_to_cluster_check_idx);
    ASSERT_EQ(cl.global_check_membership[1], &cl);
    ASSERT_EQ(cl.global_check_membership[2], &cl);


    // Test adding existing checks and bits
    cl.add_bit(expected_candidate_bit_nodes[1]);
    cl.add_check(2,true);

    ASSERT_EQ(expected_bit_nodes, cl.bit_nodes);
    ASSERT_EQ(cl.global_bit_membership[2], &cl);
    ASSERT_EQ(cl.cluster_bit_idx_to_pcm_bit_idx[0], 2);    
    ASSERT_EQ(expected_check_nodes, cl.check_nodes);
    ASSERT_EQ(expected_cluster_check_idx_to_pcm_check_idx, cl.cluster_check_idx_to_pcm_check_idx);
    ASSERT_EQ(expected_pcm_check_idx_to_cluster_check_idx, cl.pcm_check_idx_to_cluster_check_idx);
    ASSERT_EQ(cl.global_check_membership[1], &cl);
    ASSERT_EQ(cl.global_check_membership[2], &cl);

    //check that bit is remove from boundary check node is removed from boundary check nodes
    cl.compute_growth_candidate_bit_nodes();
    expected_candidate_bit_nodes = std::vector<int>{1,3};
    ASSERT_EQ(expected_candidate_bit_nodes, cl.candidate_bit_nodes);

    //add bit 3, verify that boundary check 2 is removed from the boundary check list
    cl.add_bit(3);
    cl.compute_growth_candidate_bit_nodes();
    auto expected_boundary_check_nodes = tsl::robin_set<int>{1};
    ASSERT_EQ(expected_boundary_check_nodes, cl.boundary_check_nodes);

    delete gbm;
    delete gcm;

}

TEST(LsdCluster, add_bit_node_to_cluster){


    auto pcm = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(10);
    auto gbm = new ldpc::lsd::LsdCluster *[pcm.n](); //global bit dictionary
    auto gcm = new ldpc::lsd::LsdCluster *[pcm.m](); //global check dictionary

    auto syndrome_index = 1;
    auto cl = ldpc::lsd::LsdCluster(pcm, syndrome_index, gcm, gbm);
    
    cl.compute_growth_candidate_bit_nodes();
    auto expected_candidate_bit_nodes = std::vector<int>{1,2};
    ASSERT_EQ(expected_candidate_bit_nodes, cl.candidate_bit_nodes);


    auto bit_membership = cl.global_bit_membership[0];

    // add bit 2 to the cluster
    cl.add_bit_node_to_cluster(2);

    auto expected_bit_nodes = tsl::robin_set<int>{2};
    auto expected_check_nodes = tsl::robin_set<int>{1,2};
    auto expected_cluster_check_idx_to_pcm_check_idx = std::vector<int>{1,2};
    auto expected_pcm_check_idx_to_cluster_check_idx = tsl::robin_map<int, int>{{1, 0},{2,1}};
    
    ASSERT_EQ(expected_bit_nodes, cl.bit_nodes);
    ASSERT_EQ(cl.global_bit_membership[2], &cl);
    ASSERT_EQ(cl.cluster_bit_idx_to_pcm_bit_idx[0], 2);    
    ASSERT_EQ(expected_check_nodes, cl.check_nodes);
    ASSERT_EQ(expected_cluster_check_idx_to_pcm_check_idx, cl.cluster_check_idx_to_pcm_check_idx);
    ASSERT_EQ(expected_pcm_check_idx_to_cluster_check_idx, cl.pcm_check_idx_to_cluster_check_idx);
    ASSERT_EQ(cl.global_check_membership[1], &cl);
    ASSERT_EQ(cl.global_check_membership[2], &cl);

    cl.compute_growth_candidate_bit_nodes();
    auto expected_boundary_check_nodes = tsl::robin_set<int>{1,2};
    expected_candidate_bit_nodes = std::vector<int>{1,3};

    ASSERT_EQ(expected_boundary_check_nodes, cl.boundary_check_nodes);
    ASSERT_EQ(expected_candidate_bit_nodes, cl.candidate_bit_nodes);

    //check the cluster pcm
    auto expected_column = std::vector<int>{0,1};
    ASSERT_TRUE(cl.cluster_pcm.size() == 1);
    ASSERT_EQ(expected_column, cl.cluster_pcm[0]);


    //add bit 3, verify that check 1 is removed from the boundary check list
    cl.add_bit_node_to_cluster(1);
    
    cl.compute_growth_candidate_bit_nodes();
    expected_boundary_check_nodes = tsl::robin_set<int>{0,2};
    expected_candidate_bit_nodes = std::vector<int>{0,3};
    ASSERT_EQ(expected_boundary_check_nodes, cl.boundary_check_nodes);
    ASSERT_EQ(expected_candidate_bit_nodes, cl.candidate_bit_nodes);


    expected_bit_nodes = tsl::robin_set<int>{1,2};
    expected_check_nodes = tsl::robin_set<int>{0,1,2};
    expected_cluster_check_idx_to_pcm_check_idx = std::vector<int>{1,2,0};
    expected_pcm_check_idx_to_cluster_check_idx = tsl::robin_map<int, int>{{1, 0},{2,1},{0,2}};
    
    ASSERT_EQ(expected_bit_nodes, cl.bit_nodes);
    ASSERT_EQ(cl.global_bit_membership[1], &cl);
    ASSERT_EQ(cl.cluster_bit_idx_to_pcm_bit_idx[0], 2);    
    ASSERT_EQ(expected_check_nodes, cl.check_nodes);
    ASSERT_EQ(expected_cluster_check_idx_to_pcm_check_idx, cl.cluster_check_idx_to_pcm_check_idx);
    ASSERT_EQ(expected_pcm_check_idx_to_cluster_check_idx, cl.pcm_check_idx_to_cluster_check_idx);
    ASSERT_EQ(cl.global_check_membership[0], &cl);
    ASSERT_EQ(cl.global_check_membership[2], &cl);

    //check the cluster pcm
    expected_column = std::vector<int>{0,1};
    ASSERT_TRUE(cl.cluster_pcm.size() == 2);
    ASSERT_EQ(expected_column, cl.cluster_pcm[0]);

    expected_column = std::vector<int>{2,0};
    ASSERT_EQ(expected_column, cl.cluster_pcm[1]);

    delete gbm;
    delete gcm;

}


TEST(LsdCluster, grow_cluster){


    auto pcm = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(10);
    auto gbm = new ldpc::lsd::LsdCluster *[pcm.n](); //global bit dictionary
    auto gcm = new ldpc::lsd::LsdCluster *[pcm.m](); //global check dictionary

    auto syndrome_index = 5;
    auto cl = ldpc::lsd::LsdCluster(pcm, syndrome_index, gcm, gbm);
    
    cl.compute_growth_candidate_bit_nodes();
    auto expected_candidate_bit_nodes = std::vector<int>{5,6};
    ASSERT_EQ(expected_candidate_bit_nodes, cl.candidate_bit_nodes);
    auto bit_membership = cl.global_bit_membership[5];
    ASSERT_EQ(bit_membership, nullptr);

    cl.grow_cluster();

    auto expected_bit_nodes = tsl::robin_set<int>{5,6};
    auto expected_check_nodes = tsl::robin_set<int>{5,4,6};
    auto expected_cluster_check_idx_to_pcm_check_idx = std::vector<int>{5,4,6};
    auto expected_pcm_check_idx_to_cluster_check_idx = tsl::robin_map<int, int>{{5, 0},{4,1},{6,2}};
    
    ASSERT_EQ(expected_bit_nodes, cl.bit_nodes);
    ASSERT_EQ(cl.global_bit_membership[5], &cl);
    ASSERT_EQ(cl.cluster_bit_idx_to_pcm_bit_idx[0], 5);    
    ASSERT_EQ(expected_check_nodes, cl.check_nodes);
    ASSERT_EQ(expected_cluster_check_idx_to_pcm_check_idx, cl.cluster_check_idx_to_pcm_check_idx);
    ASSERT_EQ(expected_pcm_check_idx_to_cluster_check_idx, cl.pcm_check_idx_to_cluster_check_idx);
    ASSERT_EQ(cl.global_check_membership[4], &cl);
    ASSERT_EQ(cl.global_check_membership[5], &cl);
    ASSERT_EQ(cl.global_check_membership[6], &cl);

    cl.compute_growth_candidate_bit_nodes();
    auto expected_boundary_check_nodes = tsl::robin_set<int>{4,6};
    expected_candidate_bit_nodes = std::vector<int>{4,7};

    ASSERT_EQ(expected_boundary_check_nodes, cl.boundary_check_nodes);
    ASSERT_EQ(expected_candidate_bit_nodes, cl.candidate_bit_nodes);

    //check the cluster pcm
    auto expected_column = std::vector<int>{1,0};
    ASSERT_TRUE(cl.cluster_pcm.size() == 2);
    ASSERT_EQ(expected_column, cl.cluster_pcm[0]);

    expected_column = std::vector<int>{0,2};
    ASSERT_EQ(expected_column, cl.cluster_pcm[1]);

    delete gbm;
    delete gcm;

}


TEST(LsdCluster, merge_clusters_test){


    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(5);
    auto gbm = new ldpc::lsd::LsdCluster *[pcm.n](); //global bit dictionary
    auto gcm = new ldpc::lsd::LsdCluster *[pcm.m](); //global check dictionary

    // auto syndrome_index = 0;
    auto cl1 = ldpc::lsd::LsdCluster(pcm, 0, gcm, gbm);
    auto cl2 = ldpc::lsd::LsdCluster(pcm, 3, gcm, gbm);

    cl2.grow_cluster(ldpc::lsd::NULL_DOUBLE_VECTOR, std::numeric_limits<int>::max(), true);
    cl1.grow_cluster(ldpc::lsd::NULL_DOUBLE_VECTOR, std::numeric_limits<int>::max(), true);

    ASSERT_TRUE(cl1.active);
    ASSERT_TRUE(cl2.active);

    cl2.grow_cluster(ldpc::lsd::NULL_DOUBLE_VECTOR, std::numeric_limits<int>::max(), true);

    ASSERT_FALSE(cl1.active);
    ASSERT_TRUE(cl2.active);

    auto expected_bit_nodes = tsl::robin_set<int>{0,1,2,3,4};
    auto expected_check_nodes = tsl::robin_set<int>{0,1,2,3};  
    ASSERT_EQ(expected_bit_nodes, cl2.bit_nodes);  
    ASSERT_EQ(expected_check_nodes, cl2.check_nodes);

    ASSERT_TRUE(cl2.valid);
    
    delete gbm;
    delete gcm;

}

TEST(LsdCluster, merge_clusters_otf_test){
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(5);
    auto gbm = new ldpc::lsd::LsdCluster *[pcm.n](); //global bit dictionary
    auto gcm = new ldpc::lsd::LsdCluster *[pcm.m](); //global check dictionary
    auto cl1 = ldpc::lsd::LsdCluster(pcm, 0, gcm, gbm);
    auto cl2 = ldpc::lsd::LsdCluster(pcm, 2, gcm, gbm);

    cl2.grow_cluster(std::vector<double>{0.1,0.1,0.1,0.5,0.5}, 1, true);
    cl1.grow_cluster(std::vector<double>{0.1,0.1,0.1,0.5,0.5}, 1, true);

    ASSERT_TRUE(cl1.active);
    ASSERT_TRUE(cl2.active);

    cl2.grow_cluster(std::vector<double>{0.1,0.1,0.1,0.1,0.1}, 1, true);

    ASSERT_FALSE(cl1.active);
    ASSERT_TRUE(cl2.active);

    auto expected_bit_nodes = tsl::robin_set<int>{0,1,2};
    auto expected_check_nodes = tsl::robin_set<int>{0,1,2};
    ASSERT_EQ(expected_bit_nodes, cl2.bit_nodes);
    ASSERT_EQ(expected_check_nodes, cl2.check_nodes);
    ASSERT_TRUE(cl2.valid);

    auto solution = cl2.pluDecomposition.lu_solve(cl2.cluster_pcm_syndrome);


    auto decoding = vector<uint8_t>(pcm.n,0);
    for(auto cluster_bit_idx: solution){
        decoding[cl2.cluster_bit_idx_to_pcm_bit_idx[cluster_bit_idx]] = 1;
    }

    auto decoding_syndrome = pcm.mulvec(decoding);

    auto expected_syndrome = vector<uint8_t>{1,0,1,0};

    ASSERT_EQ(decoding_syndrome,expected_syndrome);

    delete gbm;
    delete gcm;

}



TEST(LsdDecoder, otf_ring_code) {

    for (auto length = 2; length < 12; length++) {

        auto pcm = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(length);
        auto bp = ldpc::bp::BpDecoder(pcm, std::vector<double>(pcm.n, 0.1));
        bp.maximum_iterations = 3;
        auto ufd = LsdDecoder(pcm);

        for (int i = 0; i < std::pow(2, length); i++) {
            auto error = ldpc::util::decimal_to_binary(i, length);
            auto syndrome = pcm.mulvec(error);
            bp.decode(syndrome);
            auto decoding = ufd.on_the_fly_decode(syndrome, bp.log_prob_ratios);
            auto decoding_syndrome = pcm.mulvec(decoding);
            ASSERT_TRUE(syndrome == decoding_syndrome);
        }
    }
}


TEST(LsdDecoder, otf_hamming_code) {
    for (auto hamming_code_rank = 3; hamming_code_rank < 11; hamming_code_rank++) {

        auto pcm = ldpc::gf2codes::hamming_code<ldpc::bp::BpEntry>(hamming_code_rank);
        auto bp = ldpc::bp::BpDecoder(pcm, std::vector<double>(pcm.n, 0.1));
        bp.maximum_iterations = 2;
        auto ufd = LsdDecoder(pcm);
        for (int i = 0; i < std::pow(2, hamming_code_rank); i++) {
            auto syndrome = ldpc::util::decimal_to_binary(i, hamming_code_rank);
            bp.decode(syndrome);
            auto decoding = ufd.on_the_fly_decode(syndrome, bp.log_prob_ratios);
            auto decoding_syndrome = pcm.mulvec(decoding);
            ASSERT_TRUE(syndrome == decoding_syndrome);
        }
    }
}

TEST(LsdDecoder, ho_lsd_hamming_code_osd2) {
    for (auto hamming_code_rank = 3; hamming_code_rank < 9; hamming_code_rank++) {
//        std::cout << "rank: " << hamming_code_rank << std::endl;

        auto pcm = ldpc::gf2codes::hamming_code<ldpc::bp::BpEntry>(hamming_code_rank);
        auto bp = ldpc::bp::BpDecoder(pcm, std::vector<double>(pcm.n, 0.1));
        bp.maximum_iterations = 2;
        auto ufd = LsdDecoder(pcm);
        for (int i = 0; i < std::pow(2, hamming_code_rank); i++) {
//            std::cout << i << std::endl;

            auto syndrome = ldpc::util::decimal_to_binary(i, hamming_code_rank);
            bp.decode(syndrome);
            auto decoding = ufd.on_the_fly_decode(syndrome, bp.log_prob_ratios, 2);
            auto decoding_syndrome = pcm.mulvec(decoding);
            ASSERT_TRUE(syndrome == decoding_syndrome);
        }
    }
}

TEST(LsdDecoder, ho_lsd_ring_code_osd2) {

    for (auto length = 3; length < 13; length++) {
//        std::cout << "length: " << length << std::endl;

        auto pcm = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(length);
        auto bp = ldpc::bp::BpDecoder(pcm, std::vector<double>(pcm.n, 0.1));
        bp.maximum_iterations = 3;
        auto ufd = LsdDecoder(pcm);

        for (int i = 0; i < std::pow(2, length); i++) {
//            std::cout << i << std::endl;
            auto error = ldpc::util::decimal_to_binary(i, length);
            auto syndrome = pcm.mulvec(error);
            bp.decode(syndrome);
            auto decoding = ufd.on_the_fly_decode(syndrome, bp.log_prob_ratios, 2);
            auto decoding_syndrome = pcm.mulvec(decoding);

            ASSERT_TRUE(syndrome == decoding_syndrome);
        }
    }
}


TEST(LsdDecoder, ho_lsd_ring_code_osd_e3) {

    for (auto length = 3; length < 11; length++) {
//        std::cout << "length: " << length << std::endl;

        auto pcm = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(length);
        auto bp = ldpc::bp::BpDecoder(pcm, std::vector<double>(pcm.n, 0.1));
        bp.maximum_iterations = 3;
        auto ufd = LsdDecoder(pcm);

        for (int i = 0; i < std::pow(2, length); i++) {
//            std::cout << i << std::endl;
            auto error = ldpc::util::decimal_to_binary(i, length);
            auto syndrome = pcm.mulvec(error);
            bp.decode(syndrome);
            auto decoding = ufd.on_the_fly_decode(syndrome, bp.log_prob_ratios, 3, ldpc::osd::OsdMethod::EXHAUSTIVE);
            auto decoding_syndrome = pcm.mulvec(decoding);

            ASSERT_TRUE(syndrome == decoding_syndrome);
        }
    }
}

TEST(LsdDecoder, ho_lsd_hamming_code_osde_3) {
    for (auto hamming_code_rank = 3; hamming_code_rank < 7; hamming_code_rank++) {
//        std::cout << "rank: " << hamming_code_rank << std::endl;

        auto pcm = ldpc::gf2codes::hamming_code<ldpc::bp::BpEntry>(hamming_code_rank);
        auto bp = ldpc::bp::BpDecoder(pcm, std::vector<double>(pcm.n, 0.1));
        bp.maximum_iterations = 2;
        auto ufd = LsdDecoder(pcm);
        for (int i = 0; i < std::pow(2, hamming_code_rank); i++) {
//            std::cout << i << std::endl;

            auto syndrome = ldpc::util::decimal_to_binary(i, hamming_code_rank);
            bp.decode(syndrome);
            auto decoding = ufd.on_the_fly_decode(syndrome, bp.log_prob_ratios, 3, ldpc::osd::OsdMethod::EXHAUSTIVE);
            auto decoding_syndrome = pcm.mulvec(decoding);
            ASSERT_TRUE(syndrome == decoding_syndrome);
        }
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}