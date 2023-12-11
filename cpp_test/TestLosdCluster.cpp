#include <gtest/gtest.h>
#include "ldpc.hpp"
#include "gf2codes.hpp"
#include "union_find.hpp"
#include "util.hpp"
#include "bp.hpp"
#include <robin_map.h>
#include <robin_set.h>

using namespace std;
// using namespace ldpc::uf;
using namespace ldpc::sparse_matrix_util;


TEST(Cluster, init1){

    auto pcm = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(10);
    auto gbm = new ldpc::uf::Cluster *[pcm.n]; //global bit dictionary
    auto gcm = new ldpc::uf::Cluster *[pcm.m]; //global check dictionary

    auto syndrome_index = 0;
    auto cl = ldpc::uf::Cluster(pcm, syndrome_index, gcm, gbm);
    
    ASSERT_TRUE(cl.active);
    ASSERT_FALSE(cl.valid);

    auto expected_bit_nodes = tsl::robin_set<int>{};
    auto expected_check_nodes = tsl::robin_set<int>{syndrome_index};
    auto expected_boundary_check_nodes = tsl::robin_set<int>{syndrome_index};
    auto expected_enclosed_syndromes = tsl::robin_set<int>{syndrome_index};
    auto expected_cluster_check_idx_to_pcm_check_idx = std::vector<std::size_t>{syndrome_index};
    auto expected_pcm_check_idx_to_cluster_check_idx = tsl::robin_map<std::size_t, std::size_t>{{syndrome_index, 0}};

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


TEST(Cluster, add_bitANDadd_check_add){

    auto pcm = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(10);
    auto gbm = new ldpc::uf::Cluster *[pcm.n]; //global bit dictionary
    auto gcm = new ldpc::uf::Cluster *[pcm.m]; //global check dictionary

    auto syndrome_index = 1;
    auto cl = ldpc::uf::Cluster(pcm, syndrome_index, gcm, gbm);
    
    cl.compute_growth_candidate_bit_nodes();
    auto expected_candidate_bit_nodes = std::vector<int>{1,2};
    ASSERT_EQ(expected_candidate_bit_nodes, cl.candidate_bit_nodes);


    cl.add_bit(expected_candidate_bit_nodes[1]);
    cl.add_check(2,true);
    
    
    auto expected_bit_nodes = tsl::robin_set<int>{2};
    auto expected_check_nodes = tsl::robin_set<int>{syndrome_index,2};
    auto expected_cluster_check_idx_to_pcm_check_idx = std::vector<std::size_t>{1,2};
    auto expected_pcm_check_idx_to_cluster_check_idx = tsl::robin_map<std::size_t, std::size_t>{{1, 0},{2,1}};
    
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




int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}