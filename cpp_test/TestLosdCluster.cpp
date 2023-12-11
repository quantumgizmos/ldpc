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
    auto gbm = new ldpc::uf::Cluster *[pcm.n](); //global bit dictionary
    auto gcm = new ldpc::uf::Cluster *[pcm.m](); //global check dictionary

    auto syndrome_index = 0;
    auto cl = ldpc::uf::Cluster(pcm, syndrome_index, gcm, gbm);
    
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

TEST(Cluster, add_bit_node_to_cluster){


    auto pcm = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(10);
    auto gbm = new ldpc::uf::Cluster *[pcm.n](); //global bit dictionary
    auto gcm = new ldpc::uf::Cluster *[pcm.m](); //global check dictionary

    auto syndrome_index = 1;
    auto cl = ldpc::uf::Cluster(pcm, syndrome_index, gcm, gbm);
    
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


TEST(Cluster, grow_cluster){


    auto pcm = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(10);
    auto gbm = new ldpc::uf::Cluster *[pcm.n](); //global bit dictionary
    auto gcm = new ldpc::uf::Cluster *[pcm.m](); //global check dictionary

    auto syndrome_index = 5;
    auto cl = ldpc::uf::Cluster(pcm, syndrome_index, gcm, gbm);
    
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





int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}