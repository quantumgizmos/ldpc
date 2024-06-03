#include <gtest/gtest.h>
#include "ldpc.hpp"
#include "gf2codes.hpp"
#include "lsd.hpp"
#include "util.hpp"
#include "bp.hpp"
#include "rapidcsv.h"
#include "io.hpp"

using namespace std;
using namespace ldpc::lsd;
using namespace ldpc::sparse_matrix_util;


TEST(LsdCluster, init1){

    auto pcm = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(10);
//    auto gbm = new ldpc::lsd::LsdCluster *[pcm.n](); //global bit dictionary
//    auto gcm = new ldpc::lsd::LsdCluster *[pcm.m](); //global check dictionary
    auto gbm = std::make_shared<std::vector<LsdCluster*>>(std::vector<LsdCluster*>(pcm.n));
    auto gcm = std::make_shared<std::vector<LsdCluster*>>(std::vector<LsdCluster*>(pcm.m));
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
    ASSERT_EQ(gcm->at(syndrome_index), &cl);
    ASSERT_EQ(expected_cluster_check_idx_to_pcm_check_idx, cl.cluster_check_idx_to_pcm_check_idx);
    ASSERT_EQ(expected_pcm_check_idx_to_cluster_check_idx, cl.pcm_check_idx_to_cluster_check_idx);

//    delete gbm;
//    delete gcm;

}


TEST(LsdCluster, add_bitANDadd_check_add){

    auto pcm = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(10);
//    auto gbm = new ldpc::lsd::LsdCluster *[pcm.n]; //global bit dictionary
//    auto gcm = new ldpc::lsd::LsdCluster *[pcm.m]; //global check dictionary
    auto gbm = std::make_shared<std::vector<LsdCluster*>>(std::vector<LsdCluster*>(pcm.n));
    auto gcm = std::make_shared<std::vector<LsdCluster*>>(std::vector<LsdCluster*>(pcm.m));
    auto syndrome_index = 1;
    auto cl = ldpc::lsd::LsdCluster(pcm, syndrome_index, gcm, gbm);

    cl.compute_growth_candidate_bit_nodes();
    auto expected_candidate_bit_nodes = tsl::robin_set<int>{1, 2};
    ASSERT_EQ(expected_candidate_bit_nodes, cl.candidate_bit_nodes);


    cl.add_bit(*(++expected_candidate_bit_nodes.begin()));
    cl.add_check(2, true);


    auto expected_bit_nodes = tsl::robin_set<int>{2};
    auto expected_check_nodes = tsl::robin_set<int>{syndrome_index, 2};
    auto expected_cluster_check_idx_to_pcm_check_idx = std::vector<int>{1, 2};
    auto expected_pcm_check_idx_to_cluster_check_idx = tsl::robin_map<int, int>{{1, 0},
                                                                                {2, 1}};

    ASSERT_EQ(expected_bit_nodes, cl.bit_nodes);
    ASSERT_EQ(cl.global_bit_membership.get()->at(2), &cl);
    ASSERT_EQ(cl.cluster_bit_idx_to_pcm_bit_idx[0], 2);
    ASSERT_EQ(expected_check_nodes, cl.check_nodes);
    ASSERT_EQ(expected_cluster_check_idx_to_pcm_check_idx, cl.cluster_check_idx_to_pcm_check_idx);
    ASSERT_EQ(expected_pcm_check_idx_to_cluster_check_idx, cl.pcm_check_idx_to_cluster_check_idx);
    ASSERT_EQ(cl.global_check_membership.get()->at(1), &cl);
    ASSERT_EQ(cl.global_check_membership.get()->at(2), &cl);


    // Test adding existing checks and bits
    cl.add_bit(*(++expected_candidate_bit_nodes.begin()));
    cl.add_check(2, true);

    ASSERT_EQ(expected_bit_nodes, cl.bit_nodes);
    ASSERT_EQ(cl.global_bit_membership.get()->at(2), &cl);
    ASSERT_EQ(cl.cluster_bit_idx_to_pcm_bit_idx[0], 2);
    ASSERT_EQ(expected_check_nodes, cl.check_nodes);
    ASSERT_EQ(expected_cluster_check_idx_to_pcm_check_idx, cl.cluster_check_idx_to_pcm_check_idx);
    ASSERT_EQ(expected_pcm_check_idx_to_cluster_check_idx, cl.pcm_check_idx_to_cluster_check_idx);
    ASSERT_EQ(cl.global_check_membership.get()->at(1), &cl);
    ASSERT_EQ(cl.global_check_membership.get()->at(2), &cl);

    //check that bit is remove from boundary check node is removed from boundary check nodes
    cl.compute_growth_candidate_bit_nodes();
    expected_candidate_bit_nodes = tsl::robin_set<int>{1, 3};
    ASSERT_EQ(expected_candidate_bit_nodes, cl.candidate_bit_nodes);

    //add bit 3, verify that boundary check 2 is removed from the boundary check list
    cl.add_bit(3);
    cl.compute_growth_candidate_bit_nodes();
    auto expected_boundary_check_nodes = tsl::robin_set<int>{1};
    ASSERT_EQ(expected_boundary_check_nodes, cl.boundary_check_nodes);

//    delete gbm;
//    delete gcm;

}

TEST(LsdCluster, add_bit_node_to_cluster){


    auto pcm = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(10);
//    auto gbm = new ldpc::lsd::LsdCluster *[pcm.n](); //global bit dictionary
//    auto gcm = new ldpc::lsd::LsdCluster *[pcm.m](); //global check dictionary
    auto gbm = std::make_shared<std::vector<LsdCluster*>>(std::vector<LsdCluster*>(pcm.n));
    auto gcm = std::make_shared<std::vector<LsdCluster*>>(std::vector<LsdCluster*>(pcm.m));
    auto syndrome_index = 1;
    auto cl = ldpc::lsd::LsdCluster(pcm, syndrome_index, gcm, gbm);

    cl.compute_growth_candidate_bit_nodes();
    auto expected_candidate_bit_nodes = tsl::robin_set<int>{1, 2};
    ASSERT_EQ(expected_candidate_bit_nodes, cl.candidate_bit_nodes);


    auto bit_membership = cl.global_bit_membership.get()->at(0);

    // add bit 2 to the cluster
    cl.add_bit_node_to_cluster(2);

    auto expected_bit_nodes = tsl::robin_set<int>{2};
    auto expected_check_nodes = tsl::robin_set<int>{1, 2};
    auto expected_cluster_check_idx_to_pcm_check_idx = std::vector<int>{1, 2};
    auto expected_pcm_check_idx_to_cluster_check_idx = tsl::robin_map<int, int>{{1, 0},
                                                                                {2, 1}};

    ASSERT_EQ(expected_bit_nodes, cl.bit_nodes);
    ASSERT_EQ(cl.global_bit_membership.get()->at(2), &cl);
    ASSERT_EQ(cl.cluster_bit_idx_to_pcm_bit_idx[0], 2);
    ASSERT_EQ(expected_check_nodes, cl.check_nodes);
    ASSERT_EQ(expected_cluster_check_idx_to_pcm_check_idx, cl.cluster_check_idx_to_pcm_check_idx);
    ASSERT_EQ(expected_pcm_check_idx_to_cluster_check_idx, cl.pcm_check_idx_to_cluster_check_idx);
    ASSERT_EQ(cl.global_check_membership.get()->at(1), &cl);
    ASSERT_EQ(cl.global_check_membership.get()->at(2), &cl);

    cl.compute_growth_candidate_bit_nodes();
    auto expected_boundary_check_nodes = tsl::robin_set<int>{1,2};
    expected_candidate_bit_nodes = tsl::robin_set<int>{1, 3};

    ASSERT_EQ(expected_boundary_check_nodes, cl.boundary_check_nodes);
    ASSERT_EQ(expected_candidate_bit_nodes, cl.candidate_bit_nodes);

    //check the cluster pcm
    auto expected_column = std::vector<int>{0, 1};
    ASSERT_TRUE(cl.cluster_pcm.size() == 1);
    ASSERT_EQ(expected_column, cl.cluster_pcm[0]);


    //add bit 3, verify that check 1 is removed from the boundary check list
    cl.add_bit_node_to_cluster(1);

    cl.compute_growth_candidate_bit_nodes();
    expected_boundary_check_nodes = tsl::robin_set<int>{0, 2};
    expected_candidate_bit_nodes = tsl::robin_set<int>{0, 3};
    ASSERT_EQ(expected_boundary_check_nodes, cl.boundary_check_nodes);
    ASSERT_EQ(expected_candidate_bit_nodes, cl.candidate_bit_nodes);


    expected_bit_nodes = tsl::robin_set<int>{1, 2};
    expected_check_nodes = tsl::robin_set<int>{0, 1, 2};
    expected_cluster_check_idx_to_pcm_check_idx = std::vector<int>{1, 2, 0};
    expected_pcm_check_idx_to_cluster_check_idx = tsl::robin_map<int, int>{{1, 0},
                                                                           {2, 1},
                                                                           {0, 2}};

    ASSERT_EQ(expected_bit_nodes, cl.bit_nodes);
    ASSERT_EQ(cl.global_bit_membership.get()->at(1), &cl);
    ASSERT_EQ(cl.cluster_bit_idx_to_pcm_bit_idx[0], 2);
    ASSERT_EQ(expected_check_nodes, cl.check_nodes);
    ASSERT_EQ(expected_cluster_check_idx_to_pcm_check_idx, cl.cluster_check_idx_to_pcm_check_idx);
    ASSERT_EQ(expected_pcm_check_idx_to_cluster_check_idx, cl.pcm_check_idx_to_cluster_check_idx);
    ASSERT_EQ(cl.global_check_membership.get()->at(0), &cl);
    ASSERT_EQ(cl.global_check_membership.get()->at(2), &cl);

    //check the cluster pcm
    expected_column = std::vector<int>{0, 1};
    ASSERT_TRUE(cl.cluster_pcm.size() == 2);
    ASSERT_EQ(expected_column, cl.cluster_pcm[0]);

    expected_column = std::vector<int>{2, 0};
    ASSERT_EQ(expected_column, cl.cluster_pcm[1]);

//    delete gbm;
//    delete gcm;

}


TEST(LsdCluster, grow_cluster) {


    auto pcm = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(10);
//    auto gbm = new ldpc::lsd::LsdCluster *[pcm.n](); //global bit dictionary
//    auto gcm = new ldpc::lsd::LsdCluster *[pcm.m](); //global check dictionary
    auto gbm = std::make_shared<std::vector<LsdCluster*>>(std::vector<LsdCluster*>(pcm.n));
    auto gcm = std::make_shared<std::vector<LsdCluster*>>(std::vector<LsdCluster*>(pcm.m));
    auto syndrome_index = 5;
    auto cl = ldpc::lsd::LsdCluster(pcm, syndrome_index, gcm, gbm);

    cl.compute_growth_candidate_bit_nodes();
    auto expected_candidate_bit_nodes = tsl::robin_set<int>{5, 6};
    ASSERT_EQ(expected_candidate_bit_nodes, cl.candidate_bit_nodes);
    auto bit_membership = cl.global_bit_membership.get()->at(5);
    ASSERT_EQ(bit_membership, nullptr);

    cl.grow_cluster();

    auto expected_bit_nodes = tsl::robin_set<int>{5, 6};
    auto expected_check_nodes = tsl::robin_set<int>{5, 4, 6};
    auto expected_cluster_check_idx_to_pcm_check_idx = std::vector<int>{5, 4, 6};
    auto expected_pcm_check_idx_to_cluster_check_idx = tsl::robin_map<int, int>{{5, 0},
                                                                                {4, 1},
                                                                                {6, 2}};

    ASSERT_EQ(expected_bit_nodes, cl.bit_nodes);
    ASSERT_EQ(cl.global_bit_membership.get()->at(5), &cl);
    ASSERT_EQ(cl.cluster_bit_idx_to_pcm_bit_idx[0], 5);
    ASSERT_EQ(expected_check_nodes, cl.check_nodes);
    ASSERT_EQ(expected_cluster_check_idx_to_pcm_check_idx, cl.cluster_check_idx_to_pcm_check_idx);
    ASSERT_EQ(expected_pcm_check_idx_to_cluster_check_idx, cl.pcm_check_idx_to_cluster_check_idx);
    ASSERT_EQ(cl.global_check_membership.get()->at(4), &cl);
    ASSERT_EQ(cl.global_check_membership.get()->at(5), &cl);
    ASSERT_EQ(cl.global_check_membership.get()->at(6), &cl);

    cl.compute_growth_candidate_bit_nodes();
    auto expected_boundary_check_nodes = tsl::robin_set<int>{4, 6};
    expected_candidate_bit_nodes = tsl::robin_set<int>{4, 7};

    ASSERT_EQ(expected_boundary_check_nodes, cl.boundary_check_nodes);
    ASSERT_EQ(expected_candidate_bit_nodes, cl.candidate_bit_nodes);

    //check the cluster pcm
    auto expected_column = std::vector<int>{1, 0};
    ASSERT_TRUE(cl.cluster_pcm.size() == 2);
    ASSERT_EQ(expected_column, cl.cluster_pcm[0]);

    expected_column = std::vector<int>{0, 2};
    ASSERT_EQ(expected_column, cl.cluster_pcm[1]);

//    delete gbm;
//    delete gcm;

}


TEST(LsdCluster, merge_clusters_test) {


    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(5);
//    auto gbm = new ldpc::lsd::LsdCluster *[pcm.n](); //global bit dictionary
//    auto gcm = new ldpc::lsd::LsdCluster *[pcm.m](); //global check dictionary
    // auto syndrome_index = 0;
    auto gbm = std::make_shared<std::vector<LsdCluster*>>(std::vector<LsdCluster*>(pcm.n));
    auto gcm = std::make_shared<std::vector<LsdCluster*>>(std::vector<LsdCluster*>(pcm.m));
    auto cl1 = ldpc::lsd::LsdCluster(pcm, 0, gcm, gbm);
    auto cl2 = ldpc::lsd::LsdCluster(pcm, 3, gcm, gbm);

    cl2.grow_cluster(ldpc::lsd::NULL_DOUBLE_VECTOR, std::numeric_limits<int>::max(), true);
    cl1.grow_cluster(ldpc::lsd::NULL_DOUBLE_VECTOR, std::numeric_limits<int>::max(), true);

    ASSERT_TRUE(cl1.active);
    ASSERT_TRUE(cl2.active);

    cl2.grow_cluster(ldpc::lsd::NULL_DOUBLE_VECTOR, std::numeric_limits<int>::max(), true);

    ASSERT_FALSE(cl1.active);
    ASSERT_TRUE(cl2.active);

    auto expected_bit_nodes = tsl::robin_set<int>{0, 1, 2, 3, 4};
    auto expected_check_nodes = tsl::robin_set<int>{0, 1, 2, 3};
    ASSERT_EQ(expected_bit_nodes, cl2.bit_nodes);
    ASSERT_EQ(expected_check_nodes, cl2.check_nodes);

    ASSERT_TRUE(cl2.valid);

//    delete gbm;
//    delete gcm;

}

TEST(LsdCluster, merge_clusters_otf_test) {
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(5);
//    auto gbm = new ldpc::lsd::LsdCluster *[pcm.n](); //global bit dictionary
//    auto gcm = new ldpc::lsd::LsdCluster *[pcm.m](); //global check dictionary
    auto gbm = std::make_shared<std::vector<LsdCluster*>>(std::vector<LsdCluster*>(pcm.n));
    auto gcm = std::make_shared<std::vector<LsdCluster*>>(std::vector<LsdCluster*>(pcm.m));
    auto cl1 = ldpc::lsd::LsdCluster(pcm, 0, gcm, gbm);
    auto cl2 = ldpc::lsd::LsdCluster(pcm, 2, gcm, gbm);

    cl2.grow_cluster(std::vector<double>{0.1, 0.1, 0.1, 0.5, 0.5}, 1, true);
    cl1.grow_cluster(std::vector<double>{0.1, 0.1, 0.1, 0.5, 0.5}, 1, true);

    ASSERT_TRUE(cl1.active);
    ASSERT_TRUE(cl2.active);

    cl2.grow_cluster(std::vector<double>{0.1, 0.1, 0.1, 0.1, 0.1}, 1, true);

    ASSERT_FALSE(cl1.active);
    ASSERT_TRUE(cl2.active);

    auto expected_bit_nodes = tsl::robin_set<int>{0, 1, 2};
    auto expected_check_nodes = tsl::robin_set<int>{0, 1, 2};
    ASSERT_EQ(expected_bit_nodes, cl2.bit_nodes);
    ASSERT_EQ(expected_check_nodes, cl2.check_nodes);
    ASSERT_TRUE(cl2.valid);

    auto solution = cl2.pluDecomposition.lu_solve(cl2.cluster_pcm_syndrome);


    auto decoding = vector<uint8_t>(pcm.n, 0);
    for (auto cluster_bit_idx: solution) {
        decoding[cl2.cluster_bit_idx_to_pcm_bit_idx[cluster_bit_idx]] = 1;
    }

    auto decoding_syndrome = pcm.mulvec(decoding);

    auto expected_syndrome = vector<uint8_t>{1, 0, 1, 0};

    ASSERT_EQ(decoding_syndrome, expected_syndrome);

//    delete gbm;
//    delete gcm;

}



TEST(LsdDecoder, otf_ring_code) {

    for (auto length = 2; length < 12; length++) {

        auto pcm = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(length);
        auto bp = ldpc::bp::BpDecoder(pcm, std::vector<double>(pcm.n, 0.1));
        bp.maximum_iterations = 3;
        auto lsd = LsdDecoder(pcm);

        for (int i = 0; i < std::pow(2, length); i++) {
            auto error = ldpc::util::decimal_to_binary(i, length);
            auto syndrome = pcm.mulvec(error);
            bp.decode(syndrome);
            auto decoding = lsd.on_the_fly_decode(syndrome, bp.log_prob_ratios);
            auto decoding_syndrome = pcm.mulvec(decoding);
            ASSERT_TRUE(syndrome == decoding_syndrome);
        }
    }
}


TEST(LsdDecoder, otf_hamming_code) {
    for (auto hamming_code_rank = 3; hamming_code_rank < 9; hamming_code_rank++) {

        auto pcm = ldpc::gf2codes::hamming_code<ldpc::bp::BpEntry>(hamming_code_rank);
        auto bp = ldpc::bp::BpDecoder(pcm, std::vector<double>(pcm.n, 0.1));
        bp.maximum_iterations = 2;
        auto lsd = LsdDecoder(pcm);
        for (int i = 0; i < std::pow(2, hamming_code_rank); i++) {
            auto syndrome = ldpc::util::decimal_to_binary(i, hamming_code_rank);
            bp.decode(syndrome);
            auto decoding = lsd.on_the_fly_decode(syndrome, bp.log_prob_ratios);
            auto decoding_syndrome = pcm.mulvec(decoding);
            ASSERT_TRUE(syndrome == decoding_syndrome);
        }
    }
}

TEST(LsdDecoder, lsdw_decode) {
    for (auto hamming_code_rank = 3; hamming_code_rank < 9; hamming_code_rank++) {
        // std::cout << "rank " << hamming_code_rank << std::endl;
        auto pcm = ldpc::gf2codes::hamming_code<ldpc::bp::BpEntry>(hamming_code_rank);
        auto bp = ldpc::bp::BpDecoder(pcm, std::vector<double>(pcm.n, 0.1));
        bp.maximum_iterations = 2;
        auto lsd = LsdDecoder(pcm, ldpc::osd::OsdMethod::COMBINATION_SWEEP, 3);
        lsd.lsd_order = 3;
        for (int i = 0; i < std::pow(2, hamming_code_rank); i++) {
            // std::cout << i << std::endl;
            auto syndrome = ldpc::util::decimal_to_binary(i, hamming_code_rank);
            bp.decode(syndrome);
            auto decoding = lsd.lsd_decode(syndrome, bp.log_prob_ratios, 1, true);
            auto decoding_syndrome = pcm.mulvec(decoding);
            ASSERT_TRUE(syndrome == decoding_syndrome);
        }
    }
}

TEST(LsdDecoder, lsdw_decode_ring_code) {

    for (auto length = 2; length < 10; length++) {

        auto pcm = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(length);
        auto bp = ldpc::bp::BpDecoder(pcm, std::vector<double>(pcm.n, 0.1));
        bp.maximum_iterations = 3;
        auto lsd = LsdDecoder(pcm, ldpc::osd::OsdMethod::COMBINATION_SWEEP, 5);
        lsd.lsd_order = 5;
        for (int i = 0; i < std::pow(2, length); i++) {
            auto error = ldpc::util::decimal_to_binary(i, length);
            auto syndrome = pcm.mulvec(error);
            bp.decode(syndrome);
            auto decoding = lsd.lsd_decode(syndrome, bp.log_prob_ratios, 1, true);
            auto decoding_syndrome = pcm.mulvec(decoding);
            ASSERT_TRUE(syndrome == decoding_syndrome);
        }
    }
}

TEST(LsdDecoder, test_fail_case) {
    auto csv_path = ldpc::io::getFullPath("cpp_test/test_inputs/qdlpc_test.csv");
    rapidcsv::Document doc(csv_path, rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(';'));
    int row_count = doc.GetColumn<string>(0).size();
    std::vector<string> row = doc.GetRow<string>(0);

    int m = stoi(row[0]);
    int n = stoi(row[1]);
    auto input_csr_vector = ldpc::io::string_to_csr_vector(row[2]);
    auto pcm = ldpc::bp::BpSparse(m, n);
    pcm.csr_insert(input_csr_vector);

    ASSERT_TRUE(pcm.m == 192);
    ASSERT_TRUE(pcm.n == 400);

    //this is the syndrome that is currently failing in Python.
    auto syndrome_sparse = std::vector<uint8_t>{3, 5, 10, 12, 13, 16, 28, 44, 45, 50, 55, 70, 82,
                                                87, 92, 128, 130, 131, 139, 143, 157, 176};

    auto syndrome = std::vector<uint8_t>(pcm.m, 0);
    for (auto idx: syndrome_sparse) {
        syndrome[idx] = 1;
    }
    auto channel_probabilities = std::vector<double>(pcm.n, 0.01);
    //setup the BP decoder with only 2 iterations
    auto bp = ldpc::bp::BpDecoder(pcm, channel_probabilities, 100, ldpc::bp::MINIMUM_SUM, ldpc::bp::PARALLEL, 0.625);
    auto lsd = LsdDecoder(pcm, ldpc::osd::OsdMethod::COMBINATION_SWEEP, 5);
    lsd.lsd_order = 5;
    bp.decode(syndrome);
    auto decoding = lsd.lsd_decode(syndrome, bp.log_prob_ratios, 1, true);
    auto decoding_syndrome = pcm.mulvec(decoding);
    ASSERT_TRUE(bp.converge == false);
    ASSERT_TRUE(syndrome == decoding_syndrome);
}

TEST(LsdDecoder, test_cluster_stats) {
    auto length = 5;
    auto pcm = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(length);
    auto bp = ldpc::bp::BpDecoder(pcm, std::vector<double>(pcm.n, 0.1));
    bp.maximum_iterations = 3;
    auto lsd = LsdDecoder(pcm, ldpc::osd::OsdMethod::EXHAUSTIVE, 0);
    lsd.set_do_stats(true);
    auto syndrome = std::vector<uint8_t>({1, 1, 0, 0, 0});
    lsd.statistics.syndrome = std::vector<uint8_t>(pcm.m, 1);
    lsd.setLsdMethod(ldpc::osd::OsdMethod::EXHAUSTIVE);
    auto decoding = lsd.lsd_decode(syndrome, bp.log_prob_ratios, 1, true);
    lsd.statistics.error = std::vector<uint8_t>(pcm.n, 1);
    lsd.statistics.compare_recover = std::vector<uint8_t>(pcm.n, 0);

    auto stats = lsd.statistics;
    std::cout << stats.toString() << std::endl;
    ASSERT_TRUE(lsd.get_do_stats());
    ASSERT_TRUE(stats.lsd_method = ldpc::osd::OsdMethod::EXHAUSTIVE);
    ASSERT_TRUE(stats.lsd_order == 0);
    // check that there is one timestep with two entries in the statistics
    ASSERT_TRUE(stats.individual_cluster_stats.size() == 2);
    ASSERT_TRUE(stats.global_timestep_bit_history.size() == 1);
    ASSERT_TRUE(stats.global_timestep_bit_history[0].size() == 2);
    ASSERT_TRUE(stats.global_timestep_bit_history[0][0].size() == 1);
    ASSERT_TRUE(stats.global_timestep_bit_history[0][1].size() == 2);
    ASSERT_TRUE(stats.global_timestep_bit_history[1].empty());
    ASSERT_TRUE(stats.elapsed_time > 0.0);
    ASSERT_TRUE(stats.individual_cluster_stats[0].active == false);
    ASSERT_TRUE(stats.individual_cluster_stats[0].got_inactive_in_timestep == 0);
    ASSERT_TRUE(stats.individual_cluster_stats[1].got_valid_in_timestep == 0);
    ASSERT_TRUE(stats.individual_cluster_stats[0].size_history.size() == 1);
    ASSERT_TRUE(stats.individual_cluster_stats[1].size_history[0] == 2);
    ASSERT_TRUE(stats.individual_cluster_stats[1].solution.size() == 2);
    ASSERT_TRUE(stats.bit_llrs.size() == pcm.n);
    ASSERT_TRUE(stats.error.size() == pcm.n);
    ASSERT_TRUE(stats.syndrome.size() == pcm.n);
    ASSERT_TRUE(stats.compare_recover.size() == pcm.n);

    // now reset
    lsd.reset_cluster_stats();
    stats = lsd.statistics;
    ASSERT_TRUE(lsd.get_do_stats());
    ASSERT_TRUE(stats.lsd_method = ldpc::osd::OsdMethod::COMBINATION_SWEEP);
    ASSERT_TRUE(stats.lsd_order == 0);
    ASSERT_TRUE(stats.individual_cluster_stats.empty());
    ASSERT_TRUE(stats.elapsed_time == 0.0);
    ASSERT_TRUE(stats.global_timestep_bit_history.empty());
    ASSERT_TRUE(stats.bit_llrs.empty());
    ASSERT_TRUE(stats.error.empty());
    ASSERT_TRUE(stats.syndrome.empty());
    ASSERT_TRUE(stats.compare_recover.empty());
}

TEST(LsdDecoder, test_reshuffle_same_wt_indices) {
    std::vector<double> weights = {0.1, 0.1, 0.1, 0.5, 1.5, 0.1, 0.6, 0.1, 0.1, 0.1, 0.1, 1.5, 1.9, 0.1, 0.1, 0.1, 0.1};
    std::vector<int> sorted_indices = ldpc::lsd::sort_indices(weights);
    auto res = ldpc::lsd::LsdCluster::randomize_same_weight_indices(sorted_indices, weights);
    ASSERT_TRUE(res.size() == sorted_indices.size());
    // should be true whp
    ASSERT_TRUE(sorted_indices != res);
    // check if all elems are contained
    std::sort(sorted_indices.begin(), sorted_indices.end());
    std::sort(res.begin(), res.end());
    ASSERT_TRUE(sorted_indices == res);

    weights = {0.1, 0.1, 0.1, 0.5, 1.5, 0.1, 0.6, 0.1, 0.1, 0.1, 0.1, 1.5, 1.9, 0.1, 0.1, 0.1, 0.1};
    std::shuffle(weights.begin(), weights.end(), std::mt19937(std::random_device()()));
    sorted_indices = ldpc::lsd::sort_indices(weights);
    res = ldpc::lsd::LsdCluster::randomize_same_weight_indices(sorted_indices, weights);
    ASSERT_TRUE(res.size() == sorted_indices.size());
    // should be true whp
    ASSERT_TRUE(sorted_indices != res);
    // check if all elems are contained
    std::sort(sorted_indices.begin(), sorted_indices.end());
    std::sort(res.begin(), res.end());
    ASSERT_TRUE(sorted_indices == res);

    weights = {};
    sorted_indices = ldpc::lsd::sort_indices(weights);
    res = ldpc::lsd::LsdCluster::randomize_same_weight_indices(sorted_indices, weights);
    ASSERT_TRUE(res.size() == sorted_indices.size());
    // check if all elems are contained
    std::sort(sorted_indices.begin(), sorted_indices.end());
    std::sort(res.begin(), res.end());
    ASSERT_TRUE(sorted_indices == res);

    weights = {0.1};
    sorted_indices = ldpc::lsd::sort_indices(weights);
    res = ldpc::lsd::LsdCluster::randomize_same_weight_indices(sorted_indices, weights);
    ASSERT_TRUE(res.size() == sorted_indices.size());
    // check if all elems are contained
    std::sort(sorted_indices.begin(), sorted_indices.end());
    std::sort(res.begin(), res.end());
    ASSERT_TRUE(sorted_indices == res);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}