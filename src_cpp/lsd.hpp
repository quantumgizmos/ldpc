#ifndef UF2_H
#define UF2_H

#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include <memory>
#include <iterator>
#include <limits>
#include <set>
#include <map>
#include "sparse_matrix_util.hpp"
#include <robin_map.h>
#include <robin_set.h>
#include <numeric>

#include "bp.hpp"
#include "gf2dense.hpp"
#include "osd_dense.hpp"

namespace ldpc::lsd {
    const std::vector<double> NULL_DOUBLE_VECTOR = {};

    std::vector<int> sort_indices(std::vector<double> &B) {
        std::vector<int> indices(B.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int i, int j) { return B[i] < B[j]; });
        return indices;
    }

    // TODO this should probably become a class?
    struct LsdCluster {
        ldpc::bp::BpSparse &pcm;
        int cluster_id{};
        bool active{}; // if merge one becomes deactivated
        bool valid{}; //
        tsl::robin_set<int> bit_nodes;
        tsl::robin_set<int> check_nodes;
        tsl::robin_set<int> boundary_check_nodes;
        tsl::robin_set<int> candidate_bit_nodes;
        tsl::robin_set<int> enclosed_syndromes;

        std::shared_ptr<std::vector<LsdCluster *>>
                global_check_membership; // store which cluster a check belongs to
        std::shared_ptr<std::vector<LsdCluster *>> global_bit_membership; // store which cluster a check belongs to
        tsl::robin_set<LsdCluster *> merge_list;

        gf2dense::CscMatrix cluster_pcm;

        std::vector<uint8_t> cluster_pcm_syndrome;

        std::vector<int> cluster_check_idx_to_pcm_check_idx;
        tsl::robin_map<int, int> pcm_check_idx_to_cluster_check_idx;

        std::vector<int> cluster_bit_idx_to_pcm_bit_idx;
        gf2dense::PluDecomposition pluDecomposition;
        int nr_merges{};
        std::unordered_map<int, std::unordered_map<int, std::vector<int>>> *global_timestep_bit_history = nullptr;
        int curr_timestep = 0;
        int absorbed_into_cluster = -1;
        int got_inactive_in_timestep = -1;
        bool is_randomize_same_weight_indices = false;

        LsdCluster() = default;

        LsdCluster(ldpc::bp::BpSparse &parity_check_matrix,
                   int syndrome_index,
                   std::shared_ptr<std::vector<LsdCluster *>> ccm, // global check cluster membership
                   std::shared_ptr<std::vector<LsdCluster *>> bcm, // global bit cluster membership
                   bool on_the_fly = false) :
                pcm(parity_check_matrix), cluster_id(syndrome_index), active(true), valid(false) {
            this->boundary_check_nodes.insert(syndrome_index);
            this->enclosed_syndromes.insert(syndrome_index);
            this->global_check_membership = ccm;
            this->global_bit_membership = bcm;
            this->check_nodes.insert(syndrome_index);
            this->global_check_membership->at(syndrome_index) = this;
            this->cluster_pcm_syndrome.clear();
            this->pcm_check_idx_to_cluster_check_idx.insert(
                    std::pair<int, int>{syndrome_index, 0});
            this->cluster_check_idx_to_pcm_check_idx.push_back(syndrome_index);
            this->nr_merges = 0;
            this->pluDecomposition = ldpc::gf2dense::PluDecomposition(this->check_nodes.size(),
                                                                      this->bit_nodes.size(),
                                                                      this->cluster_pcm);
        }

        ~LsdCluster() {
            this->bit_nodes.clear();
            this->check_nodes.clear();
            this->boundary_check_nodes.clear();
            this->candidate_bit_nodes.clear();
            this->enclosed_syndromes.clear();
            this->merge_list.clear();
            this->cluster_pcm.clear();
            // this->pluDecomposition = nullptr;
            this->cluster_check_idx_to_pcm_check_idx.clear();
            this->pcm_check_idx_to_cluster_check_idx.clear();
            this->cluster_bit_idx_to_pcm_bit_idx.clear();
        }

        /**
         * Grows the cluster by adding bit  nodes that are adjacent to boundary check nodes.
         * If bit_weights is provided, the bits are sorted by weight and only a single bit is added per growth step.
         * Otherwise, all bits adjacent to boundary check nodes are added.
         * @param bit_weights
         * @param bits_per_step
         * @return
         */
        void grow_cluster(const std::vector<double> &bit_weights = NULL_DOUBLE_VECTOR,
                          const int bits_per_step = std::numeric_limits<int>::max(),
                          const bool is_on_the_fly = false) {
            if (!this->active) {
                return;
            }
            // compute a list of bit nodes to grow the cluster to
            // this->eliminated_col_index = this->bit_nodes.size();
            this->compute_growth_candidate_bit_nodes();
            this->merge_list.clear();
            if (bit_weights == NULL_DOUBLE_VECTOR) {
                for (auto bit_index: this->candidate_bit_nodes) {
                    this->add_bit_node_to_cluster(bit_index);
                }
            } else {
                std::vector<double> cluster_bit_weights;
                cluster_bit_weights.reserve(this->candidate_bit_nodes.size());
                for (auto bit: this->candidate_bit_nodes) {
                    cluster_bit_weights.push_back(bit_weights[bit]);
                }
                auto sorted_indices = sort_indices(cluster_bit_weights);
                auto candidate_bit_nodes_vector = std::vector<int>(this->candidate_bit_nodes.begin(),
                                                                   this->candidate_bit_nodes.end());
                if (this->is_randomize_same_weight_indices) {
                    sorted_indices = randomize_same_weight_indices(sorted_indices, cluster_bit_weights);
                }
                int count = 0;
                for (auto i: sorted_indices) {
                    if (count == bits_per_step) {
                        break;
                    }
                    int bit_index = candidate_bit_nodes_vector[i];
                    this->add_bit_node_to_cluster(bit_index);
                    count++;
                }
            }
            this->merge_with_intersecting_clusters(is_on_the_fly);
        }

        static std::vector<int> randomize_same_weight_indices(const std::vector<int> &sorted_indices,
                                                              const std::vector<double> &cluster_bit_weights) {
            if (cluster_bit_weights.empty() || sorted_indices.empty()) {
                return {};
            }
            auto reshuffeled_indices = std::vector<int>(sorted_indices.size());
            auto same_weight_indices = std::vector<int>();
            auto other_indices = std::vector<int>();
            auto least_wt = cluster_bit_weights[sorted_indices[0]];

            for (auto sorted_index: sorted_indices) {
                if (cluster_bit_weights[sorted_index] == least_wt) {
                    same_weight_indices.push_back(sorted_index);
                } else {
                    other_indices.push_back(sorted_index);
                }
            }
            // if there are bits with the same weight, randomize their indices
            if (same_weight_indices.size() > 1) {
                std::shuffle(same_weight_indices.begin(), same_weight_indices.end(),
                             std::mt19937(std::random_device()()));
                for (auto i = 0; i < same_weight_indices.size(); i++) {
                    reshuffeled_indices[i] = same_weight_indices[i];
                }
                // add all other indices to the reshuffeled list
                for (auto i = 0; i < other_indices.size(); i++) {
                    reshuffeled_indices[i + same_weight_indices.size()] = other_indices[i];
                }
            }
            return reshuffeled_indices;
        }

        /**
         * Merge this cluster with all clusters that intersect with it.
         * Keeps the larger cluster and merges the smaller cluster into it.
         * That is, the (reduced) parity check matrix of the larger cluster is kept.
         * After merging, the on-the-fly elimination is applied to the larger cluster.
         *
         * If on the fly elimination is applied true is returned if the syndrome is in the cluster.
         */
        void merge_with_intersecting_clusters(const bool is_on_the_fly = false) {
            LsdCluster *larger = this;
            // merge with overlapping clusters while keeping the larger one always and deactivating the smaller ones
            for (auto cl: merge_list) {
                larger = merge_clusters(larger, cl);
            }
            if (is_on_the_fly) {
                // finally, we apply the on-the-fly elimination to the remaining cluster
                // if on the fly returns true, syndrome is in image and the cluster is valid
                larger->valid = larger->apply_on_the_fly_elimination();
            }
        }

        /**
         * Compute a list of candidate bit nodes to add to cluster as neighbours of boundary check nodes.
         * In case there are no new candidate bit nodes for a boundary check node, the check node is removed from the
         * boundary check node list.
         */
        void compute_growth_candidate_bit_nodes() {
            std::vector<int> boundary_checks_to_erase;
            this->candidate_bit_nodes.clear();
            // we check for new candidate bit nodes as neighbours of boundary check nodes
            for (auto check_index: boundary_check_nodes) {
                bool erase = true;
                for (auto &e: this->pcm.iterate_row(check_index)) {
                    // if bit is not in this cluster, add it to the candidate list.
                    if (this->global_bit_membership->at(e.col_index) != this) {
                        candidate_bit_nodes.insert(e.col_index);
                        erase = false;
                    }
                }
                // erase from boundary check nodes if no neighbouring bits are added to candidate list.
                if (erase) {
                    boundary_checks_to_erase.push_back(check_index);
                }
            }
            for (auto check_index: boundary_checks_to_erase) {
                this->boundary_check_nodes.erase(check_index);
            }
        }

        /**
         * Adds a bit node to the cluster and updates all lists accordingly.
         * @param bit_index
         * @return true if the bit was added to the cluster, false otherwise.
         */
        bool add_bit_node_to_cluster(const int bit_index, const bool in_merge = false) {
            auto bit_membership = this->global_bit_membership->at(bit_index);
            //if the bit is already in the cluster return.
            if (bit_membership == this) {
                // bit already in current cluster
                return false;
            }
            if (bit_membership == nullptr || in_merge) {
                //if the bit has not yet been assigned to a cluster or we are in merge mode we add it directly.
                this->add_bit(bit_index);
                if (this->global_timestep_bit_history != nullptr) {
                    // add bit to timestep history with timestep the map size -1
                    (*this->global_timestep_bit_history)[this->curr_timestep][this->cluster_id].push_back(bit_index);
                }
                // add incident checks as well, i.e., whole column to cluster pcm
                this->add_column_to_cluster_pcm(bit_index);
            } else {
                // if bit is in another cluster and we are not in merge mode, we add it to the merge list for later
                this->merge_list.insert(bit_membership);
            }

            return true;
        }

        /**
         * Merge this cluster with another cluster.
         * Keeps the larger cluster and merges the smaller cluster into it.
         * That is, the (reduced) parity check matrix of the larger cluster is kept.
         * @param cl2
         */
        static LsdCluster *merge_clusters(LsdCluster *cl1, LsdCluster *cl2) {
            LsdCluster *smaller = nullptr;
            LsdCluster *larger = nullptr;
            if (cl1->bit_nodes.size() < cl2->bit_nodes.size()) {
                smaller = cl1;
                larger = cl2;
            } else {
                smaller = cl2;
                larger = cl1;
            }

            // we merge the smaller into the larger cluster
            for (auto bit_index: smaller->bit_nodes) {
                larger->add_bit_node_to_cluster(bit_index, true);
            }
            // check nodes are added with the bits
            for (auto check_index: smaller->boundary_check_nodes) {
                larger->boundary_check_nodes.insert(check_index);
            }
            for (auto j: smaller->enclosed_syndromes) {
                larger->enclosed_syndromes.insert(j);
            }
            smaller->active = false; // smaller, absorbed cluster is deactivated
            smaller->absorbed_into_cluster = larger->cluster_id;
            smaller->got_inactive_in_timestep = smaller->curr_timestep;
            larger->nr_merges++;
            return larger;
        }


        /**
         * Adds single check to cluster and updates all lists accordingly
         * @param check_index
         * @param insert_boundary
         */
        int add_check(const int check_index, const bool insert_boundary = false) {
            if (insert_boundary) {
                this->boundary_check_nodes.insert(check_index);
            }
            auto inserted = this->check_nodes.insert(check_index);
            if (!inserted.second) {
                this->global_check_membership->at(check_index) = this;
                return this->pcm_check_idx_to_cluster_check_idx[check_index];
            }

            this->global_check_membership->at(check_index) = this;
            this->cluster_check_idx_to_pcm_check_idx.push_back(check_index);
            int local_idx = this->cluster_check_idx_to_pcm_check_idx.size() - 1;
            this->pcm_check_idx_to_cluster_check_idx.insert(
                    std::pair<int, int>{check_index, local_idx});
            return local_idx;
        }

        /**
         * Adds single bit to cluster and updates all lists accordingly.
         * @param bit_index
         */
        void add_bit(const int bit_index) {
            auto inserted = this->bit_nodes.insert(bit_index);
            if (!inserted.second) {
                return;
            }
            this->global_bit_membership->at(bit_index) = this;
            // also add to cluster pcm
            this->cluster_bit_idx_to_pcm_bit_idx.push_back(bit_index);
        }

        /**
         * Adds a column to the cluster parity check matrix.
         * The indices of the overall pcm are transferred to the local cluster pcm indices.
         * @param bit_index
         */
        void add_column_to_cluster_pcm(const int bit_index) {
            std::vector<int> col;
            for (auto &e: this->pcm.iterate_column(bit_index)) {
                int check_index = e.row_index;
                auto check_membership = this->global_check_membership->at(check_index);
                if (check_membership == this) {
                    // if already in cluster, add to cluster_pcm column of the bit and go to next
                    // an index error on the map here indicates an error in the program logic.
                    col.push_back(this->pcm_check_idx_to_cluster_check_idx[check_index]);
                    continue;
                } else if (check_membership != nullptr) {
                    // check is in another cluster
                    this->merge_list.insert(check_membership);
                }
                // if check is in another cluster or none, we add it and update cluster_pcm
                auto local_idx = this->add_check(check_index, true);
                col.push_back(local_idx);
            }
            this->cluster_pcm.push_back(col);
        }


        /**
         * Apply on the fly elimination to the cluster.
         * Assumes only a single bit has been added.
         * The previously conducted row operations are applied to the new column, the new column is eliminated.
         * together with the syndrome.
         *
         * @return True if the syndrome is in the image of the cluster parity check matrix.
         */
        bool apply_on_the_fly_elimination() {
            // add columns to existing decomposition matrix
            // new bits are appended to cluster_pcm
            for (auto idx = pluDecomposition.col_count; idx < this->bit_nodes.size(); idx++) {
                this->pluDecomposition.add_column_to_matrix(this->cluster_pcm[idx]);
            }

            // convert cluster syndrome to dense vector fitting the cluster pcm dimensions for solving the system.
            // std::vector<uint8_t> cluster_syndrome;
            this->cluster_pcm_syndrome.resize(this->check_nodes.size(), 0);
            for (auto s: this->enclosed_syndromes) {
                this->cluster_pcm_syndrome[this->pcm_check_idx_to_cluster_check_idx.at(s)] = 1;
            }
            auto syndrome_in_image = this->pluDecomposition.rref_with_y_image_check(this->cluster_pcm_syndrome,
                                                                                    pluDecomposition.cols_eliminated);
            return syndrome_in_image;
        }

        /**
         * Reorders the non-pivot columns of the eliminated cluster pcm according to the bit_weights.
         * @param bit_weights
         */
        void sort_non_pivot_cols(const std::vector<double> &bit_weights) {
            if (bit_weights.empty() or this->pluDecomposition.not_pivot_cols.size() < 2) {
                return;
            }
            // global index to cluster index mapping
            tsl::robin_map<int, int> global_to_cluster_colum_map;
            // local weight index to global bit index mapping
            tsl::robin_map<int, int> weight_idx_to_global_idx;
            // local weights to reorder
            std::vector<double> weights;
            for (auto not_pivot_col: this->pluDecomposition.not_pivot_cols) {
                auto global_idx = this->cluster_bit_idx_to_pcm_bit_idx[not_pivot_col];
                global_to_cluster_colum_map[global_idx] = not_pivot_col;
                weight_idx_to_global_idx[weights.size()] = global_idx;
                weights.push_back(bit_weights.at(global_idx));
            }
            auto sorted_indices = sort_indices(weights);
            std::vector<int> resorted_np_cls;
            resorted_np_cls.reserve(sorted_indices.size());
            // iterate over resorted indices
            // map local weight index to global bit index and then to cluster index
            for (auto idx: sorted_indices) {
                resorted_np_cls.push_back(global_to_cluster_colum_map[weight_idx_to_global_idx[idx]]);
            }
            this->pluDecomposition.not_pivot_cols = resorted_np_cls;
        }

        std::string to_string() {
            int count;
            std::stringstream ss{};
            ss << "........." << std::endl;
            ss << "Cluster ID: " << this->cluster_id << std::endl;
            ss << "Active: " << this->active << std::endl;
            ss << "Enclosed syndromes: ";
            for (auto i: this->enclosed_syndromes) ss << i << " ";
            ss << std::endl;
            ss << "Cluster bits: ";
            for (auto i: this->bit_nodes) ss << i << " ";
            ss << std::endl;
            ss << "Cluster checks: ";
            for (auto i: this->check_nodes) ss << i << " ";
            ss << std::endl;
            ss << "Candidate bits: ";
            for (auto i: this->candidate_bit_nodes) ss << i << " ";
            ss << std::endl;
            ss << "Boundary Checks: ";
            for (auto i: this->boundary_check_nodes) ss << i << " ";
            ss << std::endl;

            ss << "Cluster bit idx to pcm bit idx: ";
            count = 0;
            for (auto bit_idx: this->cluster_bit_idx_to_pcm_bit_idx) {
                ss << "{" << count << "," << bit_idx << "}";
                count++;
            }
            ss << std::endl;

            ss << "Cluster check idx to pcm check idx: ";
            count = 0;
            for (auto check_idx: this->cluster_check_idx_to_pcm_check_idx) {
                ss << "{" << count << "," << check_idx << "}";
                count++;
            }
            ss << std::endl;

            ss << "Cluster syndrome: ";
            for (auto check_idx: this->cluster_pcm_syndrome) {
                ss << unsigned(check_idx);
            }
            ss << std::endl;
            return ss.str();
        }
    };

    struct ClusterStatistics {
    public:
        int final_bit_count = 0; // nr of bits in 'final' cluster version, i.e., before solving for solution
        int undergone_growth_steps = 0; // nr of growth steps the cluster underwent
        int nr_merges = 0; // nr of merges the cluster underwent
        std::vector<int> size_history = {}; // history of cluster sizes from 0 to final bit count
        bool active = false; // if cluster is active, i.e., not merged into another cluster
        int got_valid_in_timestep = -1; // timestep in which cluster got valid
        int got_inactive_in_timestep = -1; // timestep in which cluster got inactive, i.e., was absorbed by another
        int absorbed_by_cluster = -1; // cluster_id of the cluster the current one was merged into
        int nr_of_non_zero_check_matrix_entries = 0; // nr of non zero entries in the cluster pcm
        double cluster_pcm_sparsity = 0; // nr of non zero entries in the cluster pcm
        std::vector<uint8_t> solution{}; // local recovery, solution of cluster
    };

    struct Statistics {
        std::unordered_map<int, ClusterStatistics> individual_cluster_stats; // clusterid <> stats
        std::unordered_map<int, std::unordered_map<int, std::vector<int>>> global_timestep_bit_history; //timestep <> (clusterid <> added bits)
        long elapsed_time{};
        osd::OsdMethod lsd_method;
        int lsd_order{};
        std::vector<double> bit_llrs;
        std::vector<uint8_t> error; // the original error
        std::vector<uint8_t> syndrome;// the syndrome to decode
        std::vector<uint8_t> compare_recover; // a recovery vector to compare against

        void clear() {
            this->individual_cluster_stats.clear();
            this->global_timestep_bit_history.clear();
            this->elapsed_time = 0.0;
            this->lsd_method = osd::OsdMethod::COMBINATION_SWEEP;
            this->lsd_order = 0;
            this->bit_llrs = {};
            this->error = {};
            this->syndrome = {};
            this->compare_recover = {};
        }

        [[nodiscard]] std::string toString() {
            // build json like string object from individual cluster stats and global timestep bit history
            std::string result = "{";
            result += "\"elapsed_time_mu\":" + std::to_string(this->elapsed_time) + ",";
            result += "\"lsd_method\":" + std::to_string(static_cast<int>(this->lsd_method)) + ",";
            result += "\"lsd_order\":" + std::to_string(this->lsd_order) + ",";
            // print bit_llrs
            result += "\"bit_llrs\":[";
            for (auto i = 0; i < this->bit_llrs.size(); i++) {
                result += std::to_string(this->bit_llrs.at(i));
                if (i < this->bit_llrs.size() - 1) {
                    result += ",";
                }
            }
            result += "],";
            result += "\"error\":[";
            for (auto i = 0; i < this->error.size(); i++) {
                result += std::to_string(this->error.at(i));
                if (i < this->error.size() - 1) {
                    result += ",";
                }
            }
            result += "],";
            // print syndrome
            result += "\"syndrome\":[";
            for (auto i = 0; i < this->syndrome.size(); i++) {
                result += std::to_string(this->syndrome.at(i));
                if (i < this->syndrome.size() - 1) {
                    result += ",";
                }
            }
            result += "],";
            // print compare_recover
            result += "\"compare_recover\":[";
            for (auto i = 0; i < this->compare_recover.size(); i++) {
                result += std::to_string(this->compare_recover.at(i));
                if (i < this->compare_recover.size() - 1) {
                    result += ",";
                }
            }
            result += "],";
            result += "\"individual_cluster_stats\":{";
            for (auto &kv: this->individual_cluster_stats) {
                result += "\"" + std::to_string(kv.first) + "\":{";
                result += "\"active\":" + std::to_string(kv.second.active) + ",";
                result += "\"final_bit_count\":" + std::to_string(kv.second.final_bit_count) + ",";
                result += "\"undergone_growth_steps\":" + std::to_string(kv.second.undergone_growth_steps) + ",";
                result += "\"nr_merges\":" + std::to_string(kv.second.nr_merges) + ",";
                result += "\"got_valid_in_timestep\":" + std::to_string(kv.second.got_valid_in_timestep) + ",";
                result += "\"absorbed_by_cluster\":" + std::to_string(kv.second.absorbed_by_cluster) + ",";
                result += "\"got_inactive_in_timestep\":" + std::to_string(kv.second.got_inactive_in_timestep) + ",";
                result += "\"nr_of_non_zero_check_matrix_entries\":" +
                          std::to_string(kv.second.nr_of_non_zero_check_matrix_entries) + ",";
                result += "\"cluster_pcm_sparsity\":" + std::to_string(kv.second.cluster_pcm_sparsity) + ",";
                // print solution vector
                result += "\"solution\":[";
                for (auto i = 0; i < kv.second.solution.size(); i++) {
                    result += std::to_string(kv.second.solution.at(i));
                    if (i < kv.second.solution.size() - 1) {
                        result += ",";
                    }
                }
                result += "],";
                result += "\"size_history\":[";
                for (auto &s: kv.second.size_history) {
                    result += std::to_string(s) + ",";
                }
                result.pop_back();
                result += "]},";
            }
            result.pop_back();
            result += "},";
            result += "\"global_timestep_bit_history\":{";
            for (auto &kv: this->global_timestep_bit_history) {
                result += "\"" + std::to_string(kv.first) + "\":{";
                for (auto &kv2: kv.second) {
                    result += "\"" + std::to_string(kv2.first) + "\":[";
                    for (auto &b: kv2.second) {
                        result += std::to_string(b) + ",";
                    }
                    //remove last , from result
                    result.pop_back();
                    result += "],";
                }
                result.pop_back();
                result += "},";
            }
            result.pop_back();
            result += "}";
            result += "}";
            return result;
        }
    };

    // todo move this to separate file
    class LsdDecoder {
    private:
        ldpc::bp::BpSparse &pcm;
        bool do_stats;

    public:
        std::vector<uint8_t> decoding{};
        Statistics statistics{};
        int bit_count{};
        osd::OsdMethod lsd_method;
        int lsd_order;

        void reset_cluster_stats() {
            this->statistics.clear();
        }

        explicit LsdDecoder(ldpc::bp::BpSparse &parity_check_matrix,
                            osd::OsdMethod lsdMethod = osd::OsdMethod::COMBINATION_SWEEP,
                            int lsd_order = 0) : pcm(parity_check_matrix),
                                                 lsd_method(lsdMethod),
                                                 lsd_order(lsd_order) {
            this->bit_count = pcm.n;
            this->decoding.resize(this->bit_count);
            this->do_stats = false;
        }

        osd::OsdMethod getLsdMethod() const {
            return lsd_method;
        }

        void setLsdMethod(osd::OsdMethod lsdMethod) {
            lsd_method = lsdMethod;
        }

        void set_do_stats(const bool on) {
            this->do_stats = on;
        }

        bool get_do_stats() const {
            return this->do_stats;
        }

        void print_cluster_stats() {
            std::cout << this->statistics.toString() << std::endl;
        }

        void update_growth_stats(const LsdCluster *cl) {
            this->statistics.individual_cluster_stats[cl->cluster_id].undergone_growth_steps++;
            this->statistics.individual_cluster_stats[cl->cluster_id].size_history.push_back(cl->bit_nodes.size());
            this->statistics.individual_cluster_stats[cl->cluster_id].active = true;
            this->statistics.individual_cluster_stats[cl->cluster_id].absorbed_by_cluster = cl->absorbed_into_cluster;
            this->statistics.individual_cluster_stats[cl->cluster_id].got_inactive_in_timestep = cl->got_inactive_in_timestep;
        }

        void update_final_stats(const LsdCluster *cl) {
            this->statistics.individual_cluster_stats[cl->cluster_id].final_bit_count = cl->bit_nodes.size();
            this->statistics.individual_cluster_stats[cl->cluster_id].nr_merges = cl->nr_merges;
            int nr_nonzero_elems = gf2dense::count_non_zero_matrix_entries(cl->cluster_pcm);
            this->statistics.individual_cluster_stats[cl->cluster_id].nr_of_non_zero_check_matrix_entries =
                    nr_nonzero_elems;
            auto size = cl->pluDecomposition.col_count * cl->pluDecomposition.row_count;
            if (size > 0) {
                this->statistics.individual_cluster_stats[cl->cluster_id].cluster_pcm_sparsity =
                        1.0 - ((static_cast<double>(nr_nonzero_elems)) / static_cast<double>(size));
            }
        }

        std::vector<uint8_t> &on_the_fly_decode(std::vector<uint8_t> &syndrome,
                                                const std::vector<double> &bit_weights = NULL_DOUBLE_VECTOR) {
            return this->lsd_decode(syndrome, bit_weights, 1, true);
        }

        std::vector<uint8_t> &
        lsd_decode(std::vector<uint8_t> &syndrome,
                   const std::vector<double> &bit_weights = NULL_DOUBLE_VECTOR,
                   const int bits_per_step = 1,
                   const bool is_on_the_fly = true) {
            auto start_time = std::chrono::high_resolution_clock::now();
            this->statistics.clear();
            this->statistics.syndrome = syndrome;

            fill(this->decoding.begin(), this->decoding.end(), 0);

            std::vector<LsdCluster *> clusters;
            std::vector<LsdCluster *> invalid_clusters;
            auto global_bit_membership = std::make_shared<std::vector<LsdCluster *>>(
                    std::vector<LsdCluster *>(this->pcm.n));
            auto global_check_membership = std::make_shared<std::vector<LsdCluster *>>(
                    std::vector<LsdCluster *>(this->pcm.m));
            // timestep to added bits history for stats
            auto *global_timestep_bits_history = new std::unordered_map<int, std::unordered_map<int, std::vector<int>>>{};
            auto timestep = 0;
            for (auto i = 0; i < this->pcm.m; i++) {
                if (syndrome[i] == 1) {
                    auto *cl = new LsdCluster(this->pcm, i, global_check_membership, global_bit_membership);
                    clusters.push_back(cl);
                    invalid_clusters.push_back(cl);
                    if (this->do_stats) {
                        this->statistics.individual_cluster_stats[cl->cluster_id] = ClusterStatistics();
                        cl->global_timestep_bit_history = global_timestep_bits_history;
                    }
                }
            }

            while (!invalid_clusters.empty()) {
                std::vector<int> new_bits;
                for (auto cl: invalid_clusters) {
                    if (cl->active) {
                        cl->curr_timestep = timestep; // for stats
                        cl->grow_cluster(bit_weights, bits_per_step, is_on_the_fly);
                    }
                }
                invalid_clusters.clear();
                for (auto cl: clusters) {
                    if (this->do_stats) {
                        this->update_growth_stats(cl);
                    }
                    if (cl->active && !cl->valid) {
                        invalid_clusters.push_back(cl);
                    } else if (cl->active && cl->valid && this->do_stats) {
                        this->statistics.individual_cluster_stats[cl->cluster_id].got_valid_in_timestep = timestep;
                    }
                    if (do_stats) {
                        this->statistics.individual_cluster_stats[cl->cluster_id].active = cl->active;
                    }
                }
                std::sort(invalid_clusters.begin(), invalid_clusters.end(),
                          [](const LsdCluster *lhs, const LsdCluster *rhs) {
                              return lhs->bit_nodes.size() < rhs->bit_nodes.size();
                          });
                timestep++; // for stats
            }

            if (lsd_order == 0) {
                this->statistics.lsd_order = 0;
                this->statistics.lsd_method = osd::OSD_0;
                for (auto cl: clusters) {
                    if (do_stats) {
                        this->update_final_stats(cl);
                    }
                    if (cl->active) {
                        auto solution = cl->pluDecomposition.lu_solve(cl->cluster_pcm_syndrome);
                        this->statistics.individual_cluster_stats[cl->cluster_id].solution = solution;
                        for (auto i = 0; i < solution.size(); i++) {
                            if (solution[i] == 1) {
                                int bit_idx = cl->cluster_bit_idx_to_pcm_bit_idx[i];
                                this->decoding[bit_idx] = 1;
                            }
                        }
                    }
                }
            } else {
                this->statistics.lsd_order = lsd_order;
                this->statistics.lsd_method = this->lsd_method;
                this->apply_lsdw(clusters, lsd_order, bit_weights);
            }
            auto end_time = std::chrono::high_resolution_clock::now();

            if (do_stats) {
                this->statistics.global_timestep_bit_history = *global_timestep_bits_history;
                this->statistics.bit_llrs = bit_weights;

            }
            // always take time
            this->statistics.elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(
                    end_time - start_time).count();
            // cleanup
            for (auto cl: clusters) {
                delete cl;
            }
            global_bit_membership->clear();
            global_check_membership->clear();
            delete global_timestep_bits_history;
            return this->decoding;
        }

        void apply_lsdw(const std::vector<LsdCluster *> &clusters,
                        int lsd_order,
                        const std::vector<double> &bit_weights, std::size_t timestep = 0) {
            //cluster growth stage
            for (auto cl: clusters) {
                if (cl->active) {
                    // first we measure the dimension of each cluster
                    auto cluster_dimension = cl->pluDecomposition.not_pivot_cols.size();
                    //if the cluster dimension is smaller than the lsd order, we grow the cluster until it reaches
                    // the lsd order. The number of bits added is limited to be at most the lsd order.
                    auto cluster_growth_count = 0;
                    auto initial_cluster_size = cl->bit_nodes.size();
                    while (cluster_dimension < lsd_order &&
                           cluster_growth_count < lsd_order &&
                           cl->bit_nodes.size() < this->pcm.n &&
                           cluster_growth_count <= initial_cluster_size) {
                        cl->curr_timestep = timestep; // for stats
                        cl->grow_cluster(bit_weights, 1, true);
                        cluster_dimension = cl->pluDecomposition.not_pivot_cols.size();
                        cluster_growth_count++;
                        if (this->do_stats) {
                            this->update_growth_stats(cl);
                        }
                        timestep++;
                    }
                }
            }
            // apply lsd-w to clusters
            for (auto cl: clusters) {
                if (do_stats) {
                    this->update_final_stats(cl);
                }
                if (cl->active) {
                    cl->sort_non_pivot_cols(bit_weights);
                    auto cl_osd_decoder = osd::DenseOsdDecoder(
                            cl->cluster_pcm,
                            cl->pluDecomposition,
                            this->lsd_method,
                            lsd_order,
                            cl->bit_nodes.size(),
                            cl->check_nodes.size(),
                            bit_weights);
                    auto res = cl_osd_decoder.osd_decode(cl->cluster_pcm_syndrome);

                    for (auto i = 0; i < res.size(); i++) {
                        if (res[i] == 1) {
                            int bit_idx = cl->cluster_bit_idx_to_pcm_bit_idx[i];
                            this->decoding[bit_idx] = 1;
                        }
                    }
                }
            }
        }
    };
}//end namespace lsd

#endif