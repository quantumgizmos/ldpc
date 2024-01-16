#ifndef UF2_H
#define UF2_H

#include <iostream>
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

namespace ldpc::lsd {

    const std::vector<double> NULL_DOUBLE_VECTOR = {};

    std::vector<int> sort_indices(std::vector<double> &B) {
        std::vector<int> indices(B.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int i, int j) { return B[i] < B[j]; });
        return indices;
    }

    // TODO this should probably become a class
    struct LsdCluster {
        ldpc::bp::BpSparse &pcm;
        int cluster_id;
        bool active; // if merge one becomes deactivated
        bool valid; //
        tsl::robin_set<int> bit_nodes;
        tsl::robin_set<int> check_nodes;
        tsl::robin_set<int> boundary_check_nodes;
        std::vector<int> candidate_bit_nodes;
        tsl::robin_set<int> enclosed_syndromes;
        LsdCluster **global_check_membership; // store which cluster a check belongs to
        LsdCluster **global_bit_membership; // store which cluster a bit belongs to
        tsl::robin_set<LsdCluster *> merge_list;
        gf2dense::CscMatrix cluster_pcm;
        std::vector<uint8_t> cluster_pcm_syndrome;
        std::vector<int> cluster_check_idx_to_pcm_check_idx;
        tsl::robin_map<int, int> pcm_check_idx_to_cluster_check_idx;
        std::vector<int> cluster_bit_idx_to_pcm_bit_idx;
        gf2dense::PluDecomposition pluDecomposition;

        LsdCluster() = default;

        LsdCluster(ldpc::bp::BpSparse &parity_check_matrix,
                int syndrome_index,
                LsdCluster **ccm, // global check cluster membership
                LsdCluster **bcm, // global bit cluster membership
                bool on_the_fly = false) :
                pcm(parity_check_matrix) {
            this->active = true;
            this->valid = false;
            this->cluster_id = syndrome_index;
            this->boundary_check_nodes.insert(syndrome_index);
            this->enclosed_syndromes.insert(syndrome_index);
            this->global_check_membership = ccm;
            this->global_bit_membership = bcm;
            this->check_nodes.insert(syndrome_index);
            this->global_check_membership[syndrome_index] = this;
            this->cluster_pcm_syndrome.clear();
            this->pcm_check_idx_to_cluster_check_idx.insert(
                    std::pair<int, int>{syndrome_index, 0});
            this->cluster_check_idx_to_pcm_check_idx.push_back(syndrome_index);

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
                int count = 0;
                for (auto i: sorted_indices) {
                    if (count == bits_per_step) {
                        break;
                    }
                    int bit_index = this->candidate_bit_nodes[i];
                    this->add_bit_node_to_cluster(bit_index);
                    count++;
                }
            }
            this->merge_with_intersecting_clusters(is_on_the_fly);
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
                    if (this->global_bit_membership[e.col_index] != this) {
                        candidate_bit_nodes.push_back(e.col_index);
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
            auto bit_membership = this->global_bit_membership[bit_index];
            //if the bit is already in the cluster return.
            if (bit_membership == this) {
                // bit already in current cluster
                return false;
            }
            if (bit_membership == nullptr) {
                //if the bit has not yet been assigned to a cluster we add it.
                // if we are in merge mode we add it
                this->add_bit(bit_index);
            } else {
                //if the bit already exists in a cluster, we mark down that this cluster should be
                //merged with the exisiting cluster.
                if (in_merge) {
                    // if we are in merge mode we add it anyways
                    this->add_bit(bit_index);
                } else {
                    // otherwise, we add its cluster to merge list and do not add directly.
                    this->merge_list.insert(bit_membership);
                }
            }
            // add incident checks
            this->add_column_to_cluster_pcm(bit_index);
            return true;
        }

        /**
         * Merge this cluster with another cluster.
         * Keeps the larger cluster and merges the smaller cluster into it.
         * That is, the (reduced) parity check matrix of the larger cluster is kept.
         * @param cl2
         */
        static LsdCluster *merge_clusters(LsdCluster *cl1, LsdCluster *cl2) {
            LsdCluster *smaller;
            LsdCluster *larger;
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
                this->global_check_membership[check_index] = this;
                return this->pcm_check_idx_to_cluster_check_idx[check_index];
            }

            this->global_check_membership[check_index] = this;
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
            this->global_bit_membership[bit_index] = this;
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
                auto check_membership = this->global_check_membership[check_index];
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

        std::string to_string();
    };


    // todo move this to separate file
    class LsdDecoder {

    private:
        bool weighted;
        ldpc::bp::BpSparse &pcm;

    public:
        std::vector<uint8_t> decoding;
        std::vector<int> cluster_size_stats;
        int bit_count;
        int check_count;

        LsdDecoder(ldpc::bp::BpSparse &parity_check_matrix) : pcm(parity_check_matrix) {
            this->bit_count = pcm.n;
            this->check_count = pcm.m;
            this->decoding.resize(this->bit_count);
            this->weighted = false;
        }


        std::vector<uint8_t> &on_the_fly_decode(std::vector<uint8_t> &syndrome,
                                                const std::vector<double> &bit_weights = NULL_DOUBLE_VECTOR) {
            return this->lsd_decode(syndrome, bit_weights, 1, true);
        }

        std::vector<uint8_t> &
        lsd_decode(std::vector<uint8_t> &syndrome,
                      const std::vector<double> &bit_weights = NULL_DOUBLE_VECTOR,
                      const int bits_per_step = 1,
                      const bool is_on_the_fly = true,
                      const int lsd_order = 0) {

            this->cluster_size_stats.clear();
            fill(this->decoding.begin(), this->decoding.end(), 0);

            std::vector<LsdCluster *> clusters;
            std::vector<LsdCluster *> invalid_clusters;
            auto **global_bit_membership = new LsdCluster *[pcm.n]();
            auto **global_check_membership = new LsdCluster *[pcm.m]();

            for (auto i = 0; i < this->pcm.m; i++) {
                if (syndrome[i] == 1) {
                    auto *cl = new LsdCluster(this->pcm, i, global_check_membership, global_bit_membership);
                    clusters.push_back(cl);
                    invalid_clusters.push_back(cl);
                }
            }

            while (!invalid_clusters.empty()) {
                for (auto cl: invalid_clusters) {
                    if (cl->active) {
                        cl->grow_cluster(bit_weights, bits_per_step, is_on_the_fly);
                    }
                }
                invalid_clusters.clear();
                for (auto cl: clusters) {
                    if (cl->active && !cl->valid) {
                        invalid_clusters.push_back(cl);
                    }
                }
                std::sort(invalid_clusters.begin(), invalid_clusters.end(),
                          [](const LsdCluster *lhs, const LsdCluster *rhs) {
                              return lhs->bit_nodes.size() < rhs->bit_nodes.size();
                          });
            }

            if(lsd_order == 0){
                for (auto cl: clusters) {
                    if (cl->active) {
                        this->cluster_size_stats.push_back(cl->bit_nodes.size());
                        auto solution = cl->pluDecomposition.lu_solve(cl->cluster_pcm_syndrome);
                        for (auto i = 0; i < solution.size(); i++) {
                            if (solution[i] == 1) {
                                int bit_idx = cl->cluster_bit_idx_to_pcm_bit_idx[i];
                                this->decoding[bit_idx] = 1;
                            }
                        }
                    }
                    delete cl; //delete the cluster now that we have the solution.
                }
            }
            else{
                for(auto cl: clusters){
                    if(cl->active){
                        
                        // first we measure the dimension of each cluster
                        int cluster_dimension = cl->pluDecomposition.not_pivot_cols.size();
                        
                        //if the cluster dimension is smaller than the lsd order, we grow the cluster until it reaches the lsd order. The number of bits added is limited to be at most the lsd order.

                        int cluster_growth_count = 0;
                        while(cluster_dimension < lsd_order && cluster_growth_count < lsd_order && cl->bit_nodes.size()<this->pcm.n){
                            cl->grow_cluster(bit_weights, 1, is_on_the_fly);
                            cluster_dimension = cl->pluDecomposition.not_pivot_cols.size();
                            cluster_growth_count++;
                        }

                        // std::cout<<cl->to_string()<<std::endl;

                        int search_depth = std::min(lsd_order, cluster_dimension);
                        // we now solve for the LSD-0 solution

                        auto solution = cl->pluDecomposition.lu_solve(cl->cluster_pcm_syndrome);

                        int solution_weight = 0;
                        for (auto i = 0; i < solution.size(); i++) {
                            if (solution[i] == 1) {
                                solution_weight++;
                            }
                        }

                        int min_solution_weight = solution_weight;
                        auto best_solution = solution;
                        
                        //measure the rank of the cluster
                        int cluster_rank = cl->pluDecomposition.pivot_cols.size();

                        //iterate over all of the weight-1 perturbations to the LSD-0 solution

                        for(int j = 0; j < search_depth; j++){
                            
                            int np_col = cl->pluDecomposition.not_pivot_cols[j];

                            //calculate the correction to the cluster syndrome
                            auto lsd_w_cluster_syndrome = cl->cluster_pcm_syndrome;
                            for(int cluster_check_idx: cl->cluster_pcm[np_col]){
                                lsd_w_cluster_syndrome[cluster_check_idx] ^= 1;
                            }

                            //solve for the LSD-w solution
                            auto lsd_w_solution = cl->pluDecomposition.lu_solve(lsd_w_cluster_syndrome);

                            //add the correction to the non-information set
                            lsd_w_cluster_syndrome[cluster_rank + j ] = 1;

                            //calculate the weight of the LSD-w solution
                            solution_weight = 0;
                            for (auto i = 0; i < lsd_w_solution.size(); i++) {
                                if (lsd_w_solution[i] == 1) {
                                    solution_weight++;
                                }
                            }
                            if(solution_weight < min_solution_weight){
                                min_solution_weight = solution_weight;
                                best_solution = lsd_w_solution;
                            }

                        }

                        for (auto i = 0; i < best_solution.size(); i++) {
                            if (best_solution[i] == 1) {
                                int bit_idx = cl->cluster_bit_idx_to_pcm_bit_idx[i];
                                this->decoding[bit_idx] = 1;
                            }
                        }

                    }
                    delete cl;
                }
            }
            delete[] global_bit_membership;
            delete[] global_check_membership;
            return this->decoding;
        }

    };

    std::string LsdCluster::to_string() {
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


}//end namespace lsd

#endif