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
#include "bp.hpp"
#include "osd.hpp"
#include <robin_map.h>
#include <robin_set.h>
#include <numeric>

#include "gf2sparse_linalg.hpp"
#include "bp.hpp"
#include "gf2dense.hpp"

namespace ldpc::uf {
    const std::vector<double> NULL_DOUBLE_VECTOR = {};

    std::vector<int> sort_indices(std::vector<double> &B) {
        std::vector<int> indices(B.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int i, int j) { return B[i] < B[j]; });
        return indices;
    }

    // TODO this should probably become a class
    struct Cluster {
        ldpc::bp::BpSparse &pcm;
        int cluster_id;
        bool active; // if merge one becomes deactivated
        bool valid; //
        tsl::robin_set<int> bit_nodes;
        tsl::robin_set<int> check_nodes;
        tsl::robin_set<int> boundary_check_nodes;
        std::vector<int> candidate_bit_nodes;
        tsl::robin_set<int> enclosed_syndromes;
        tsl::robin_map<int, int> spanning_tree_check_roots;
        tsl::robin_set<int> spanning_tree_bits;
        tsl::robin_set<int> spanning_tree_leaf_nodes;
        Cluster **global_check_membership; // store which cluster a check belongs to
        Cluster **global_bit_membership; // store which cluster a bit belongs to
        tsl::robin_set<Cluster *> merge_list;
        std::vector<int> cluster_decoding;
        std::vector<int> matrix_to_cluster_bit_map;
        tsl::robin_map<int, int> cluster_to_matrix_bit_map;
        std::vector<int> matrix_to_cluster_check_map;
        tsl::robin_map<int, int> cluster_to_matrix_check_map;
        // parity check matrix corresponding to the cluster. Indices are local to the cluster.
        gf2dense::CscMatrix cluster_pcm;
        std::vector<uint8_t> cluster_pcm_syndrome;
        std::vector<int> cluster_check_idx_to_pcm_check_idx;
        tsl::robin_map<int, int> pcm_check_idx_to_cluster_check_idx;
        std::vector<int> cluster_bit_idx_to_pcm_bit_idx;
        gf2dense::PluDecomposition *pluDecomposition = nullptr;

        Cluster() = default;

        Cluster(ldpc::bp::BpSparse &parity_check_matrix,
                int syndrome_index,
                Cluster **ccm, // global check cluster membership
                Cluster **bcm, // global bit cluster membership
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
        }

        ~Cluster() {
            this->bit_nodes.clear();
            this->check_nodes.clear();
            this->boundary_check_nodes.clear();
            this->candidate_bit_nodes.clear();
            this->enclosed_syndromes.clear();
            this->merge_list.clear();
            this->cluster_pcm.clear();
            this->pluDecomposition = nullptr;
            this->cluster_check_idx_to_pcm_check_idx.clear();
            this->pcm_check_idx_to_cluster_check_idx.clear();
            this->cluster_bit_idx_to_pcm_bit_idx.clear();
        }

        [[nodiscard]] int parity() const {
            return static_cast<int>(this->enclosed_syndromes.size() % 2);
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
            Cluster *larger = this;
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
        static Cluster *merge_clusters(Cluster *cl1, Cluster *cl2) {
            Cluster *smaller;
            Cluster *larger;
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
            // ldpc::sparse_matrix_util::print_vector(col);
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
            int eliminated_cols = 0;
            if (this->pluDecomposition == nullptr) {
                // no existing decomposition yet so we create one
                this->pluDecomposition = new ldpc::gf2dense::PluDecomposition(this->check_nodes.size(),
                                                                              this->bit_nodes.size(),
                                                                              this->cluster_pcm);
            }else{
                eliminated_cols = this->pluDecomposition->cols_eliminated;
                // add columns to existing decomposition matrix
                for (auto idx = eliminated_cols; idx < this->bit_nodes.size(); idx++) {
                    this->pluDecomposition->add_column_to_matrix(this->cluster_pcm[idx]);
                }
            }
            // convert cluster syndrome to dense vector fitting the cluster pcm dimensions for solving the system.
            // std::vector<uint8_t> cluster_syndrome;
            this->cluster_pcm_syndrome.resize(this->check_nodes.size(), 0);
            for (auto s: this->enclosed_syndromes) {
                this->cluster_pcm_syndrome[this->pcm_check_idx_to_cluster_check_idx.at(s)] = 1;
            }
            auto syndrome_in_image = this->pluDecomposition->rref_with_y_image_check(this->cluster_pcm_syndrome, eliminated_cols);
            return syndrome_in_image;
        }
        void print();

    };


    // todo move this to separate file
    class UfDecoder {

    private:
        bool weighted;
        ldpc::bp::BpSparse &pcm;

    public:
        std::vector<uint8_t> decoding;
        int bit_count;
        int check_count;

        UfDecoder(ldpc::bp::BpSparse &parity_check_matrix) : pcm(parity_check_matrix) {
            this->bit_count = pcm.n;
            this->check_count = pcm.m;
            this->decoding.resize(this->bit_count);
            this->weighted = false;
        }

        
        std::vector<uint8_t> &on_the_fly_decode(const std::vector<uint8_t> &syndrome,
                                                const std::vector<double> &bit_weights = NULL_DOUBLE_VECTOR) {
            return this->matrix_decode(syndrome, bit_weights, 1, true);
        }

        std::vector<uint8_t> &
        matrix_decode(const std::vector<uint8_t> &syndrome,
                      const std::vector<double> &bit_weights = NULL_DOUBLE_VECTOR,
                      const int bits_per_step = 1,
                      const bool is_on_the_fly = false) {

            fill(this->decoding.begin(), this->decoding.end(), 0);

            std::vector<Cluster *> clusters;
            std::vector<Cluster *> invalid_clusters;
            auto **global_bit_membership = new Cluster *[pcm.n]();
            auto **global_check_membership = new Cluster *[pcm.m]();

            for (auto i = 0; i < this->pcm.m; i++) {
                if (syndrome[i] == 1) {
                    auto *cl = new Cluster(this->pcm, i, global_check_membership, global_bit_membership);
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
                          [](const Cluster *lhs, const Cluster *rhs) {
                              return lhs->bit_nodes.size() < rhs->bit_nodes.size();
                          });
            }

            for (auto cl: clusters) {
                if (cl->active) {
                    cl->print();
                    gf2dense::print_csc(cl->cluster_pcm);

                    std::cout<<std::endl;

                    gf2dense::print_csr(cl->pluDecomposition->L);

                    std::cout<<std::endl;

                    gf2dense::print_csr(cl->pluDecomposition->U);
                    std::cout<<"Pivots: ";
                    ldpc::sparse_matrix_util::print_vector(cl->pluDecomposition->pivot_cols);

                    auto temp_pcm = ldpc::gf2sparse::csc_to_gf2sparse(cl->cluster_pcm);

                    auto solution = cl->pluDecomposition->lu_solve(cl->cluster_pcm_syndrome);

                    ldpc::sparse_matrix_util::print_vector(temp_pcm.mulvec(solution));

                    for (auto i = 0; i < solution.size(); i++) {
                        if (solution[i] == 1) {
                            int bit_idx = cl->cluster_bit_idx_to_pcm_bit_idx[i];
                            this->decoding[bit_idx] = 1;
                        }
                    }
                }
                delete cl; //delete the cluster now that we have the solution.
            }
            delete[] global_bit_membership;
            delete[] global_check_membership;
            return this->decoding;
        }

    };

    void Cluster::print() {
        int count;
        std::cout << "........." << std::endl;
        std::cout << "Cluster ID: " << this->cluster_id << std::endl;
        std::cout << "Active: " << this->active << std::endl;
        std::cout << "Enclosed syndromes: ";
        for (auto i: this->enclosed_syndromes) std::cout << i << " ";
        std::cout << std::endl;
        std::cout << "Cluster bits: ";
        for (auto i: this->bit_nodes) std::cout << i << " ";
        std::cout << std::endl;
        std::cout << "Cluster checks: ";
        for (auto i: this->check_nodes) std::cout << i << " ";
        std::cout << std::endl;
        std::cout << "Candidate bits: ";
        for (auto i: this->candidate_bit_nodes) std::cout << i << " ";
        std::cout << std::endl;
        std::cout << "Boundary Checks: ";
        for (auto i: this->boundary_check_nodes) std::cout << i << " ";
        std::cout << std::endl;

        std::cout<<"Cluster bit idx to pcm bit idx: ";
        count = 0;
        for (auto bit_idx: this->cluster_bit_idx_to_pcm_bit_idx){
            std::cout<<"{"<<count<<","<<bit_idx<<"}";
            count++;
        }
        std::cout<<std::endl;

        std::cout<<"Cluster check idx to pcm check idx: ";
        count = 0;
        for (auto check_idx: this->cluster_check_idx_to_pcm_check_idx){
            std::cout<<"{"<<count<<","<<check_idx<<"}";
            count++;
        }
        std::cout<<std::endl;

        std::cout<<"Cluster syndrome: ";
        count = 0;
        for (auto check_idx: this->cluster_pcm_syndrome){
            std::cout<<unsigned(check_idx);
        }
        std::cout<<std::endl;

    }


}//end namespace uf

#endif