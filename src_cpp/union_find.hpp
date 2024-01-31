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

namespace ldpc::uf {

    const std::vector<double> EMPTY_DOUBLE_VECTOR = {};
    tsl::robin_set<int> EMPTY_INT_ROBIN_SET = {};

    std::vector<int> sort_indices(std::vector<double> &B) {
        std::vector<int> indices(B.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int i, int j) { return B[i] < B[j]; });
        return indices;
    }

    struct Cluster {
        ldpc::bp::BpSparse &pcm;
        tsl::robin_set<int> &planar_code_boundary_bits;
        int cluster_id;
        bool contains_boundary_bits;
        bool active;
        bool valid;
        tsl::robin_set<int> bit_nodes;
        tsl::robin_set<int> check_nodes;
        tsl::robin_set<int> boundary_check_nodes;
        std::vector<int> candidate_bit_nodes;
        tsl::robin_set<int> enclosed_syndromes;
        tsl::robin_map<int, int> spanning_tree_check_roots;
        tsl::robin_set<int> spanning_tree_bits;
        tsl::robin_set<int> spanning_tree_leaf_nodes;
        int spanning_tree_boundary_bit;
        Cluster **global_check_membership;
        Cluster **global_bit_membership;
        tsl::robin_set<Cluster *> merge_list;
        std::vector<int> cluster_decoding;
        std::vector<int> matrix_to_cluster_bit_map;
        tsl::robin_map<int, int> cluster_to_matrix_bit_map;
        std::vector<int> matrix_to_cluster_check_map;
        tsl::robin_map<int, int> cluster_to_matrix_check_map;

        Cluster() = default;

        Cluster(ldpc::bp::BpSparse &parity_check_matrix, int syndrome_index, Cluster **ccm, Cluster **bcm,
                tsl::robin_set<int> &planar_code_boundary_bits = EMPTY_INT_ROBIN_SET) :
                pcm(parity_check_matrix), planar_code_boundary_bits(planar_code_boundary_bits) {
            this->active = true;
            this->valid = false;
            this->cluster_id = syndrome_index;
            this->boundary_check_nodes.insert(syndrome_index);
            this->check_nodes.insert(syndrome_index);
            this->enclosed_syndromes.insert(syndrome_index);
            this->global_check_membership = ccm;
            this->global_bit_membership = bcm;
            this->global_check_membership[syndrome_index] = this;
            this->contains_boundary_bits = false;
            this->spanning_tree_boundary_bit = -1;
        }

        ~Cluster() {
            this->bit_nodes.clear();
            this->check_nodes.clear();
            this->boundary_check_nodes.clear();
            this->candidate_bit_nodes.clear();
            this->enclosed_syndromes.clear();
            this->merge_list.clear();
        }

        int parity() {
            return this->enclosed_syndromes.size() % 2;
        }

        void get_candidate_bit_nodes() {
            std::vector<int> erase_boundary_check;
            this->candidate_bit_nodes.clear();
            for (int check_index: boundary_check_nodes) {
                bool erase = true;
                for (auto &e: this->pcm.iterate_row(check_index)) {
                    if (this->global_bit_membership[e.col_index] != this) {
                        candidate_bit_nodes.push_back(e.col_index);
                        erase = false;
                    }
                }
                if (erase) {
                    erase_boundary_check.push_back(check_index);
                }
            }
            for (int check_index: erase_boundary_check) {
                this->boundary_check_nodes.erase(check_index);
            }
        }

        int add_bit_node_to_cluster(int bit_index) {
            auto bit_membership = this->global_bit_membership[bit_index];
            if (bit_membership == this) return 0; //if the bit is already in the cluster terminate.
            else if (bit_membership == NULL) {
                //if the bit has not yet been assigned to a cluster we add it.
                this->bit_nodes.insert(bit_index);
                this->global_bit_membership[bit_index] = this;
            } else {
                //if the bit already exists in a cluster, we mark down that this cluster should be
                //merged with the exisiting cluster.
                this->merge_list.insert(bit_membership);
                this->global_bit_membership[bit_index] = this;
            }
            if (this->planar_code_boundary_bits.contains(bit_index)) {
                this->contains_boundary_bits = true;
            }
            for (auto &e: this->pcm.iterate_column(bit_index)) {
                int check_index = e.row_index;
                auto check_membership = this->global_check_membership[check_index];
                if (check_membership == this) continue;
                else if (check_membership == NULL) {
                    this->check_nodes.insert(check_index);
                    this->boundary_check_nodes.insert(check_index);
                    this->global_check_membership[check_index] = this;
                } else {
                    this->check_nodes.insert(check_index);
                    this->boundary_check_nodes.insert(check_index);
                    this->merge_list.insert(check_membership);
                    this->global_check_membership[check_index] = this;
                }
            }
            return 1;
        }

        void merge_with_cluster(Cluster *cl2) {
            for (auto bit_index: cl2->bit_nodes) {
                this->bit_nodes.insert(bit_index);
                this->global_bit_membership[bit_index] = this;
            }
            for (auto check_index: cl2->check_nodes) {
                this->check_nodes.insert(check_index);
                this->global_check_membership[check_index] = this;
            }
            for (auto check_index: cl2->boundary_check_nodes) {
                this->boundary_check_nodes.insert(check_index);
            }
            if (cl2->contains_boundary_bits) {
                this->contains_boundary_bits = true;
            }
            cl2->active = false;
            for (auto j: cl2->enclosed_syndromes) {
                this->enclosed_syndromes.insert(j);
            }
        }

        int grow_cluster(const std::vector<double> &bit_weights = EMPTY_DOUBLE_VECTOR, int bits_per_step = 0) {
            if (!this->active) {
                return 0;
            }
            this->get_candidate_bit_nodes();
            this->merge_list.clear();

            if (bit_weights == EMPTY_DOUBLE_VECTOR) {
                for (int bit_index: this->candidate_bit_nodes) {
                    this->add_bit_node_to_cluster(bit_index);
                }
            } else {
                std::vector<double> cluster_bit_weights;
                for (auto bit: this->candidate_bit_nodes) {
                    cluster_bit_weights.push_back(bit_weights[bit]);
                }
                auto sorted_indices = sort_indices(cluster_bit_weights);
                int count = 0;
                for (auto i: sorted_indices) {
                    if (count == bits_per_step) break;
                    int bit_index = this->candidate_bit_nodes[i];
                    this->add_bit_node_to_cluster(bit_index);
                    count++;
                }
            }
            for (auto cl: merge_list) {
                this->merge_with_cluster(cl);
                cl->active = false;
            }
            return 1;
        }

        int find_spanning_tree_parent(const int check_index) {
            int parent = this->spanning_tree_check_roots[check_index];
            if (parent != check_index) {
                return find_spanning_tree_parent(parent);
            } else {
                return parent;
            }
        }

        void find_spanning_tree() {
            this->spanning_tree_bits.clear();
            this->spanning_tree_check_roots.clear();
            this->spanning_tree_leaf_nodes.clear();
            for (int bit_index: this->bit_nodes) {
                this->spanning_tree_bits.insert(bit_index);
            }
            // add the virtual boundary check
            if (this->contains_boundary_bits) {
                this->check_nodes.insert(-1);
            }
            for (int check_index: this->check_nodes) {
                this->spanning_tree_check_roots[check_index] = check_index;
            }
            int check_neighbours[2];

            for (int bit_index: this->bit_nodes) {
                check_neighbours[0] = this->pcm.column_heads[bit_index]->up->row_index;
                check_neighbours[1] = this->pcm.column_heads[bit_index]->down->row_index;
                if (check_neighbours[0] == check_neighbours[1]) {
                    check_neighbours[1] = -1; //set the first check neighbour to the boundary check.
                    this->spanning_tree_boundary_bit = bit_index;
                }
                int root0 = this->find_spanning_tree_parent(check_neighbours[0]);
                int root1 = this->find_spanning_tree_parent(check_neighbours[1]);
                if (root0 != root1) {
                    this->spanning_tree_check_roots[root1] = root0;
                } else {
                    this->spanning_tree_bits.erase(bit_index);
                }
            }
            for (int check_index: this->check_nodes) {
                if (check_index == -1) {
                    this->spanning_tree_leaf_nodes.insert(check_index);
                } else {
                    int spanning_tree_connectivity = 0;
                    for (auto &e: this->pcm.iterate_row(check_index)) {
                        if (this->spanning_tree_bits.contains(e.col_index)) {
                            spanning_tree_connectivity += 1;
                        }
                    }
                    if (spanning_tree_connectivity == 1) {
                        this->spanning_tree_leaf_nodes.insert(check_index);
                    }
                }
            }
        }

        std::vector<int> peel_decode(const std::vector<uint8_t> &syndrome) {
            std::vector<int> erasure;
            tsl::robin_set<int> synds;
            for (auto check_index: check_nodes) {
                if (syndrome[check_index] == 1) synds.insert(check_index);
            }
            if (this->contains_boundary_bits == true && this->parity() == 1) {
                synds.insert(-1);
            }
            this->find_spanning_tree();

            while (!synds.empty()) {
                int leaf_node_index = *(this->spanning_tree_leaf_nodes.begin());
                int bit_index;
                int check2 = -1; //we assume it is a boundary node at first.

                if (leaf_node_index == -1) {
                    bit_index = this->spanning_tree_boundary_bit;
                } else {
                    for (auto &e: this->pcm.iterate_row(leaf_node_index)) {
                        bit_index = e.col_index;
                        if (this->spanning_tree_bits.contains(bit_index)) break;
                    }
                }
                for (auto &e: this->pcm.iterate_column(bit_index)) {
                    if (e.row_index != leaf_node_index) {
                        check2 = e.row_index;
                    }
                }

                if (synds.contains(leaf_node_index)) {
                    this->spanning_tree_leaf_nodes.erase(leaf_node_index);
                    // this->spanning_tree_leaf_nodes.insert(check2);
                    erasure.push_back(bit_index);
                    this->spanning_tree_bits.erase(bit_index);
                    if (synds.contains(check2)) synds.erase(check2);
                    else synds.insert(check2);
                    synds.erase(leaf_node_index);
                } else {
                    this->spanning_tree_leaf_nodes.erase(leaf_node_index);
                    // this->spanning_tree_leaf_nodes.insert(check2);
                    this->spanning_tree_bits.erase(bit_index);
                }
                //check whether new check node is a leaf
                if (check2 == -1) {
                    this->spanning_tree_leaf_nodes.insert(check2);
                } else {
                    int spanning_tree_connectivity = 0;
                    for (auto &e: this->pcm.iterate_row(check2)) {
                        if (this->spanning_tree_bits.contains(e.col_index)) {
                            spanning_tree_connectivity += 1;
                        }
                    }
                    if (spanning_tree_connectivity == 1) {
                        this->spanning_tree_leaf_nodes.insert(check2);
                    }
                }
            }
            return erasure;
        }

        ldpc::bp::BpSparse convert_to_matrix(const std::vector<double> &bit_weights = EMPTY_DOUBLE_VECTOR) {
            this->matrix_to_cluster_bit_map.clear();
            this->matrix_to_cluster_check_map.clear();
            this->cluster_to_matrix_bit_map.clear();
            this->cluster_to_matrix_check_map.clear();

            if (bit_weights != EMPTY_DOUBLE_VECTOR) {
                std::vector<double> cluster_bit_weights;
                std::vector<int> bit_nodes_temp;
                for (int bit: this->bit_nodes) {
                    cluster_bit_weights.push_back(bit_weights[bit]);
                    bit_nodes_temp.push_back(bit);
                }
                auto sorted_indices = sort_indices(cluster_bit_weights);
                int count = 0;
                for (auto i: sorted_indices) {
                    int bit_index = bit_nodes_temp[i];
                    this->matrix_to_cluster_bit_map.push_back(bit_index);
                    this->cluster_to_matrix_bit_map[bit_index] = count;
                    count++;
                }
            } else {
                int count = 0;
                for (auto bit_index: this->bit_nodes) {
                    this->matrix_to_cluster_bit_map.push_back(bit_index);
                    this->cluster_to_matrix_bit_map[bit_index] = count;
                    count++;
                }
            }
            int count = 0;

            for (auto check_index: this->check_nodes) {
                this->matrix_to_cluster_check_map.push_back(check_index);
                this->cluster_to_matrix_check_map[check_index] = count;
                count++;
            }
            auto cluster_pcm = ldpc::bp::BpSparse(this->check_nodes.size(), this->bit_nodes.size());

            for (int check_index: this->check_nodes) {
                for (auto &e: this->pcm.iterate_row(check_index)) {
                    int bit_index = e.col_index;
                    if (this->bit_nodes.contains(bit_index)) {
                        int matrix_bit_index = cluster_to_matrix_bit_map[bit_index];
                        int matrix_check_index = cluster_to_matrix_check_map[check_index];
                        cluster_pcm.insert_entry(matrix_check_index, matrix_bit_index);
                    }
                }
            }
            return cluster_pcm;
        }

        std::vector<int> invert_decode(const std::vector<uint8_t> &syndrome, const std::vector<double> &bit_weights) {
            auto cluster_pcm = this->convert_to_matrix(bit_weights);
            std::vector<uint8_t> cluster_syndrome;

            for (auto check_index: check_nodes) {
                cluster_syndrome.push_back(syndrome[check_index]);
            }
            auto rr = ldpc::gf2sparse_linalg::RowReduce(cluster_pcm);
            auto cluster_solution = rr.fast_solve(cluster_syndrome);
            auto candidate_cluster_syndrome = cluster_pcm.mulvec(cluster_solution);
            bool equal = true;

            for (auto i = 0; i < cluster_syndrome.size(); i++) {
                if (cluster_syndrome[i] != candidate_cluster_syndrome[i]) {
                    equal = false;
                    break;
                }
            }

            this->cluster_decoding.clear();
            this->valid = equal;
            for (auto i = 0; i < cluster_solution.size(); i++) {
                if (cluster_solution[i] == 1) {
                    this->cluster_decoding.push_back(this->matrix_to_cluster_bit_map[i]);
                }
            }
            return this->cluster_decoding;
        }

        void print();
    };


    class UfDecoder {

    private:
        bool weighted;
        ldpc::bp::BpSparse &pcm;

    public:
        std::vector<uint8_t> decoding;
        int bit_count;
        int check_count;
        tsl::robin_set<int> planar_code_boundary_bits;
        bool pcm_max_bit_degree_2;

        UfDecoder(ldpc::bp::BpSparse &parity_check_matrix) : pcm(parity_check_matrix) {
            this->bit_count = pcm.n;
            this->check_count = pcm.m;
            this->decoding.resize(this->bit_count);
            this->weighted = false;

            this->pcm_max_bit_degree_2 = true;
            for (auto i = 0; i < this->pcm.n; i++) {
                int col_weight = this->pcm.get_col_degree(i);
                if (col_weight > 2) {
                    this->pcm_max_bit_degree_2 = false;
                } else if (col_weight == 1) {
                    this->planar_code_boundary_bits.insert(i);
                } else if (col_weight == 0) {
                    throw (std::runtime_error("Invalid parity check matrix. Column weight is zero."));
                }
            }
        }

        std::vector<uint8_t> &
        peel_decode(const std::vector<uint8_t> &syndrome, const std::vector<double> &bit_weights = EMPTY_DOUBLE_VECTOR,
                    int bits_per_step = 1) {

            if (!this->pcm_max_bit_degree_2) {
                throw (std::runtime_error(
                        "Peel decoder only works for planar codes. Use the matrix_decode method for more general codes."));
            }

            fill(this->decoding.begin(), this->decoding.end(), 0);
            std::vector<Cluster *> clusters;
            std::vector<Cluster *> invalid_clusters;
            auto **global_bit_membership = new Cluster *[pcm.n]();
            auto **global_check_membership = new Cluster *[pcm.m]();

            for (int i = 0; i < this->pcm.m; i++) {
                if (syndrome[i] == 1) {
                    auto *cl = new Cluster(this->pcm, i, global_check_membership, global_bit_membership,
                                           this->planar_code_boundary_bits);
                    clusters.push_back(cl);
                    invalid_clusters.push_back(cl);
                }
            }

            while (!invalid_clusters.empty()) {
                for (auto cl: invalid_clusters) {
                    if (cl->active) {
                        cl->grow_cluster(bit_weights, bits_per_step);
                    }
                }

                invalid_clusters.clear();
                for (auto cl: clusters) {
                    if (cl->active && cl->parity() == 1 && !cl->contains_boundary_bits) {
                        invalid_clusters.push_back(cl);
                    }
                }
                std::sort(invalid_clusters.begin(), invalid_clusters.end(), [](const Cluster *lhs, const Cluster *rhs) {
                    return lhs->bit_nodes.size() < rhs->bit_nodes.size();
                });
            }
            for (auto cl: clusters) {
                if (cl->active) {
                    auto erasure = cl->peel_decode(syndrome);
                    for (int bit: erasure) this->decoding[bit] = 1;
                }
                delete cl;
            }
            delete[] global_bit_membership;
            delete[] global_check_membership;
            return this->decoding;
        }


        std::vector<uint8_t> &
        matrix_decode(const std::vector<uint8_t> &syndrome,
                      const std::vector<double> &bit_weights = EMPTY_DOUBLE_VECTOR,
                      int bits_per_step = 1) {
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
                        cl->grow_cluster(bit_weights, bits_per_step);
                        auto cluster_decoding = cl->invert_decode(syndrome, bit_weights);
                    }
                }

                invalid_clusters.clear();
                for (auto cl: clusters) {
                    if (cl->active && !cl->valid) {
                        invalid_clusters.push_back(cl);
                    }
                }
                std::sort(invalid_clusters.begin(), invalid_clusters.end(), [](const Cluster *lhs, const Cluster *rhs) {
                    return lhs->bit_nodes.size() < rhs->bit_nodes.size();
                });
            }

            for (auto cl: clusters) {
                if (cl->active) {
                    for (int bit: cl->cluster_decoding) this->decoding[bit] = 1;
                }
                delete cl;
            }
            delete[] global_bit_membership;
            delete[] global_check_membership;
            return this->decoding;
        }
    };

    void Cluster::print() {
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
        std::cout << "Spanning tree bits: ";
        for (auto i: this->spanning_tree_bits) std::cout << i << " ";
        std::cout << std::endl;
        std::cout << "Spanning tree leaf nodes: ";
        for (auto i: this->spanning_tree_leaf_nodes) std::cout << i << " ";
        std::cout << std::endl;
        std::cout << "Contains boundary bits: " << this->contains_boundary_bits << std::endl;
        std::cout << "Boundary bits: ";
        for (auto i: this->planar_code_boundary_bits) std::cout << i << " ";
        std::cout << std::endl;
        std::cout << "........." << std::endl;
    }
}//end namespace uf

#endif