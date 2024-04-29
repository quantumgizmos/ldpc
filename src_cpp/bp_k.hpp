#ifndef BPK_H
#define BPK_H

#include <vector>
#include <memory>
#include <iterator>
#include <cmath> 
#include <limits>
#include <set>
#include <numeric>

#include "bp.hpp"
#include "sort.hpp"
#include "util.hpp"
#include "gf2sparse.hpp"
#include "gf2sparse_linalg.hpp"

namespace ldpc::bpk{

    int find_spanning_tree_parent(const int check_index, std::vector<int>& spanning_tree_check_roots) {
        int parent = spanning_tree_check_roots[check_index];
        if (parent != check_index) {
            return find_spanning_tree_parent(parent, spanning_tree_check_roots);
        } else {
            return parent;
        }
    }

    struct SpanningTreeBits{
        std::vector<int> spanning_tree_bits;
        std::vector<int> not_spanning_tree_bits;
    };

    SpanningTreeBits find_weighted_spanning_tree(ldpc::bp::BpSparse& pcm, std::vector<int>& bit_order){

        int check_count = pcm.m;
        std::vector<int> spanning_tree_check_roots(check_count, 0);
        for(int i = 0; i < check_count; i++) spanning_tree_check_roots[i] = i;
        
        SpanningTreeBits spanning_tree;

        for(int bit: bit_order){
            std::set<int> check_roots;
            bool loop_found = false;
            bool first_check = true;
            int first_root = 0;
            for(auto& e: pcm.iterate_column(bit)){
                int check_index = e.row_index;
                int check_root = find_spanning_tree_parent(check_index, spanning_tree_check_roots);
                if(first_check){
                    first_root = check_root;
                    first_check = false;
                }

                if(check_roots.contains(check_root)){
                    loop_found = true;
                }
                check_roots.insert(check_root);

                spanning_tree_check_roots[check_root] = first_root;

            }

            if(loop_found){
                spanning_tree.not_spanning_tree_bits.push_back(bit);
            }
            else{
                spanning_tree.spanning_tree_bits.push_back(bit);
            }

        }

        return spanning_tree;

    }

    std::vector<uint8_t>& bp_k_decode(ldpc::bp::BpDecoder& bpd, std::vector<uint8_t>& syndrome){

        std::vector<int> bit_order(bpd.pcm.n);
        for(int i=0; i<bpd.pcm.n; i++) bit_order[i] = i;

        ldpc::sort::soft_decision_col_sort(bpd.log_prob_ratios, bit_order,bpd.pcm.n);

        auto pcm = bpd.pcm;
        auto stb = find_weighted_spanning_tree(bpd.pcm, bit_order);
        auto channel_probabilites_backup = bpd.channel_probabilities;
        auto max_iter_backup = bpd.maximum_iterations;

        for(int bit: stb.not_spanning_tree_bits){
            bpd.channel_probabilities[bit] = 0;
        }

        // ldpc::sparse_matrix_util::print_vector(stb.spanning_tree_bits);
        // ldpc::sparse_matrix_util::print_vector(stb.not_spanning_tree_bits);
        
        bpd.maximum_iterations = bpd.pcm.n;
        auto decoding = bpd.decode(syndrome);
        bpd.channel_probabilities = channel_probabilites_backup;
        bpd.maximum_iterations = max_iter_backup;
        
        return bpd.decoding;
    
    }

std::vector<uint8_t>& bp_k_decode_ps(ldpc::bp::BpDecoder& bpd, std::vector<uint8_t>& syndrome){

        std::vector<int> bit_order(bpd.pcm.n);
        for(int i=0; i<bpd.pcm.n; i++) bit_order[i] = i;
        ldpc::sort::soft_decision_col_sort(bpd.log_prob_ratios, bit_order,bpd.pcm.n);
        auto pcm = bpd.pcm;
        auto stb = find_weighted_spanning_tree(bpd.pcm, bit_order);
        auto pcm_st = ldpc::gf2sparse::copy_cols(pcm, stb.spanning_tree_bits);

        // std::cout<<"Copy cols and spanning tree done"<<std::endl;

        auto bpd_st_channel = std::vector<double>(pcm_st.n, 0);
        for(int i = 0; i < pcm_st.n; i++){
            bpd_st_channel[i] = bpd.channel_probabilities[stb.spanning_tree_bits[i]];
        }

        // std::cout<<"Channel probabilities copied"<<std::endl;

        ldpc::bp::BpDecoder bpd_st(pcm_st, bpd_st_channel, pcm_st.n, ldpc::bp::PRODUCT_SUM, ldpc::bp::SERIAL, 0.625,1,ldpc::bp::NULL_INT_VECTOR,3,true);

        // std::cout<<"Bpd_st initialized"<<std::endl;

        auto decoding_st = bpd_st.decode(syndrome);

        // std::cout<<"Decoding done"<<std::endl;

        auto decoding = std::vector<uint8_t>(pcm.n, 0);
        for(int i = 0; i < pcm_st.n; i++){
            if(decoding_st[i] == 1){
                decoding[stb.spanning_tree_bits[i]] = 1;
            }
        }

        if(bpd_st.converge){
            bpd.converge = true;
        }
        else{
            bpd.converge = false;
        }

        bpd.decoding = decoding;
        
        return decoding;
    
    }

}//end bpk namespace


#endif