#ifndef BPK_H
#define BPK_H

#include <vector>
#include <memory>
#include <iterator>
#include <cmath> 
#include <limits>
#include <set>

#include "bp.hpp"
#include "sort.hpp"
#include "util.hpp"
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

    std::vector<int> find_weighted_spanning_tree(ldpc::bp::BpSparse& pcm, std::vector<int>& bit_order){

        int check_count = pcm.m;
        std::vector<int> spanning_tree_check_roots(check_count, 0);
        for(int i = 0; i < check_count; i++) spanning_tree_check_roots[i] = i;
        
        std::vector<int> spanning_tree_bits;

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

            if(!loop_found){
                spanning_tree_bits.push_back(bit);
            }

        }

        return spanning_tree_bits;

    }

}//end bpk namespace


#endif