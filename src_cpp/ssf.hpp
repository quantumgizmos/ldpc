#ifndef FLIP_H
#define FLIP_H

#include <vector>
#include <memory>
#include <iterator>
#include <cmath>
#include "bp.hpp"
#include <random>
#include <chrono>
#include <map>
#include <set>
#include "rng.hpp"
#include "util.hpp"

namespace ssf{

int hamming_weight(vector<uint8_t> v){
    int count = 0;
    for(auto e: v){
        if(e==1) count++;
    }
    return count;
}

class SsfCluster
{
public:
    
    ~SsfCluster(){};

    vector<int> bit_node_pcm_mapping;
    vector<int> inv_bit_node_pcm_mapping;
    vector<int> check_node_pcm_mapping;
    vector<int> inv_check_node_pcm_mapping;
    map<int,vector<uint8_t>> lookup_table;
    
        
    SsfCluster(shared_ptr<bp::BpSparse> pcm, int check_node){

        set<int> bit_nodes;
        set<int> check_nodes;
        check_nodes.insert(check_node);

        for(auto e: pcm->iterate_row(check_node)){
            bit_nodes.insert(e->col_index);
        }

        for(int bit_node: bit_nodes){
            for(auto e: pcm->iterate_column(bit_node)){
                check_nodes.insert(e->row_index);
            }
        }

        int count = 0;
        for(int bit: bit_nodes){
            this->bit_node_pcm_mapping.push_back(bit);
            this->inv_bit_node_pcm_mapping[bit] = count;
            count++;
        }

        for(int check: check_nodes){
            this->check_node_pcm_mapping.push_back(check);
            this->inv_check_node_pcm_mapping[check] = count;
            count++;
        }

        auto cluster = new GF2Sparse<>(this->bit_node_pcm_mapping.size(),this->check_node_pcm_mapping.size());

        for(int check: this->check_node_pcm_mapping){
            for(auto e: pcm->iterate_row(check)){
                cluster->insert_entry(this->inv_check_node_pcm_mapping[e->row_index], this->inv_bit_node_pcm_mapping[e->col_index]);
            }
        }

        //contruct the local lookup table for this check node

        for


        delete cluster;

    }


    

};

class SmallSetFlipDecoder
{

private:
    vector<uint8_t> local_syndrome(vector<int> &bit_neighbourhood)
    {
        vector<uint8_t> local_syndrome;
        local_syndrome.resize(bit_neighbourhood.size());
        for(int i=0; i<bit_neighbourhood.size(); i++){
            local_syndrome[i] = this->syndrome[bit_neighbourhood[i]];
        }
        return local_syndrome;
    }


public:

    shared_ptr<bp::BpSparse> pcm;
    int bit_count;
    int check_count;
    int converge;
    int iterations;
    int max_iter;

    // vector<shared_ptr<bp::BpSparse>> stab_clusters;

    vector<uint8_t> syndrome;
    vector<uint8_t> decoding;

    SmallSetFlipDecoder(shared_ptr<bp::BpSparse> pcm, int max_iter=0)
    {
        this->pcm = pcm;
        this->max_iter = max_iter;
        if(this->max_iter == 0) this->max_iter = this->pcm->n;
        this->check_count = this->pcm->m;
        this->bit_count = this->pcm->n;
        this->decoding.resize(this->pcm->n);





    }

    ~SmallSetFlipDecoder()
    {
        this->decoding.clear();
    }

    vector<uint8_t> &decode(vector<uint8_t> &synd)
    {

        std::fill(this->decoding.begin(), this->decoding.end(), 0);
        this->syndrome = synd;

        int syndrome_hamming_weight = 0;
        for (auto bit : this->syndrome)
            syndrome_hamming_weight += bit;

        for(int iter = 1; iter<=this->max_iter; iter++){

            // for (int check_idx = 0; check_idx < this->bit_count; check_idx++)
            // {

            //    if(check_idx ==1){
            
            //         map<int,uint8_t> local_bits;
            //         for(auto e: this->pcm->iterate_row(check_idx)){
            //             bit_neighbourhood.push_back(e->col_index);
            //         }

            //         int initial_local_syndrome_weight = 0;
            //         map<int,uint8_t> local_checks;
            //         for(int bit: bit_neighbourhood){
            //             for(auto e: this->pcm->iterate_column(bit)){
            //                 if(syndrome[e->row_index] == 1) initial_local_syndrome_weight++;
            //                 local_checks[e->row_index] = syndrome[e->row_index];
            //             }
            //         }


            //         int max_syndrome_reduction = std::numeric_limits<int>::min();




            //    }

            // }
        }

        this->converge = 0;
        this->iterations = max_iter;
        return this->decoding;

    } //end of SmallSetFlipDecoder::decode method

}; //end of class SmallSetFlipDecoder

}//end of flip namespace

#endif