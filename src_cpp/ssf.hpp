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
#include "rng.hpp"

namespace ssf{


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

            for (int check_idx = 0; check_idx < this->bit_count; check_idx++)
            {

               if(check_idx ==1){
            
                    map<int,uint8_t> local_bits;
                    for(auto e: this->pcm->iterate_row(check_idx)){
                        local_bits[e->col_index] = this->decoding[e->col_index];
                    }

                    int initial_local_syndrome_weight = 0;
                    map<int,uint8_t> local_checks;
                    for(int bit: bit_neighbourhood){
                        for(auto e: this->pcm->iterate_column(bit)){
                            if(syndrome[e->row_index] == 1) initial_local_syndrome_weight++;
                            local_checks[e->row_index] = syndrome[e->row_index];
                        }
                    }


                    int max_syndrome_reduction = std::numeric_limits<int>::min();




               }

            }
        }

        this->converge = 0;
        this->iterations = max_iter;
        return this->decoding;

    } //end of SmallSetFlipDecoder::decode method

}; //end of class SmallSetFlipDecoder

}//end of flip namespace

#endif