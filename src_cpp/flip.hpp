#ifndef FLIP_H
#define FLIP_H

#include <vector>
#include <memory>
#include <iterator>
#include <cmath>
#include <random>
#include <chrono>

#include "bp.hpp"
#include "rng.hpp"


namespace ldpc { namespace flip{

class FlipDecoder
{

public:

    ldpc::bp::BpSparse& pcm;
    int bit_count;
    int check_count;
    int converge{};
    int iterations{};
    int max_iter;
    int pfreq;
    int seed;
    ldpc::rng::RandomNumberGenerator* RNG;
    std::vector<uint8_t> syndrome;
    std::vector<uint8_t> decoding;

    explicit FlipDecoder(ldpc::bp::BpSparse& pcm, int max_iter=0, int pfreq=0, int seed = 0):
        pcm(pcm)
    {
        this->max_iter = max_iter;
        if(this->max_iter == 0) { this->max_iter = this->pcm.n;
}
        this->pfreq = pfreq;
        if(this->pfreq == 0) { this->pfreq = std::numeric_limits<int>::max();
}
        this->seed = seed;
        if (this->seed == 0){
            this->RNG = new ldpc::rng::RandomNumberGenerator();
        }
        else{
            this->RNG = new ldpc::rng::RandomNumberGenerator(this->seed);
            }
        this->check_count = this->pcm.m;
        this->bit_count = this->pcm.n;
        this->decoding.resize(this->pcm.n);
    }

    ~FlipDecoder()
    {
        this->decoding.clear();
        delete this->RNG;
    }

    std::vector<uint8_t>& decode(std::vector<uint8_t> &synd)
    {

        std::fill(this->decoding.begin(), this->decoding.end(), 0);

        this->syndrome = synd;

        int syndrome_hamming_weight = 0;
        for (auto bit : this->syndrome) { {
            syndrome_hamming_weight += bit;
}
}

        for(int iter = 1; iter<=this->max_iter; iter++){

            for (int bit_idx = 0; bit_idx < this->bit_count; bit_idx++)
            {

                std::vector<int> unsatisfied_checks;
                std::vector<int> satisfied_checks;

                for (auto& e : this->pcm.iterate_column(bit_idx))
                {
                    int check_idx = e.row_index;
                    if (this->syndrome[check_idx] == 1)
                    {
                        unsatisfied_checks.push_back(check_idx);
                    }
                    else
                    {
                        satisfied_checks.push_back(check_idx);
                    }
                }

                if (satisfied_checks.size() < unsatisfied_checks.size())
                {
                    this->decoding[bit_idx] ^= 1;
                    for (auto check_idx : unsatisfied_checks)
                    {
                        this->syndrome[check_idx] ^= 1;
                        syndrome_hamming_weight -= 1;
                    }
                    for (auto check_idx : satisfied_checks)
                    {
                        this->syndrome[check_idx] ^= 1;
                        syndrome_hamming_weight += 1;
                    }
                }
                else if(iter%this->pfreq == 0 && satisfied_checks.size() == unsatisfied_checks.size()){
                    if(this->RNG->random_double()<0.5){
                        this->decoding[bit_idx] ^= 1;
                        for (auto check_idx : unsatisfied_checks)
                        {
                            this->syndrome[check_idx] ^= 1;
                            syndrome_hamming_weight -= 1;
                        }
                        for (auto check_idx : satisfied_checks)
                        {
                            this->syndrome[check_idx] ^= 1;
                            syndrome_hamming_weight += 1;
                        }
                    }
                }
                else
                {
                    continue;
                }

                if (syndrome_hamming_weight == 0)
                {
                    this->converge = 1;
                    this->iterations = iter;
                    return this->decoding;
                }

            }
        }

        this->converge = 0;
        this->iterations = max_iter;
        return this->decoding;

    } //end of FlipDecoder::decode method

}; //end of class FlipDecoder

} }  // namespace ldpc::flip

#endif