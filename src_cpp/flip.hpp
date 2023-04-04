#ifndef FLIP_H
#define FLIP_H

#include <vector>
#include <memory>
#include <iterator>
#include <cmath>
#include "bp.hpp"
#include "sparse_matrix_util.hpp"

class FlipDecoder
{
public:
    vector<uint8_t> decoding;
    shared_ptr<bp::BpSparse> pcm;
    int bit_count;
    int check_count;
    int converge;
    int iterations;
    int max_iter;
    vector<uint8_t> syndrome;

    FlipDecoder(shared_ptr<bp::BpSparse> pcm, int max_iter)
    {
        this->pcm;
        this->max_iter = max_iter;
        this->check_count = this->pcm->m;
        this->bit_count = this->pcm->n;
        this->decoding.resize(this->pcm->n);
    }

    ~FlipDecoder()
    {
        this->decoding.clear();
    }

    vector<uint8_t> &decode(vector<uint8_t> &synd)
    {

        this->syndrome = synd;

        int syndrome_hamming_weight = 0;
        for (auto bit : this->syndrome)
            syndrome_hamming_weight += bit;

        for(int iter = 1; iter<this->max_iter+1; iter++){

            for (int bit_idx = 0; bit_idx < this->bit_count; bit_idx++)
            {

                vector<int> unsatisfied_checks;
                vector<int> satisfied_checks;

                for (auto e : this->pcm->iterate_column(bit_idx))
                {
                    int check_idx = e->row_index;
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

    }
};

#endif FLIP_H