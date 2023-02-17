#ifndef BP2_H
#define BP2_H

#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include "sparse_matrix_util.hpp"

using namespace std;

class gf2csr{
    public:
        vector<int> row_indices;
        vector<int> col_indices;
        vector<int> col_index_map;
        vector<int> row_widths;
        vector<int> col_heights;
        int max_row_weight;
        int max_col_weight;
        int row_count;
        int col_count;

        gf2csr(int rows, int cols, int max_row_weight, int max_col_weight){
            this->row_count = rows;
            this->col_count = cols;
            this->max_row_weight = max_row_weight;
            this->max_col_weight = max_col_weight;
            this->row_indices.resize(this->row_count*this->max_row_weight,-1);
            this->col_indices.resize(this->col_count*this->max_col_weight,-1);
            this->col_index_map.resize(this->col_indices.size(),-1);
            this->row_widths.resize(this->row_count,0);
            this->col_heights.resize(this->col_count,0);
        }

        int get_row_index(int i){
            return this->max_row_weight*i;
        }

        int get_col_index(int j){
            return this->max_col_weight*j;
        }

        void insert_entry(int i, int j){
            int row_index = this->get_row_index(i);
            int col_index = this->get_col_index(j);
            this->row_indices[row_index+this->row_widths[i]]=j;
            this->col_indices[col_index+this->col_heights[j]]=i;
            this->col_index_map[col_index+this->col_heights[j]]=row_index+this->row_widths[i];
            this->row_widths[i] += 1;
            this->col_heights[j] += 1;
        }

        vector<uint8_t>& mulvec(vector<uint8_t>& input_vector, vector<uint8_t>& output_vector){
            for(auto& i: output_vector) i = 0;
            for(auto check = 0; check<this->row_count; check++){
                int total = 0;
                auto ci = this->get_row_index(check);
                auto width = this->row_widths[check];
                for(auto i=ci; i<(ci+width); i++){
                    auto bit_index = this->row_indices[i];
                    if(input_vector[bit_index] == 1){
                        total^=1;
                    }
                }
                output_vector[check] = total;
            }

            return output_vector;

        }

        bool entry_exists(int i, int j){
            int row_index = this->get_row_index(i);
            for(int k = 0; k<this->max_row_weight; k++){
                if(this->row_indices[row_index+k]==j) return true;
            }
            return false;
        }

        void print(){

            for(auto row = 0; row<this->row_count; row++){
                for(auto col = 0; col<this->col_count; col++){
                    if(this->entry_exists(row,col)) cout<<1;
                    else cout<<0;
                }
                cout<<endl;
            }
        }
};

class bpcsr_decoder{

    public:
        gf2csr* pcm;
        int check_count;
        int bit_count;
        int max_iter;
        double alpha;
        double error_rate;
        int iterations;
        vector<double> log_prob_ratios;
        vector<uint8_t> decoding;
        vector<uint8_t> candidate_syndrome;
        int converge;
        vector<double> bit_to_check;
        vector<double> check_to_bit;

  
        bpcsr_decoder(gf2csr* pcm, double error_rate, int max_iter, double alpha){
            this->pcm = pcm;
            this->error_rate = error_rate;
            this->alpha = alpha;
            this->max_iter = max_iter;
            this->check_count = this->pcm->row_count;
            this->bit_count = this->pcm->col_count;
            this->decoding.resize(bit_count);
            this->candidate_syndrome.resize(check_count);
            this->log_prob_ratios.resize(bit_count);
            this->check_to_bit.resize(pcm->row_indices.size());
            this->bit_to_check.resize(pcm->col_indices.size());

        }

        vector<uint8_t>& decode(vector<uint8_t>& syndrome){

            for(auto& llr: this->bit_to_check){
                llr = log((1-error_rate)/error_rate);
            }

            // print_vector(this->bit_to_check);

            for(int iter = 1; iter<=this->max_iter; iter++){

                // cout<<"Iter "<<iter<<endl;

                for(int check = 0; check<check_count; check++){

                    int ci = this->pcm->get_row_index(check);
                    int width = this->pcm->row_widths[check];
                    double temp = numeric_limits<double>::max();
                    int total_sgn, sgn;
                    total_sgn = syndrome[check];

                    // cout<<ci<<", "<<width<<endl;

                    for(int i = ci; i<(ci+width); i++){
                        if(this->bit_to_check[i]<=0) total_sgn+=1;
                        this->check_to_bit[i] = temp;
                        if(abs(this->bit_to_check[i]) < temp){
                            temp = abs(this->bit_to_check[i]);
                        }
                    }

                    temp = numeric_limits<double>::max();
                    for(int i = ci+width-1; i>=ci; i--){
                        sgn = total_sgn;
                        if(temp<this->check_to_bit[i]){
                            this->check_to_bit[i] = temp;
                        }
                        if(this->bit_to_check[i]<=0) sgn+=1;
                        this->check_to_bit[i]*=pow(-1.0,sgn)*alpha;

                        // cout<<"C2B: "<<check<<"->"<<this->pcm->row_indices[i]<<": "<< this->check_to_bit[i] <<endl;

                        if(abs(this->bit_to_check[i])<temp){
                            temp = abs(this->bit_to_check[i]);
                        }

                    } 

                }

                // cout<<"check to bit complete"<<endl;

                for(int bit = 0; bit<this->bit_count; bit++){
                    int ci = this->pcm->get_col_index(bit);
                    int height = this->pcm->col_heights[bit];

                    // cout<<ci<<", "<<height<<endl;

                    double temp = log((1-error_rate)/error_rate);
                    for(auto i = ci; i<ci+height; i++){
                        auto j = this->pcm->col_index_map[i];
                        this->bit_to_check[j] = temp;
                        temp += this->check_to_bit[j];
                    }

                    this->log_prob_ratios[bit] = temp;
                    if(temp<=0) this->decoding[bit] = 1;
                    else this->decoding[bit] = 0;
                }

                // print_vector(log_prob_ratios);
                this->pcm->mulvec(this->decoding,this->candidate_syndrome);
                this->converge = true;
                for(int i = 0; i<this->check_count; i++){
                    if(this->candidate_syndrome[i]!=syndrome[i]){
                        this->converge = false;
                        break;
                    }
                }
                if(this->converge){
                    this->iterations = iter;
                    return this->decoding;
                }


                for(int bit = 0; bit<this->bit_count; bit++){
                    int ci = this->pcm->get_col_index(bit);
                    int height = this->pcm->col_heights[bit];
                    double temp = 0;
                    for(auto i = ci+height-1; i>=ci; i--){
                        auto j = this->pcm->col_index_map[i];
                        this->bit_to_check[j] += temp;
                        temp += this->check_to_bit[j];
                        // cout<<"B2C: "<<bit<<"->"<<this->pcm->col_indices[i]<<": "<< this->bit_to_check[j] <<endl;

                    }
                }

            this->iterations = iter;

            }

            // print_vector(syndrome);
            // print_vector(candidate_syndrome);


            return this->decoding;


        }

};

#endif