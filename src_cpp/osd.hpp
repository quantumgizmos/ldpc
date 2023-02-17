#ifndef BPOSD_H
#define BPOSD_H

#include <vector>
#include <memory>
#include <iterator>
#include <cmath> 
#include <limits>
#include "bp.hpp"
#include "sort.hpp"

using namespace std;

vector<uint8_t> decimal_to_binary_reverse(int n,int k)
{
   vector<uint8_t> binary_number;
   int divisor;
   int remainder;
   divisor=n;

   binary_number.resize(k);

   for(int i=0; i<k;i++)
   {
        remainder=divisor%2;
        binary_number[i]=remainder;
        divisor=divisor/2;
        if(divisor==0) break;
   }

   return  binary_number;
}


class osd_decoder{
    public:
        int osd_method;
        int osd_order;
        int k, bit_count, check_count;
        bp_sparse* pcm;
        vector<uint8_t> osd0_decoding;
        vector<uint8_t> osdw_decoding;
        vector<uint8_t> bp_decoding;
        vector<vector<uint8_t>> osd_candidate_strings;
        vector<double> channel_probs;
        
        osd_decoder(bp_sparse* parity_check_matrix, int osd_method, int osd_order, vector<double> channel_probabilities){

            this->pcm = parity_check_matrix;
            this->bit_count = pcm->n;
            this->check_count = pcm->m;
            this->channel_probs = channel_probabilities;


            if(osd_method!=-1){

                this->osdw_decoding.resize(pcm->n);
                this->osd0_decoding.resize(pcm->n);
                int osd_candidate_string_count;
                this->pcm->lu_decomposition(); 
                this->osd_order = osd_order;
                this->osd_method = osd_method;
                this->k = pcm->n - pcm->rank;


                if(osd_method==0 && osd_order!=0){
                    osd_candidate_string_count = pow(2,osd_order);
                    for(int i=1; i<osd_candidate_string_count; i++){
                        osd_candidate_strings.push_back(decimal_to_binary_reverse(i,k));
                    }
                }

                if(osd_method==1 && osd_order!=0){
                    for(int i=0; i<k; i++) {
                        vector<uint8_t> osd_candidate;
                        osd_candidate.resize(k,0);
                        osd_candidate[i]=1; 
                        osd_candidate_strings.push_back(osd_candidate);
                    }

                    for(int i = 0; i<osd_order;i++){
                        for(int j = 0; j<osd_order; j++){
                            if(j<=i) continue;
                            vector<uint8_t> osd_candidate;
                            osd_candidate.resize(k,0);
                            osd_candidate[i]=1;
                            osd_candidate[j]=1; 
                            osd_candidate_strings.push_back(osd_candidate);
                        }
                    }

                }

            }
           
        }

        ~osd_decoder(){};


        vector<uint8_t>& decode(vector<uint8_t>& syndrome, vector<double>& log_prob_ratios) {

            soft_decision_col_sort(log_prob_ratios, pcm->cols,bit_count);

            pcm->lu_decomposition(false);
            osd0_decoding=pcm->lu_solve(syndrome,osd0_decoding);

            for(int i=0;i<pcm->n;i++) osdw_decoding[i]=osd0_decoding[i];

            if(osd_order==0){
                return osd0_decoding;
            }

            double candidate_weight, osd_min_weight;

            osd_min_weight=0;
            for(int i=0; i<pcm->n; i++){
                if(osd0_decoding[i]==1){
                    osd_min_weight+=log(1/channel_probs[i]);
                }
            }

            vector<int> non_pivot_columns;
            for(int i = pcm->rank; i<pcm->n; i++){
                non_pivot_columns.push_back(pcm->cols[i]);
            }
            
            auto pcm_t = copy_cols(pcm, non_pivot_columns);

            vector<uint8_t> t_syndrome;
            t_syndrome.resize(pcm->m);
            vector<uint8_t> candidate_solution;
            candidate_solution.resize(pcm->n);


            for(auto candidate_string: osd_candidate_strings){

                pcm_t->mulvec(candidate_string,t_syndrome);
                for(int i=0;i<pcm->m;i++){
                    t_syndrome[i] ^= syndrome[i];
                }

                pcm->lu_solve(t_syndrome,candidate_solution);
                for(int i=0; i<k; i++){
                    candidate_solution[non_pivot_columns[i]]=candidate_string[i];
                }
                candidate_weight=0;
                for(int i=0; i<pcm->n; i++){
                    if(candidate_solution[i]==1){
                        candidate_weight+=log(1/channel_probs[i]);
                    }
                }
                if(candidate_weight<osd_min_weight){
                    osd_min_weight = candidate_weight;
                    for(int i=0; i<pcm->n; i++){
                        osdw_decoding[i] = candidate_solution[i];
                    }
                }

            }

            delete pcm_t;
            return osdw_decoding;

        }

};


#endif