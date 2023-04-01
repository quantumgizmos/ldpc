#ifndef BPOSD_H
#define BPOSD_H

#include <vector>
#include <memory>
#include <iterator>
#include <cmath> 
#include <limits>
#include "bp.hpp"
#include "gf2sparse.hpp"
#include "gf2sparse_linalg.hpp"
#include "sort.hpp"
#include "sparse_matrix_util.hpp"

using namespace std;

namespace osd{

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


class OsdDecoder{
    public:
        int osd_method;
        int osd_order;
        int k, bit_count, check_count;
        shared_ptr<bp::BpSparse> pcm;
        vector<uint8_t> osd0_decoding;
        vector<uint8_t> osdw_decoding;
        vector<vector<uint8_t>> osd_candidate_strings;
        vector<double> channel_probs;
        vector<int> column_ordering;
        gf2sparse_linalg::RowReduce<shared_ptr<bp::BpSparse>>* LuDecomposition;
        
        OsdDecoder(shared_ptr<bp::BpSparse> parity_check_matrix, int osd_method, int osd_order, vector<double> channel_probabilities){

            this->pcm = parity_check_matrix;
            this->bit_count = this->pcm->n;
            this->check_count = this->pcm->m;
            this->channel_probs = channel_probabilities;

            this->osd_order = osd_order;
            this->osd_method = osd_method;

            this->osd_setup();


        }

        int osd_setup(){

            this->osd_candidate_strings.clear();
            
            if(this->osd_method == -1) return 0;

            this->LuDecomposition = new gf2sparse_linalg::RowReduce<shared_ptr<bp::BpSparse>>(this->pcm);
            this->column_ordering.resize(this->pcm->n);
            int osd_candidate_string_count;
            this->LuDecomposition->rref(false,true); 
            this->k = this->pcm->n - this->LuDecomposition->rank;

            if(this->osd_method==0 || this->osd_order==0){
                return 1;
            }

            if(this->osd_method==1){
                osd_candidate_string_count = pow(2,this->osd_order);
                for(int i=1; i<osd_candidate_string_count; i++){
                    this->osd_candidate_strings.push_back(decimal_to_binary_reverse(i,k));
                }
            }

            if(this->osd_method==2){
                for(int i=0; i<k; i++) {
                    vector<uint8_t> osd_candidate;
                    osd_candidate.resize(k,0);
                    osd_candidate[i]=1; 
                    this->osd_candidate_strings.push_back(osd_candidate);
                }

                for(int i = 0; i<this->osd_order;i++){
                    for(int j = 0; j<this->osd_order; j++){
                        if(j<=i) continue;
                        vector<uint8_t> osd_candidate;
                        osd_candidate.resize(k,0);
                        osd_candidate[i]=1;
                        osd_candidate[j]=1; 
                        this->osd_candidate_strings.push_back(osd_candidate);
                    }
                }

            }
            return 1;
        }

        ~OsdDecoder(){
            delete this->LuDecomposition;
        };


        vector<uint8_t>& decode(vector<uint8_t>& syndrome, vector<double>& log_prob_ratios) {

            soft_decision_col_sort(log_prob_ratios, this->column_ordering,bit_count);

            //row reduce the matrix according to the new column ordering
            this->LuDecomposition->rref(false,true,this->column_ordering);

            //find the OSD0 solution
            this->osd0_decoding = this->osdw_decoding = LuDecomposition->lu_solve(syndrome);

            if(osd_order==0){
                return this->osd0_decoding;
            }

            double candidate_weight, osd_min_weight;

            osd_min_weight=0;
            for(int i=0; i<this->pcm->n; i++){
                if(this->osd0_decoding[i]==1){
                    osd_min_weight+=log(1/this->channel_probs[i]);
                }
            }

            vector<int> non_pivot_columns;
            for(int i = this->LuDecomposition->rank; i<this->pcm->n; i++){
                non_pivot_columns.push_back(this->LuDecomposition->cols[i]);
            }
            
            auto pcm_t = gf2sparse::copy_cols(this->pcm, non_pivot_columns);

            vector<uint8_t> t_syndrome;
            t_syndrome.resize(this->pcm->m);
            vector<uint8_t> candidate_solution;
            candidate_solution.resize(this->pcm->n);


            for(auto candidate_string: this->osd_candidate_strings){

                pcm_t->mulvec(candidate_string,t_syndrome);
                for(int i=0;i<this->pcm->m;i++){
                    t_syndrome[i] ^= syndrome[i];
                }

                candidate_solution = this->LuDecomposition->lu_solve(t_syndrome);
                for(int i=0; i<k; i++){
                    candidate_solution[non_pivot_columns[i]]=candidate_string[i];
                }
                candidate_weight=0;
                for(int i=0; i<this->pcm->n; i++){
                    if(candidate_solution[i]==1){
                        candidate_weight+=log(1/this->channel_probs[i]);
                    }
                }
                if(candidate_weight<osd_min_weight){
                    osd_min_weight = candidate_weight;
                    for(int i=0; i<this->pcm->n; i++){
                        this->osdw_decoding[i] = candidate_solution[i];
                    }
                }

            }

            return this->osdw_decoding;

        }

};

}//end osd namespace


#endif