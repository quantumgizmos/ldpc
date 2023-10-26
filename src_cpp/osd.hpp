#ifndef BPOSD_H
#define BPOSD_H

#include <vector>
#include <memory>
#include <iterator>
#include <cmath> 
#include <limits>

#include "bp.hpp"
#include "sort.hpp"
#include "util.hpp"
#include "gf2sparse_linalg.hpp"

namespace ldpc::osd{


enum OsdMethod{
    OSD_OFF,
    OSD_0,
    EXHAUSTIVE,
    COMBINATION_SWEEP
};


class OsdDecoder{
    public:
        OsdMethod osd_method;
        int osd_order;
        int k, bit_count, check_count;
        ldpc::bp::BpSparse& pcm;
        std::vector<double>& channel_probabilities;
        std::vector<uint8_t> osd0_decoding;
        std::vector<uint8_t> osdw_decoding;
        std::vector<std::vector<uint8_t>> osd_candidate_strings;
        std::vector<int> column_ordering;
        ldpc::gf2sparse_linalg::RowReduce<ldpc::bp::BpEntry>* LuDecomposition;
        
        OsdDecoder(
            ldpc::bp::BpSparse& parity_check_matrix,
            OsdMethod osd_method,
            int osd_order,
            std::vector<double>& channel_probs):
            pcm(parity_check_matrix),
            channel_probabilities(channel_probs)    
        {

            this->bit_count = this->pcm.n;
            this->check_count = this->pcm.m;

            this->osd_order = osd_order;
            this->osd_method = osd_method;

            this->osd_setup();


        }

        int osd_setup(){

            this->osd_candidate_strings.clear();
            
            if(this->osd_method == OSD_OFF) return 0;

            this->LuDecomposition = new ldpc::gf2sparse_linalg::RowReduce<ldpc::bp::BpEntry>(this->pcm);
            this->column_ordering.resize(this->pcm.n);
            int osd_candidate_string_count;
            this->LuDecomposition->rref(false,true); 
            this->k = this->pcm.n - this->LuDecomposition->rank;

            if(this->osd_method == OSD_0 || this->osd_order==0){
                return 1;
            }

            if(this->osd_method == EXHAUSTIVE){
                osd_candidate_string_count = pow(2,this->osd_order);
                for(int i=1; i<osd_candidate_string_count; i++){
                    this->osd_candidate_strings.push_back(ldpc::util::decimal_to_binary_reverse(i,k));
                }
            }

            if(this->osd_method == COMBINATION_SWEEP){
                for(int i=0; i<k; i++) {
                    std::vector<uint8_t> osd_candidate;
                    osd_candidate.resize(k,0);
                    osd_candidate[i]=1; 
                    this->osd_candidate_strings.push_back(osd_candidate);
                }

                for(int i = 0; i<this->osd_order;i++){
                    for(int j = 0; j<this->osd_order; j++){
                        if(j<=i) continue;
                        std::vector<uint8_t> osd_candidate;
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


        std::vector<uint8_t>& decode(std::vector<uint8_t>& syndrome, std::vector<double>& log_prob_ratios) {

            ldpc::sort::soft_decision_col_sort(log_prob_ratios, this->column_ordering,bit_count);

            if(this->osd_order == 0){
                this->osd0_decoding = this->osdw_decoding =  this->LuDecomposition->fast_solve(syndrome,this->column_ordering);
                return this->osd0_decoding;
            }

            //row reduce the matrix according to the new column ordering
            this->LuDecomposition->rref(false,true,this->column_ordering);

            // find the OSD0 solution
            this->osd0_decoding = this->osdw_decoding = LuDecomposition->lu_solve(syndrome);

            // this->osd0_decoding = this->LuDecomposition->fast_solve(syndrome,this->column_ordering);

            // if(osd_order==0){
            //     return this->osd0_decoding;
            // }

            double candidate_weight, osd_min_weight;

            osd_min_weight=0;
            for(int i=0; i<this->pcm.n; i++){
                if(this->osd0_decoding[i]==1){
                    osd_min_weight+=log(1/this->channel_probabilities[i]);
                }
            }

            std::vector<int> non_pivot_columns;
            std::vector<ldpc::gf2sparse::GF2Entry*> delete_entries;
            for(int i = this->LuDecomposition->rank; i<this->pcm.n; i++){
                int col = this->LuDecomposition->cols[i];
                non_pivot_columns.push_back(col);
                
                for(auto& e: this->LuDecomposition->U.iterate_column(col)){
                    delete_entries.push_back(&e);
                }
            }

            for (auto e: delete_entries){
                this->LuDecomposition->U.remove(*e);
            }
            
            for(auto& candidate_string: this->osd_candidate_strings){

                auto t_syndrome = syndrome;
                int col_index = 0;
                for(int col: non_pivot_columns){
                    if(candidate_string[col_index]==1){
                        for(auto& e: this->pcm.iterate_column(col)){
                            t_syndrome[e.row_index] ^= 1;
                        }
                    }
                    col_index++;
                }

                auto candidate_solution = this->LuDecomposition->lu_solve(t_syndrome);
                for(int i=0; i<k; i++){
                    candidate_solution[non_pivot_columns[i]]=candidate_string[i];
                }
                candidate_weight=0;
                for(int i=0; i<this->pcm.n; i++){
                    if(candidate_solution[i]==1){
                        candidate_weight+=log(1/this->channel_probabilities[i]);
                    }
                }
                if(candidate_weight<osd_min_weight){
                    osd_min_weight = candidate_weight;
                    
                    this->osdw_decoding = candidate_solution;

                }

            }

            return this->osdw_decoding;

        }

};

}//end osd namespace


#endif