#ifndef BP_H
#define BP_H

#include <vector>
#include <memory>
#include <iterator>
#include <cmath> 
#include <limits>
#include <omp.h>
#include <random>
#include <chrono>       
#include "sparse_matrix.hpp"
#include "sparse_matrix_util.hpp"
#include "gf2sparse.hpp"


using namespace std;
using namespace sparse_matrix;
using namespace gf2sparse;

namespace bp{

const vector<int> NULL_INT_VECTOR = {};

class BpEntry: public EntryBase<BpEntry>{ 
    public:  
        double bit_to_check_msg=0.0;
        double check_to_bit_msg=0.0;
        ~BpEntry(){};
};


typedef GF2Sparse<BpEntry> BpSparse;


class BpDecoder{
    public:
        shared_ptr<BpSparse> pcm;
        int check_count,max_iter,decoding_method, schedule;
        int bit_count;
        double ms_scaling_factor;
        vector<uint8_t> decoding;
        vector<uint8_t> candidate_syndrome;
        vector<double>& channel_probs;
        vector<double> log_prob_ratios;
        vector<double> initial_log_prob_ratios;
        vector<int> serial_schedule_order;
        int random_serial_schedule;
        int iterations;
        int omp_thread_count;
        bool converge;
        unsigned random_schedule_seed;

        BpDecoder(
            shared_ptr<BpSparse> matrix,
            vector<double>& channel_probabilities,
            int maximum_iterations,
            int bp_method=1,
            double min_sum_scaling_factor = 0.625,
            int bp_schedule=0,
            int omp_threads = 1,
            vector<int> serial_schedule = NULL_INT_VECTOR,
            int random_schedule = 0):
            channel_probs(channel_probabilities){
            
            this->pcm = matrix;
            this->check_count = pcm->m;
            this->bit_count = pcm->n;
            // this->channel_probs=channel_probabilities;
            this->ms_scaling_factor=min_sum_scaling_factor;
            this->max_iter=maximum_iterations;
            this->initial_log_prob_ratios.resize(bit_count);
            this->log_prob_ratios.resize(bit_count);
            this->candidate_syndrome.resize(check_count);
            this->decoding.resize(bit_count);
            this->converge=0;
            this->decoding_method=bp_method;
            this->iterations=0;
            this->schedule = bp_schedule;
            this->omp_thread_count = omp_threads;
            this->random_schedule_seed = random_schedule_seed;

            if(serial_schedule != NULL_INT_VECTOR){
                this->serial_schedule_order = serial_schedule;
            }
            else{
                this->serial_schedule_order.resize(bit_count);
                for(int i = 0; i < bit_count; i++) this->serial_schedule_order[i] = i;
                if(random_schedule>0){
                    random_schedule_seed = std::chrono::system_clock::now().time_since_epoch().count();
                    shuffle(this->serial_schedule_order.begin(), this->serial_schedule_order.end(), std::default_random_engine(random_schedule_seed));
                }
            }
            this->random_serial_schedule = random_schedule;

            //Initialise OMP thread pool
            omp_set_num_threads(this->omp_thread_count);

        }

        ~BpDecoder(){};


        void initialise_log_domain_bp(){
            // initialise BP
            #pragma omp for
            for(int i=0;i<this->bit_count;i++){
                this->initial_log_prob_ratios[i] = log((1-this->channel_probs[i])/this->channel_probs[i]);

                for(auto e: this->pcm->iterate_column(i)){
                    e->bit_to_check_msg = this->initial_log_prob_ratios[i];
                }
            }
        }

        vector<uint8_t> decode(vector<uint8_t>& syndrome){

            if(schedule==0) bp_decode_parallel(syndrome);
            else if(schedule==1) return bp_decode_serial(syndrome);
            return decoding;

        }

        vector<uint8_t>& bp_decode_parallel(vector<uint8_t>& syndrome){

            converge=0;
            int CONVERGED = false;

            initialise_log_domain_bp();

            //main interation loop
            for(int it=1;it<=max_iter;it++){

                if(CONVERGED) continue;
                if(decoding_method==0){
                    #pragma omp for
                    for(int i=0;i<check_count;i++){
                        double temp=1.0;
                        for(auto e: pcm->iterate_row(i)){
                            e->check_to_bit_msg=temp;
                            temp*=tanh(e->bit_to_check_msg/2);
                        }

                        temp=1;
                        for(auto e: pcm->reverse_iterate_row(i)){
                            e->check_to_bit_msg*=temp;
                            e->check_to_bit_msg = pow(-1,syndrome[i])*log((1+e->check_to_bit_msg)/(1-e->check_to_bit_msg));
                            temp*=tanh(e->bit_to_check_msg/2);
                        }
                    }
                }

                else if(decoding_method==1){
                    //check to bit updates
                    #pragma omp for
                    for(int i=0;i<check_count;i++){

                        int total_sgn,sgn;
                        total_sgn=syndrome[i];
                        double temp = numeric_limits<double>::max();

                        for(auto e: pcm->iterate_row(i)){
                            if(e->bit_to_check_msg<=0) total_sgn+=1;   
                            e->check_to_bit_msg = abs(temp);
                            if(abs(e->bit_to_check_msg)<temp){
                                temp = abs(e->bit_to_check_msg);
                            }
                        }

                        temp = numeric_limits<double>::max();
                        for(auto e: pcm->reverse_iterate_row(i)){
                            sgn=total_sgn;
                            if(e->bit_to_check_msg<=0) sgn+=1;
                            if(temp<e->check_to_bit_msg){
                                e->check_to_bit_msg = temp;
                            }
                            
                            e->check_to_bit_msg*=pow(-1.0,sgn)*ms_scaling_factor;
                        
                            if(abs(e->bit_to_check_msg)<temp){
                                temp = abs(e->bit_to_check_msg);
                            }
                            
                        }
                        
                    }
                }


                //compute log probability ratios
                #pragma omp for
                for(int i=0;i<bit_count;i++){
                    double temp=initial_log_prob_ratios[i];
                    for(auto e: pcm->iterate_column(i)){
                        e->bit_to_check_msg=temp;
                        temp+=e->check_to_bit_msg;
                        if(isnan(temp)) temp = e->bit_to_check_msg;


                    }

                    //make hard decision on basis of log probability ratio for bit i
                    log_prob_ratios[i]=temp;
                    // if(isnan(log_prob_ratios[i])) log_prob_ratios[i] = initial_log_prob_ratios[i];
                    if(temp<=0) decoding[i] = 1;
                    else decoding[i]=0;
                }



                //compute the syndrome for the current candidate decoding solution
                candidate_syndrome = pcm->mulvec_parallel(decoding,candidate_syndrome);
                int loop_break = false;
                CONVERGED = true;
                #pragma omp barrier

                #pragma omp for
                for(int i=0;i<check_count;i++){
                    if(loop_break) continue;
                    if(candidate_syndrome[i]!=syndrome[i]){
                        CONVERGED=false;
                        loop_break=true;
                    }

                }
                
                #pragma omp barrier
                iterations = it;

                if(CONVERGED) continue;

                
                //compute bit to check update
                #pragma omp for
                for(int i=0;i<bit_count;i++){
                    double temp=0;
                    for(auto e: pcm->reverse_iterate_column(i)){
                        e->bit_to_check_msg+=temp;
                        temp+=e->check_to_bit_msg;
                    }
                }
    
            }
        

            converge=CONVERGED;
            return decoding;

        }

        vector<uint8_t>& bp_decode_serial(vector<uint8_t>& syndrome){

            int check_index;
            converge=0;
            // int it;
            int CONVERGED = 0;
            bool loop_break = false;

            // initialise BP

            this->initialise_log_domain_bp();



            for(int it=1;it<=max_iter;it++){

                if(CONVERGED) continue;

                if(random_serial_schedule>1 && omp_thread_count == 1){
                    shuffle(serial_schedule_order.begin(), serial_schedule_order.end(), std::default_random_engine(random_schedule_seed));
                }

                #pragma omp for
                for(int bit_index: serial_schedule_order){
                    double temp;
                    log_prob_ratios[bit_index]=log((1-channel_probs[bit_index])/channel_probs[bit_index]);
                    // cout<<log_prob_ratios[bit_index]<<endl;
                
                    if(decoding_method==0){
                        for(auto e: pcm->iterate_column(bit_index)){
                            check_index = e->row_index;
                            
                            e->check_to_bit_msg=1.0;
                            for(auto g: pcm->iterate_row(check_index)){
                                if(g!=e){
                                    e->check_to_bit_msg*=tanh(g->bit_to_check_msg/2);
                                }
                            }
                            e->check_to_bit_msg = pow(-1,syndrome[check_index])*log((1+e->check_to_bit_msg)/(1-e->check_to_bit_msg));
                            e->bit_to_check_msg=log_prob_ratios[bit_index];
                            log_prob_ratios[bit_index]+=e->check_to_bit_msg;
                        }
                    }
                    else if(decoding_method==1){
                        for(auto e: pcm->iterate_column(bit_index)){
                            check_index = e->row_index;
                            int sgn=syndrome[check_index];
                            temp = numeric_limits<double>::max();
                            for(auto g: pcm->iterate_row(check_index)){
                                if(g!=e){
                                    if(abs(g->bit_to_check_msg)<temp) temp = abs(g->bit_to_check_msg);
                                    if(g->bit_to_check_msg<=0) sgn+=1;
                                }
                            }
                            e->check_to_bit_msg = ms_scaling_factor*pow(-1,sgn)*temp;
                            e->bit_to_check_msg=log_prob_ratios[bit_index];
                            log_prob_ratios[bit_index]+=e->check_to_bit_msg;
                        }
                    
                    }

                    if(log_prob_ratios[bit_index]<=0) decoding[bit_index] = 1;
                    else decoding[bit_index]=0;

                    temp=0;
                    for(auto e: pcm->reverse_iterate_column(bit_index)){
                        e->bit_to_check_msg+=temp;
                        temp += e->check_to_bit_msg;
                    }

                }

            
            

                // compute the syndrome for the current candidate decoding solution
                loop_break = false;
                CONVERGED = 1;

                
                candidate_syndrome = pcm->mulvec(decoding,candidate_syndrome);

                for(int i=0;i<check_count;i++){
                    if(loop_break) continue;
                    if(candidate_syndrome[i]!=syndrome[i]){
                        CONVERGED=0;
                        loop_break=true;
                    }

                }

                iterations = it;

            

            }

            converge=CONVERGED;
            return decoding;

        }


};

} // end namespace bp

typedef bp::BpEntry cybp_entry;

#endif