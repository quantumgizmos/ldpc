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

const vector<int> NULL_INT_VECTOR = {};

class bp_entry: public EntryBase<bp_entry>{ 
    public:  
        double bit_to_check_msg=0.0;
        double check_to_bit_msg=0.0;
        uint8_t value = uint8_t(0);
        ~bp_entry(){};
};


typedef GF2Sparse<bp_entry> bp_sparse;


class bp_decoder{
    public:
        bp_sparse* pcm;
        int check_count,max_iter,decoding_method, schedule;
        int bit_count;
        double alpha;
        vector<uint8_t> decoding;
        vector<uint8_t> candidate_syndrome;
        vector<uint8_t> synd;
        vector<double> channel_probs;
        vector<double> log_prob_ratios;
        vector<double> initial_log_prob_ratios;
        vector<int> serial_schedule_order;
        int random_serial_schedule;
        int iterations;
        int omp_thread_count;
        bool converge;
        unsigned seed;

        bp_decoder(
            bp_sparse *matrix,
            vector<double>& channel_probabilities,
            int maximum_iterations,
            int bp_method=1,
            double min_sum_scaling_factor=0.625,
            int bp_schedule=1,
            int omp_threads = 1,
            vector<int> serial_schedule = NULL_INT_VECTOR,
            int random_schedule = 0){
            
            pcm = matrix;
            check_count = pcm->m;
            bit_count = pcm->n;
            channel_probs=channel_probabilities;
            max_iter=maximum_iterations;
            initial_log_prob_ratios.resize(bit_count);
            log_prob_ratios.resize(bit_count);
            candidate_syndrome.resize(check_count);
            decoding.resize(bit_count);
            converge=0;
            alpha = min_sum_scaling_factor;
            decoding_method=bp_method;
            iterations=0;
            schedule = bp_schedule;
            omp_thread_count = omp_threads;

            if(serial_schedule != NULL_INT_VECTOR){
                serial_schedule_order = serial_schedule;
            }
            else{
                serial_schedule_order.resize(bit_count);
                for(int i = 0; i < bit_count; i++) serial_schedule_order[i] = i;
                if(random_schedule>0){
                    seed = std::chrono::system_clock::now().time_since_epoch().count();
                    shuffle(serial_schedule_order.begin(), serial_schedule_order.end(), std::default_random_engine(seed));
                }
            }
            random_serial_schedule = random_schedule;

        }

        vector<uint8_t> decode(vector<uint8_t>& syndrome){

            if(schedule==0) bp_decode_parallel(syndrome);
            else if(schedule==1) return bp_decode_serial(syndrome);
            return decoding;

        }

        vector<uint8_t>& bp_decode_parallel(vector<uint8_t>& syndrome){

            synd=syndrome;
            converge=0;
            double temp;
            int CONVERGED = false;

            // initialise_log_domain_bp();

            if(omp_thread_count == 0) omp_set_num_threads(omp_get_max_threads());
            else omp_set_num_threads(omp_thread_count);

            #pragma omp parallel private(temp)
            {

                // initialise BP
                #pragma omp for
                for(int i=0;i<bit_count;i++){
                    log_prob_ratios[i]=log((1-channel_probs[i])/channel_probs[i]);
                    initial_log_prob_ratios[i] = log_prob_ratios[i];

                    for(auto e: pcm->iterate_column(i)){
                        e->bit_to_check_msg = log_prob_ratios[i];
                    }
                }

                //main interation loop
                for(int it=1;it<=max_iter;it++){

                    if(CONVERGED) continue;
                    if(decoding_method==0){
                        #pragma omp for
                        for(int i=0;i<check_count;i++){
                            
                            temp=1.0;
                            for(auto e: pcm->iterate_row(i)){
                                e->check_to_bit_msg=temp;
                                temp*=tanh(e->bit_to_check_msg/2);
                            }

                            temp=1;
                            for(auto e: pcm->reverse_iterate_row(i)){
                                e->check_to_bit_msg*=temp;
                                e->check_to_bit_msg = pow(-1,synd[i])*log((1+e->check_to_bit_msg)/(1-e->check_to_bit_msg));
                                temp*=tanh(e->bit_to_check_msg/2);
                            }
                        }
                    }

                    else if(decoding_method==1){
                        //check to bit updates
                        #pragma omp for
                        for(int i=0;i<check_count;i++){

                            int total_sgn,sgn;
                            total_sgn=synd[i];
                            temp = numeric_limits<double>::max();

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
                                
                                e->check_to_bit_msg*=pow(-1.0,sgn)*alpha;
                            
                                if(abs(e->bit_to_check_msg)<temp){
                                    temp = abs(e->bit_to_check_msg);
                                }
                                
                            }
                            
                        }
                    }


                    //compute log probability ratios
                    #pragma omp for
                    for(int i=0;i<bit_count;i++){
                        temp=initial_log_prob_ratios[i];
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
                        if(candidate_syndrome[i]!=synd[i]){
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
                        temp=0;
                        for(auto e: pcm->reverse_iterate_column(i)){
                            e->bit_to_check_msg+=temp;
                            temp+=e->check_to_bit_msg;
                        }
                    }
        
                }
            }

            converge=CONVERGED;
            return decoding;

        }

        vector<uint8_t>& bp_decode_serial(vector<uint8_t>& syndrome){

            synd=syndrome;
            int check_index;
            converge=0;
            // int it;
            int CONVERGED = 0;
            bool loop_break = false;

            if(omp_thread_count == 0) omp_set_num_threads(omp_get_max_threads());
            else omp_set_num_threads(omp_thread_count);

            // initialise BP

            for(int i=0;i<bit_count;i++){
                log_prob_ratios[i]=log((1-channel_probs[i])/channel_probs[i]);
                initial_log_prob_ratios[i] = log_prob_ratios[i];

                for(auto e: pcm->iterate_column(i)){
                    e->bit_to_check_msg = log_prob_ratios[i];
                }
            }



            for(int it=1;it<=max_iter;it++){

                if(CONVERGED) continue;

                if(random_serial_schedule>1 && omp_thread_count == 1){
                    shuffle(serial_schedule_order.begin(), serial_schedule_order.end(), std::default_random_engine(seed));
                }

                #pragma omp parallel for
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
                            e->check_to_bit_msg = pow(-1,synd[check_index])*log((1+e->check_to_bit_msg)/(1-e->check_to_bit_msg));
                            e->bit_to_check_msg=log_prob_ratios[bit_index];
                            log_prob_ratios[bit_index]+=e->check_to_bit_msg;
                        }
                    }
                    else if(decoding_method==1){
                        for(auto e: pcm->iterate_column(bit_index)){
                            check_index = e->row_index;
                            int sgn=synd[check_index];
                            temp = numeric_limits<double>::max();
                            for(auto g: pcm->iterate_row(check_index)){
                                if(g!=e){
                                    if(abs(g->bit_to_check_msg)<temp) temp = abs(g->bit_to_check_msg);
                                    if(g->bit_to_check_msg<=0) sgn+=1;
                                }
                            }
                            e->check_to_bit_msg = alpha*pow(-1,sgn)*temp;
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
                    if(candidate_syndrome[i]!=synd[i]){
                        CONVERGED=0;
                        loop_break=true;
                    }

                }

                iterations = it;

            

            }


            // print_vector(synd);
            // print_vector(candidate_syndrome);
            // print_vector(decoding);
            // cout<<endl;
            converge=CONVERGED;
            return decoding;

        }

        ~bp_decoder(){};

};

typedef bp_entry cybp_entry;

#endif