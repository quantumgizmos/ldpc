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
#include <stdexcept> // required for std::runtime_error
#include <set>

#include "sparse_matrix.hpp"
#include "sparse_matrix_util.hpp"
#include "gf2sparse.hpp"


using namespace std;
// using namespace sparse_matrix_base;
// using namespace gf2sparse;

namespace bp{

enum BpMethod{
    PRODUCT_SUM = 0,
    MINIMUM_SUM = 1
};

enum BpSchedule{
    SERIAL = 0,
    PARALLEL = 1,
};

const vector<int> NULL_INT_VECTOR = {};

class BpEntry: public sparse_matrix_base::EntryBase<BpEntry>{ 
    public:  
        double bit_to_check_msg=0.0;
        double check_to_bit_msg=0.0;
        ~BpEntry() = default;
};


typedef gf2sparse::GF2Sparse<BpEntry> BpSparse;


class BpDecoder{
    public:
        BpSparse& pcm;
        vector<double> channel_probabilities;
        int check_count;
        int bit_count;
        int maximum_iterations;
        BpMethod bp_method;
        BpSchedule schedule;
        double ms_scaling_factor;
        vector<uint8_t> decoding;
        vector<uint8_t> candidate_syndrome;
        
        vector<double> log_prob_ratios;
        vector<double> initial_log_prob_ratios;
        vector<double> soft_syndrome;
        vector<int> serial_schedule_order;
        int iterations;
        int omp_thread_count;
        bool converge;
        unsigned random_schedule_seed;
        bool random_schedule_at_every_iteration;

        BpDecoder(
            BpSparse& parity_check_matrix,
            vector<double> channel_probabilities,
            int maximum_iterations = 0,
            BpMethod bp_method = PRODUCT_SUM,
            BpSchedule schedule = PARALLEL,
            double min_sum_scaling_factor = 0.625,
            int omp_threads = 1,
            vector<int> serial_schedule = NULL_INT_VECTOR,
            int random_schedule_seed = 0,
            bool random_schedule_at_every_iteration = false):
            pcm(parity_check_matrix) //the parity check matrix is passed in by reference
            {
            
            this->check_count = pcm.m;
            this->bit_count = pcm.n;
            this->channel_probabilities = channel_probabilities;
            this->ms_scaling_factor=min_sum_scaling_factor;
            this->maximum_iterations=maximum_iterations;
            this->initial_log_prob_ratios.resize(bit_count);
            this->log_prob_ratios.resize(bit_count);
            this->candidate_syndrome.resize(check_count);
            this->decoding.resize(bit_count);
            this->converge=0;
            this->bp_method=bp_method;
            this->iterations=0;
            this->schedule = schedule;
            this->omp_thread_count = omp_threads;
            this->random_schedule_seed = random_schedule_seed;
            this->random_schedule_at_every_iteration = random_schedule_at_every_iteration;

            if(serial_schedule != NULL_INT_VECTOR){
                this->serial_schedule_order = serial_schedule;
            }
            else{
                this->serial_schedule_order.resize(bit_count);
                for(int i = 0; i < bit_count; i++) this->serial_schedule_order[i] = i;
                if(random_schedule_seed>0){
                    random_schedule_seed = std::chrono::system_clock::now().time_since_epoch().count();
                    shuffle(this->serial_schedule_order.begin(), this->serial_schedule_order.end(), std::default_random_engine(random_schedule_seed));
                }
            }

            //Initialise OMP thread pool
            this->omp_thread_count = omp_threads;
            this->set_omp_thread_count(this->omp_thread_count);
        }

        ~BpDecoder() = default;

        void set_omp_thread_count(int count){
            this->omp_thread_count = count;
            omp_set_num_threads(this->omp_thread_count);
        }


        void initialise_log_domain_bp(){
            // initialise BP
            #pragma omp for
            for(int i=0;i<this->bit_count;i++){
                this->initial_log_prob_ratios[i] = log((1-this->channel_probabilities[i])/this->channel_probabilities[i]);

                for(auto& e: this->pcm.iterate_column(i)){
                    e.bit_to_check_msg = this->initial_log_prob_ratios[i];
                }
            }
        }

        vector<uint8_t> decode(vector<uint8_t>& syndrome){

            if(schedule == PARALLEL) return bp_decode_parallel(syndrome);
            else if(schedule == SERIAL) return bp_decode_serial(syndrome);
            else throw std::runtime_error("Invalid BP schedule");

        }

        vector<uint8_t>& bp_decode_parallel(vector<uint8_t>& syndrome){

            converge=0;
            int CONVERGED = false;

            initialise_log_domain_bp();

            //main interation loop
            for(int it=1;it<=maximum_iterations;it++){

                if(CONVERGED) continue;

                std::fill(candidate_syndrome.begin(), candidate_syndrome.end(), 0);

                if(bp_method == PRODUCT_SUM){
                    #pragma omp for
                    for(int i=0;i<check_count;i++){
                        double temp=1.0;
                        for(auto& e: pcm.iterate_row(i)){
                            e.check_to_bit_msg=temp;
                            temp*=tanh(e.bit_to_check_msg/2);
                        }

                        temp=1;
                        for(auto& e: pcm.reverse_iterate_row(i)){
                            e.check_to_bit_msg*=temp;
                            e.check_to_bit_msg = pow(-1,syndrome[i])*log((1+e.check_to_bit_msg)/(1-e.check_to_bit_msg));
                            temp*=tanh(e.bit_to_check_msg/2);
                        }
                    }
                }

                else if(bp_method == MINIMUM_SUM){
                    //check to bit updates
                    #pragma omp for
                    for(int i=0;i<check_count;i++){

                        int total_sgn,sgn;
                        total_sgn=syndrome[i];
                        double temp = numeric_limits<double>::max();

                        for(auto& e: pcm.iterate_row(i)){
                            if(e.bit_to_check_msg<=0) total_sgn+=1;   
                            e.check_to_bit_msg = abs(temp);
                            if(abs(e.bit_to_check_msg)<temp){
                                temp = abs(e.bit_to_check_msg);
                            }
                        }

                        temp = numeric_limits<double>::max();
                        for(auto& e: pcm.reverse_iterate_row(i)){
                            sgn=total_sgn;
                            if(e.bit_to_check_msg<=0) sgn+=1;
                            if(temp<e.check_to_bit_msg){
                                e.check_to_bit_msg = temp;
                            }
                            
                            e.check_to_bit_msg*=pow(-1.0,sgn)*ms_scaling_factor;
                        
                            if(abs(e.bit_to_check_msg)<temp){
                                temp = abs(e.bit_to_check_msg);
                            }
                            
                        }
                        
                    }
                }


                //compute log probability ratios
                #pragma omp for
                for(int i=0;i<bit_count;i++){
                    double temp=initial_log_prob_ratios[i];
                    for(auto& e: pcm.iterate_column(i)){
                        e.bit_to_check_msg=temp;
                        temp+=e.check_to_bit_msg;
                        // if(isnan(temp)) temp = e.bit_to_check_msg;


                    }

                    //make hard decision on basis of log probability ratio for bit i
                    log_prob_ratios[i]=temp;
                    // if(isnan(log_prob_ratios[i])) log_prob_ratios[i] = initial_log_prob_ratios[i];
                    if(temp<=0){
                        decoding[i] = 1;
                        for(auto& e: pcm.iterate_column(i)){
                            candidate_syndrome[e.row_index]^=1;                        }
                    }
                    else decoding[i]=0;
                }



                //compute the syndrome for the current candidate decoding solution
                // candidate_syndrome = pcm.mulvec_parallel(decoding,candidate_syndrome);
                int loop_break = false;
                CONVERGED = false;
                #pragma omp barrier


                if(std::equal(candidate_syndrome.begin(), candidate_syndrome.end(), syndrome.begin())){
                    CONVERGED = true;
                }

                // #pragma omp for
                // for(int i=0;i<check_count;i++){
                //     if(loop_break) continue;
                //     if(candidate_syndrome[i]!=syndrome[i]){
                //         CONVERGED=false;
                //         loop_break=true;
                //     }

                // }
                
                iterations = it;

                if(CONVERGED) continue;

                
                //compute bit to check update
                #pragma omp for
                for(int i=0;i<bit_count;i++){
                    double temp=0;
                    for(auto& e: pcm.reverse_iterate_column(i)){
                        e.bit_to_check_msg+=temp;
                        temp+=e.check_to_bit_msg;
                    }
                }
    
            }
        

            converge=CONVERGED;
            return decoding;

        }

        vector<uint8_t>& bp_decode_single_scan(vector<uint8_t>& syndrome){

            converge=0;
            int CONVERGED = false;

            vector<double> log_prob_ratios_old;
            log_prob_ratios_old.resize(bit_count);

            for(int i=0;i<bit_count;i++){
                this->initial_log_prob_ratios[i] = log((1-this->channel_probabilities[i])/this->channel_probabilities[i]);
                this->log_prob_ratios[i] = this->initial_log_prob_ratios[i];

            }

            // initialise_log_domain_bp();

            //main interation loop
            for(int it=1;it<=maximum_iterations;it++){

                if(CONVERGED) continue;

                // std::fill(candidate_syndrome.begin(), candidate_syndrome.end(), 0);

                log_prob_ratios_old = this->log_prob_ratios;
                
                if(it != 1) {
                    this->log_prob_ratios = this->initial_log_prob_ratios;
                }

                //check to bit updates
                for(int i=0;i<check_count;i++){

                    this->candidate_syndrome[i] = 0;

                    int total_sgn,sgn;
                    total_sgn=syndrome[i];
                    double temp = numeric_limits<double>::max();

                    double bit_to_check_msg;

                    for(auto& e: pcm.iterate_row(i)){
                        if( it == 1 ) e.check_to_bit_msg = 0;
                        bit_to_check_msg = log_prob_ratios_old[e.col_index] - e.check_to_bit_msg;
                        if(bit_to_check_msg<=0) total_sgn+=1;   
                        e.bit_to_check_msg = temp;
                        double abs_bit_to_check_msg = abs(bit_to_check_msg);
                        if(abs_bit_to_check_msg < temp){
                            temp = abs_bit_to_check_msg;
                        }
                    }

                    temp = numeric_limits<double>::max();
                    for(auto& e: pcm.reverse_iterate_row(i)){
                        sgn=total_sgn;
                        if( it == 1 ) e.check_to_bit_msg = 0;
                        bit_to_check_msg = log_prob_ratios_old[e.col_index] - e.check_to_bit_msg;
                        if(bit_to_check_msg<=0) sgn+=1;
                        if(temp<e.bit_to_check_msg){
                            e.bit_to_check_msg = temp;
                        }
                        
                        e.check_to_bit_msg = pow(-1.0,sgn)*ms_scaling_factor*e.bit_to_check_msg;
                        this->log_prob_ratios[e.col_index] += e.check_to_bit_msg;


                        double abs_bit_to_check_msg = abs(bit_to_check_msg);
                        if(abs_bit_to_check_msg < temp){
                            temp = abs_bit_to_check_msg;
                        }
                        
                    }

                    
                    
                }
                


                //compute hard decisions and calculate syndrome
                for(int i=0;i<bit_count;i++){
                    if(this->log_prob_ratios[i]<=0){
                        this->decoding[i] = 1;
                        for(auto& e: pcm.iterate_column(i)){
                            this->candidate_syndrome[e.row_index]^=1;                        }
                    }
                    else this->decoding[i]=0;
                }

                int loop_break = false;
                CONVERGED = false;

                if(std::equal(candidate_syndrome.begin(), candidate_syndrome.end(), syndrome.begin())){
                    CONVERGED = true;
                }

                iterations = it;

                if(CONVERGED){
                    converge = CONVERGED;
                    return decoding;
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



            for(int it=1;it<=maximum_iterations;it++){

                if(CONVERGED) continue;

                if(this->random_schedule_at_every_iteration){
                    shuffle(serial_schedule_order.begin(), serial_schedule_order.end(), std::default_random_engine(random_schedule_seed));
                }

                #pragma omp for
                for(int bit_index: serial_schedule_order){
                    double temp;
                    log_prob_ratios[bit_index]=log((1-channel_probabilities[bit_index])/channel_probabilities[bit_index]);
                    // cout<<log_prob_ratios[bit_index]<<endl;
                
                    if(bp_method==0){
                        for(auto& e: pcm.iterate_column(bit_index)){
                            check_index = e.row_index;
                            
                            e.check_to_bit_msg=1.0;
                            for(auto& g: pcm.iterate_row(check_index)){
                                if(&g != &e){
                                    e.check_to_bit_msg*=tanh(g.bit_to_check_msg/2);
                                }
                            }
                            e.check_to_bit_msg = pow(-1,syndrome[check_index])*log((1+e.check_to_bit_msg)/(1-e.check_to_bit_msg));
                            e.bit_to_check_msg=log_prob_ratios[bit_index];
                            log_prob_ratios[bit_index]+=e.check_to_bit_msg;
                        }
                    }
                    else if(bp_method==1){
                        for(auto& e: pcm.iterate_column(bit_index)){
                            check_index = e.row_index;
                            int sgn=syndrome[check_index];
                            temp = numeric_limits<double>::max();
                            for(auto& g: pcm.iterate_row(check_index)){
                                if(&g != &e){
                                    if(abs(g.bit_to_check_msg)<temp) temp = abs(g.bit_to_check_msg);
                                    if(g.bit_to_check_msg<=0) sgn+=1;
                                }
                            }
                            e.check_to_bit_msg = ms_scaling_factor*pow(-1,sgn)*temp;
                            e.bit_to_check_msg=log_prob_ratios[bit_index];
                            log_prob_ratios[bit_index]+=e.check_to_bit_msg;
                        }
                    
                    }

                    if(log_prob_ratios[bit_index]<=0) decoding[bit_index] = 1;
                    else decoding[bit_index]=0;

                    temp=0;
                    for(auto& e: pcm.reverse_iterate_column(bit_index)){
                        e.bit_to_check_msg+=temp;
                        temp += e.check_to_bit_msg;
                    }

                }

            
            

                // compute the syndrome for the current candidate decoding solution
                loop_break = false;
                CONVERGED = 1;

                
                candidate_syndrome = pcm.mulvec(decoding,candidate_syndrome);

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

//         vector<uint8_t>& soft_info_decode_serial(vector<double>& soft_info_syndrome, double cutoff, double sigma){

//             //calculate the syndrome log-likelihoods
//             this->soft_syndrome = soft_info_syndrome;
//             for(int i = 0; i<this->check_count; i++){
//                 this->soft_syndrome[i] = 2*this->soft_syndrome[i]/(sigma*sigma);
//             }


//             //calculate the hard syndrome from the log-likelihoods
//             vector<uint8_t> syndrome;
//             for(double value: this->soft_syndrome){
//                 if(value<=0){
//                     syndrome.push_back(1);
//                 }
//                 else{
//                     syndrome.push_back(0);
//                 }
//             }

//             int check_index;
//             converge=0;

//             int CONVERGED = 0;
//             bool loop_break = false;

//             // initialise BP

//             this->initialise_log_domain_bp();


//             set<int> check_indices_updated;

//             for(int it=1;it<=maximum_iterations;it++){

//                 if(CONVERGED) continue;

//                 if(random_serial_schedule>1 && omp_thread_count == 1){
//                     shuffle(serial_schedule_order.begin(), serial_schedule_order.end(), std::default_random_engine(random_schedule_seed));
//                 }

//                 check_indices_updated.clear();
//                 #pragma omp for
//                 for(int bit_index: serial_schedule_order){
//                     double temp;
//                     log_prob_ratios[bit_index]=log((1-channel_probabilities[bit_index])/channel_probabilities[bit_index]);
//                     // cout<<log_prob_ratios[bit_index]<<endl;
                
//                     if(bp_method==0){
//                         for(auto& e: pcm.iterate_column(bit_index)){
//                             check_index = e.row_index;
                            
//                             e.check_to_bit_msg=1.0;
//                             for(auto& g: pcm.iterate_row(check_index)){
//                                 if(&g != &e){
//                                     e.check_to_bit_msg*=tanh(g.bit_to_check_msg/2);
//                                 }
//                             }
//                             e.check_to_bit_msg = pow(-1,syndrome[check_index])*log((1+e.check_to_bit_msg)/(1-e.check_to_bit_msg));
//                             e.bit_to_check_msg=log_prob_ratios[bit_index];
//                             log_prob_ratios[bit_index]+=e.check_to_bit_msg;
//                         }
//                     }
//                     else if(bp_method==1){
//                         for(auto& e: pcm.iterate_column(bit_index)){
//                             check_index = e.row_index;
//                             // int sgn=syndrome[check_index];
//                             int sgn = 0;
//                             temp = numeric_limits<double>::max();

//                             for(auto& g: pcm.iterate_row(check_index)){
//                                 if(&g != &e){
//                                     if(abs(g.bit_to_check_msg)<temp){
//                                         temp = abs(g.bit_to_check_msg);
//                                     }
//                                     if(g.bit_to_check_msg<=0) sgn^=1;
//                                 }
//                             }


//                             double min_bit_to_check_msg = temp;
//                             double propagated_msg = min_bit_to_check_msg;

//                             //VIRTUAL CHECK NODE UPDATE

//                             //first we calculate the magnitude of the soft syndrome
//                             double soft_syndrome_magnitude = abs(this->soft_syndrome[check_index]);
                            
//                             //then we check if the magnitude is less than the cutoff.
  
                                
//                             if(soft_syndrome_magnitude<cutoff){

//                                 if(soft_syndrome_magnitude<abs(min_bit_to_check_msg)){
//                                     propagated_msg = soft_syndrome_magnitude;

//                                     int check_node_sgn = sgn;
//                                     if(e.bit_to_check_msg<=0) check_node_sgn^=1;

//                                     if(check_node_sgn==syndrome[check_index]){
//                                         if(abs(e.bit_to_check_msg)<min_bit_to_check_msg){
//                                             this->soft_syndrome[check_index] = pow(-1,syndrome[check_index])*abs(e.bit_to_check_msg);
//                                         }
//                                         else{
//                                             this->soft_syndrome[check_index] = pow(-1,syndrome[check_index])*min_bit_to_check_msg;
//                                         }
//                                     }
//                                     else{
//                                         syndrome[check_index]^=1;
//                                         this->soft_syndrome[check_index]*=-1;
//                                     }


//                                 } //end syndrome bit message if
                            
//                             } //end cutoff if

                            

//                             sgn^=syndrome[check_index];
//                             e.check_to_bit_msg = ms_scaling_factor*pow(-1,sgn)*propagated_msg;
//                             e.bit_to_check_msg=log_prob_ratios[bit_index];
//                             log_prob_ratios[bit_index]+=e.check_to_bit_msg;


//                         }


                    
//                     }

//                     if(log_prob_ratios[bit_index]<=0) decoding[bit_index] = 1;
//                     else decoding[bit_index]=0;

//                     temp=0;
//                     for(auto& e: pcm.reverse_iterate_column(bit_index)){
//                         e.bit_to_check_msg+=temp;
//                         temp += e.check_to_bit_msg;
//                     }

//                     // cout<<"Iteration: "<<it<<"; Bit index: "<<unsigned(bit_index)<<endl;
//                     // cout<<"Decoding: ";
//                     // print_vector(decoding);
//                     // cout<<"Log Prob Ratios: ";
//                     // print_vector(log_prob_ratios);
//                     // cout<<"Syndrome: ";
//                     // print_vector(syndrome);
//                     // cout<<"Soft Syndrome: ";
//                     // print_vector(this->soft_syndrome);
//                     // cout<<endl;

//                 }

//                 // cout<<"Soft Syndrome: ";
//                 // print_vector(this->soft_syndrome);

            
            

//                 // compute the syndrome for the current candidate decoding solution
//                 loop_break = false;
//                 CONVERGED = 1;

                
//                 candidate_syndrome = pcm.mulvec(decoding,candidate_syndrome);

//                 for(int i=0;i<check_count;i++){
//                     if(loop_break) continue;
//                     if(candidate_syndrome[i]!=syndrome[i]){
//                         CONVERGED=0;
//                         loop_break=true;
//                     }

//                 }

//                 iterations = it;

            

//             }

//             converge=CONVERGED;
//             return decoding;

//         }


};

} // end namespace bp

#endif