#ifndef mbp_H
#define mbp_H

#include <iostream>
#include <memory>
#include <cmath> 
#include <limits>
#include "sparse_matrix.hpp"
#include "sparse_matrix_util.hpp"

using namespace std;
// using namespace gf2sparse;
#define pauli value

//This is the data structure at every non-zero location in the sparse matrix
template <class T = uint8_t>
class mbp_entry: public EntryBase<mbp_entry<T>>{ 
    public:
        double qubit_to_stab_msgs[3]={0.0,0.0,0.0};
        double stab_to_qubit_msgs=0.0;
        T value = T(0); //For mbp decoder, we can access this using the macro "pauli" (see definition above)
        ~mbp_entry(){};
};

typedef mbp_entry<uint8_t> cymbp_entry;

class mbp_sparse: public SparseMatrix<uint8_t,mbp_entry>{
    public:
        typedef SparseMatrix<uint8_t,mbp_entry> BASE;
        using BASE::row_heads; using BASE::column_heads; using BASE::m; using BASE::n;
        // using BASE::lu_decomposition;
        mbp_sparse(int m, int n): BASE::SparseMatrix(m,n){}
        ~mbp_sparse(){}

        //Calculates the Pauli syndrome. Can we make use of the extra information in the GF4 syndrome?
        vector<uint8_t>& pauli_syndrome(vector<uint8_t>& input_vector, vector<uint8_t>& output_vector){
            int row_sum;
            for(int stab=0; stab<m;stab++){
                output_vector[stab]=0;
                row_sum=0;
                for(auto e: BASE::iterate_row_ptr(stab)){
                    if(input_vector[e->col_index]!=0 && input_vector[e->col_index]!=e->pauli){
                        row_sum+=1;
                    }
                }
                output_vector[stab] = row_sum % 2;
            }
            return output_vector;
        }

        int print(){
            ldpc::sparse_matrix_util::print_sparse_matrix(*this);
            return 1;
        }

};

class mbp_decoder{
    public:
        mbp_sparse* pcm;
        int stab_count,max_iter,decoding_method;
        int qubit_count;
        // double alpha;
        double beta;
        double gamma;
        vector<uint8_t> decoding;
        vector<uint8_t> candidate_syndrome;
        vector<uint8_t> synd;
        vector<vector<double>> channel_probs;
        vector<vector<double>> log_prob_ratios;
        vector<vector<double>> alpha;
        int iterations;
        bool converge;
        int bp_method;

        mbp_decoder(mbp_sparse *matrix, vector<vector<double>>& channel_probabilities, int maximum_iterations, vector<vector<double>> alpha_parameter, double beta_parameter=0.0, int bp_method=0, double min_sum_gamma=1.0){
            
            pcm = matrix;
            stab_count = pcm->m;
            qubit_count = pcm->n;
            channel_probs=channel_probabilities;
            max_iter=maximum_iterations;
            log_prob_ratios.resize(3);
            for(int w=0; w<3; w++) log_prob_ratios[w].resize(qubit_count);
            candidate_syndrome.resize(stab_count);
            decoding.resize(qubit_count);
            converge=0;
            alpha = alpha_parameter;
            beta = beta_parameter;
            decoding_method=bp_method;
            iterations=0;
            this->bp_method=bp_method;
            this->gamma = min_sum_gamma;

        }

        ~mbp_decoder(){
            delete pcm;
        }

    vector<uint8_t>& decode(vector<uint8_t>& syndrome){

        synd = syndrome;

        int stab_index;
        double stab_update_product;
        double stab_update_min;
        int pauli_type;
        double numerator;
        double demoninator;
        bool all_positive;
        double min;
        int iter;
        int sgn;
        double temp;

        converge=false;

        //initialise decoder
        for(int i=0; i<qubit_count; i++){
            for(auto e: pcm->iterate_column_ptr(i)){
                for(int w=0; w<3; w++){
                    if(e->pauli!=(w+1)){
                        e->qubit_to_stab_msgs[w]=log((1-channel_probs[w][i])/channel_probs[w][i]);
                    }
                    else e->qubit_to_stab_msgs[w] = 0;
                }
            }
        }

        //main iteration loop
        for(iter=1;iter<=max_iter;iter++){ 

            //loop over all the qubits
            for(int qubit=0; qubit<qubit_count;qubit++){ 

                if(bp_method==0){
                // Product sum update. We should also implement min-sum at some point?
                for(auto e: pcm->iterate_column_ptr(qubit)){

                    stab_index = e->row_index;
                    stab_update_product = 1.0;
                    // cout<<"qubit: "<<qubit<<"; stab: "<<stab_index<<"; Syndrome: "<<unsigned(synd[stab_index])<<"; qubit_neighbours: ";
                    for(auto g: pcm->iterate_row_ptr(stab_index)){
                        
                        if(g!=e){

                            // cout<<g->col_index<<" ";
                        
                            //this is where we merge the qubit_to_stab_probabilities
                            numerator = 1.0+ exp(-1*g->qubit_to_stab_msgs[g->pauli-1]);
                            demoninator=0;
                            for(int w=0; w<3; w++){
                                if(w!=(g->pauli-1)){
                                    demoninator+=exp(-1*g->qubit_to_stab_msgs[w]);
                                }
                            }

                            // @Arthur. Why is this less than or equal?
                            if(demoninator <= 0) {
                                stab_update_product = 1e-12;
                            }

                            stab_update_product*=tanh(log(1e-12+(numerator/demoninator))/2);
                        }

                    }

                    // copying Arthur's fixes for numerical stability.
                    if(stab_update_product>=1) stab_update_product = 1-1e-8;
                    else if(stab_update_product<=-1) stab_update_product = 1+1e-8;

                    // e->stab_to_qubit_msgs = pow(-1,synd[stab_index])*2*atanh(stab_update_product);
                    e->stab_to_qubit_msgs = pow(-1,synd[stab_index])*log((1+stab_update_product)/(1-stab_update_product));



                    // cout<<"; Message: "<<e->stab_to_qubit_msgs<<endl;
                
                }

                }

                else if(bp_method==1){
                // Min sum update.
                for(auto e: pcm->iterate_column_ptr(qubit)){

                    stab_index = e->row_index;
                    stab_update_min = numeric_limits<double>::max();
                    sgn=int(synd[stab_index]);
                    for(auto g: pcm->iterate_row_ptr(stab_index)){
                        
                        if(g!=e){

                            //this is where we merge the qubit_to_stab_probabilities
                            numerator = 1.0+ exp(-1*g->qubit_to_stab_msgs[g->pauli-1]);
                            demoninator=0;
                            for(int w=0; w<3; w++){
                                if(w!=(g->pauli-1)){
                                    demoninator+=exp(-1*g->qubit_to_stab_msgs[w]);
                                }
                            }

                            if(demoninator <= 0) {
                                stab_update_product = 1e-12;
                            }

                            temp=log(1e-12+(numerator/demoninator));
                            if(abs(temp)<stab_update_min) stab_update_min = abs(temp);
                            if(temp<=0) sgn+=1;
                        }

                    }

                    // // copying Arthur's fixes for numerical stability.
                    // if(stab_update_min>=1) stab_update_product = 1-1e-8;
                    // else if(stab_update_product<=-1) stab_update_product = 1+1e-8;

                    e->stab_to_qubit_msgs = pow(-1,sgn)*gamma*stab_update_min;



                    // cout<<"; Message: "<<e->stab_to_qubit_msgs<<endl;
                
                }

                }

                //qubit to stab update
                for(int w=0;w<3;w++){
                    log_prob_ratios[w][qubit] = log( (1-channel_probs[w][qubit])/channel_probs[w][qubit] );
                }
                for(auto e: pcm->iterate_column_ptr(qubit)){
                    for(int w=0; w<3;w++){
                        if(e->pauli!=(w+1)) log_prob_ratios[w][qubit]+=(1/alpha[w][qubit])*e->stab_to_qubit_msgs;
                        else log_prob_ratios[w][qubit]+=beta*e->stab_to_qubit_msgs;
                    }

                }
                // cout<<"qubit: "<<qubit<<"; Log_prob_ratio : ["<<log_prob_ratios[qubit][0]<<", "<<log_prob_ratios[qubit][1]<<", "<<log_prob_ratios[qubit][2]<<"]"<<endl;
                // cout<<endl;

                //Hard decision. There is some ambiguity when two of the probabilites are equal.
                all_positive = true;
                min = numeric_limits<double>::infinity();
                for(int w=0;w<3;w++){
                    if(log_prob_ratios[w][qubit]<min){
                        min = log_prob_ratios[w][qubit];
                        decoding[qubit]=w+1;
                    }
                    if(log_prob_ratios[w][qubit]<0) all_positive=false;
                }

                if(all_positive) decoding[qubit] = 0;

                //inhibition loop
                for(auto e: pcm->iterate_column_ptr(qubit)){
                    for(int w=0; w<3;w++){
                        e->qubit_to_stab_msgs[w]=log_prob_ratios[w][qubit];
                        if(e->pauli!=(w+1)) e->qubit_to_stab_msgs[w] -= e->stab_to_qubit_msgs;
                    }
                }

            }

            // cout<<"Iteration "<<it<<" decoding: ";
            // print_array(decoding,pcm->n);

            //stabing for convergence
            pcm->pauli_syndrome(decoding, candidate_syndrome);
            bool equal=true;
            for(int i=0; i<pcm->m; i++){
                if(synd[i]!=candidate_syndrome[i]){
                    equal = false;
                    break;
                }
            }

            //exit if solution has converged
            if(equal){
                iterations=iter;
                converge=true;
                return decoding;
            }

        }

    //this is where the decoder admits defeat.
    converge = false;
    iterations = iter - 1;
    return decoding;

    }

};

#endif