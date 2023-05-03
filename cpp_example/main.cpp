#include <iostream>
#include <sparse_matrix.hpp>
#include "gf2sparse.hpp"
#include "gf2sparse_linalg.hpp"
#include "sparse_matrix_util.hpp"
#include "rapidcsv.h"
#include "io.hpp"
#include <string>
#include "rng.hpp"
#include "util.hpp"
#include "gf2codes.hpp"
// #include "ssf.hpp"


using namespace std;



// Define a custom node type. eg. Here I define a node type that
// would be useful for BP.
class ExampleNode: public sparse_matrix::EntryBase<ExampleNode>{ //inherit from the node parent class in <sparse_matrix.hpp>. 
    public:  
        double bit_to_check_msg=0.0; //you can put whatever data type you like in here.
        double check_to_bit_msg=0.0;
        uint8_t value = uint8_t(0); //you should always specify a value field.
};

typedef gf2sparse::GF2Sparse<ExampleNode> GF2Custom; //The custom node can then be passed as a template paremeter to `gf2sparse<T>`

int main()
{
    
    //repetition code example

    int N = 10;
    auto pcm = GF2Custom::New(N,N); //this is a shared pointer object, so no need to worry about memory management.

    for(auto i = 0; i<N; i++){
        pcm->insert_entry(i,i);
        pcm->insert_entry(i,(i+1)%N);
    }

    
    print_sparse_matrix(*pcm); //this funciton is from "sparse_matrix_util.hpp"


    //filling the matrix. Ok, so now lets fill the matrix with some meta_data. To
    //do this we can use the sparse matrix iterators:

    //Eg. to set the `bit_to_check_msg` to 4 and the `check_to_bit_msg` to 42 for all nodes in column 5, we do the following:

    for(auto e: pcm->iterate_column(5)){
        //here e is a pointer to the node
        e->bit_to_check_msg = 4;
        e->check_to_bit_msg = 42;
    }

    //It is also possible to iterate over the columns:

    for(auto e: pcm->iterate_row(5)){
        e->bit_to_check_msg = 150;
    }


    //GF2 operations
    //Matrix multiplication
    //The `GF2Sparse<T>.mulvec(vector<uint8_t>& input_vector, vector<uint8_t>& output_vector)` can be used for
    // matrix vector multiplication. eg:

    vector<uint8_t> error; //defines error vector
    error.resize(pcm->n,0); //fill error vector with zeros. `pcm->n` returns the number of columns in the pcm.
    error[4] = 1;

    vector<uint8_t> syndrome; //defines syndrome vector
    syndrome.resize(pcm->m,0); //fill error vector with zeros. `pcm->n` returns the number of rows in the pcm.
    syndrome = pcm->mulvec(error, syndrome);

    //nb. Why does mulvec take the syndrome vector (or more generlaly output vector) as an argument? This allows an existing vector to be reused, rather
    // than allocating a new vector each time. This seems to make a difference in Monte Carlo simulations at low error rates.

    cout<<endl;
    cout<<"Error: ";
    print_vector(error);
    cout<<"Syndrome: ";
    print_vector(syndrome);


    auto H = gf2codes::rep_code(5);

    auto coords = H->nonzero_coordinates();

    for(auto c: coords){
        print_vector(c);
    }

    auto h = gf2sparse::GF2Sparse<gf2sparse::GF2Entry>::New(3,3);
    h->insert_entry(0,0);
    print_sparse_matrix(*h);

    auto ker = cy_kernel(h);
    print_sparse_matrix(*ker);

    return 0;


}