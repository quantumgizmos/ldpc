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
    
    // //repetition code example

    // int N = 10;
    // auto pcm = GF2Custom::New(N,N); //this is a shared pointer object, so no need to worry about memory management.

    // pcm->insert_entry(0,0);

    // for(auto e: pcm->iterate_row(0)){
    //     e->bit_to_check_msg = 1;
    // }

    // print_sparse_matrix(*pcm);


    auto pcm = gf2codes::hamming_code(3);
    auto pcmT = pcm->transpose();

    // print_sparse_matrix(*pcm);

    auto r = gf2sparse_linalg::kernel(pcm);

    for(auto row: r){
        print_vector(row);
    }


    return 0;


}