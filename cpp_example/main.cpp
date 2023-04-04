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


using namespace std;
using namespace gf2sparse;
using namespace util;


// Define a custom node type. eg. Here I define a node type that
// would be useful for BP.
class example_node: public EntryBase<example_node>{ //inherit from the node parent class in <sparse_matrix.hpp>. 
    public:  
        double bit_to_check_msg=0.0; //you can put whatever data type you like in here.
        double check_to_bit_msg=0.0;
        uint8_t value = uint8_t(0); //you should always specify a value field.
};

typedef GF2Sparse<example_node> gf2custom; //The custom node can then be passed as a template paremeter to `gf2sparse<T>`

int main()
{
    
    
    auto random = rng::RandomNumberGenerator(42);

    for(int i =0 ; i<10; i++){
        cout << random.random_double() << endl;
    }


    for (int i = 0; i < 100; i++)
    {
        auto bin = decimal_to_binary(i,20);
        print_vector(bin);
    }

    return 0;


}