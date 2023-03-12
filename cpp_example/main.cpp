#include <iostream>
#include <sparse_matrix.hpp>
#include "gf2sparse.hpp"
#include "sparse_matrix_util.hpp"
#include "rapidcsv.h"
#include "io.hpp"
#include <string>


using namespace std;
using namespace gf2sparse;


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
    
    // //repetition code example

    // int N = 10;
    // auto pcm = new gf2custom(N,N);

    // for(auto i = 0; i<N; i++){
    //     pcm->insert_entry(i,i,1);
    //     pcm->insert_entry(i,(i+1)%N,1);
    // }

    
    // print_sparse_matrix(*pcm); //this funciton is from "sparse_matrix_util.hpp"


    // //filling the matrix. Ok, so now lets fill the matrix with some meta_data. To
    // //do this we can use the sparse matrix iterators:

    // //Eg. to set the `bit_to_check_msg` to 4 and the `check_to_bit_msg` to 42 for all nodes in column 5, we do the following:

    // for(auto e: pcm->iterate_column(5)){
    //     //here e is a pointer to the node
    //     e->bit_to_check_msg = 4;
    //     e->check_to_bit_msg = 42;
    // }

    // //It is also possible to iterate over the columns:

    // for(auto e: pcm->iterate_row(5)){
    //     e->bit_to_check_msg = 150;
    // }


    // //gf2 operations
    // //Matrix multiplication
    // //The `gf2sparse<T>.mulvec(vector<uint8_t>& input_vector, vector<uint8_t>& output_vector)` can be used for
    // // matrix vector multiplication. eg:

    // vector<uint8_t> error; //defines error vector
    // error.resize(pcm->n,0); //fill error vector with zeros. `pcm->n` returns the number of columns in the pcm.
    // error[4] = 1;

    // vector<uint8_t> syndrome; //defines syndrome vector
    // syndrome.resize(pcm->m,0); //fill error vector with zeros. `pcm->n` returns the number of rows in the pcm.
    // syndrome = pcm->mulvec(error, syndrome);


    // cout<<"Error: ";
    // print_vector(error);
    // cout<<"Syndrome: ";
    // print_vector(syndrome);

    // auto coords = pcm->nonzero_coordinates();

    // for(auto x: coords){
    //     cout<<x[0]<<","<<x[1]<<endl;
    // }


    // auto matrix = new gf2sparse(4,7);


    // matrix->insert_entry(0, 0, 1);
    // matrix->insert_entry(2,1,1);
    // matrix->insert_entry(3,1,1);
    // matrix->insert_entry(0, 1, 1);
    // matrix->insert_entry(0, 4, 1);

    // matrix->insert_entry(1, 4, 1);

    // matrix->insert_entry(2, 2, 1);
    // matrix->insert_entry(2, 5, 1);

    // matrix->insert_entry(3, 3, 1);
    // matrix->insert_entry(3, 6, 1);

    // print_sparse_matrix(*matrix);


    // // create a new gf2sparse matrix object
    // auto matrix = new gf2sparse(4, 7);

    // // insert entries into the matrix, reversing the order of the columns
    // matrix->insert_entry(0, 6, 1);
    // matrix->insert_entry(0, 5, 1);
    // matrix->insert_entry(0, 4, 1);
    // matrix->insert_entry(0, 2, 1);
    // matrix->insert_entry(0, 1, 1);

    // matrix->insert_entry(1, 6, 1);
    // matrix->insert_entry(1, 4, 1);
    // matrix->insert_entry(1, 3, 1);
    // matrix->insert_entry(1, 1, 1);
    // matrix->insert_entry(1, 0, 1);

    // matrix->insert_entry(2, 5, 1);
    // matrix->insert_entry(2, 4, 1);
    // matrix->insert_entry(2, 3, 1);
    // matrix->insert_entry(2, 0, 1);

    // matrix->insert_entry(3, 6, 1);
    // matrix->insert_entry(3, 3, 1);
    // matrix->insert_entry(3, 2, 1);
    // matrix->insert_entry(3, 1, 1);

    // print_sparse_matrix(*matrix);

    // // exit(22);


    // matrix->row_reduce(true,false);


    // // print_sparse_matrix(*matrix);

    // cout<<endl;

    // print_sparse_matrix(*matrix->L);

    // cout<<endl;

    // auto output = matrix->L->matmul(matrix);

    // print_sparse_matrix(*output);

    // cout<<endl;


    // matrix->display_U();
    // // matrix->U->reorder_rows(matrix->rows);

    // // print_sparse_matrix(*matrix->U);

    // print_vector(matrix->rows);
    // print_vector(matrix->cols);


    // cout<<endl;

    // print_sparse_matrix(*matrix);
    // matrix->add_rows(1,0);
    // cout<<endl;
    // print_sparse_matrix(*matrix);


    auto matrix = GF2Sparse(1,1000);
    matrix.insert_entry(0,0);
    matrix.add_rows(0,0);
    matrix.insert_entry(0,0);
    matrix.insert_entry(0,0);
    print_sparse_matrix(matrix);


   

    // delete pcm;




    


    // long long volume = doc.GetCell<long long>(4, 2);
    // std::cout << "Volume " << volume << " on 2017-02-22." << std::endl;

    return 0;


}