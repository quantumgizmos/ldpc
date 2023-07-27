#include <iostream>
#include "sparse_matrix.hpp"
#include "sparse_matrix_util.hpp"
#include <numeric> // Include the <numeric> header for std::accumulate
// #include "ssf.hpp"


using namespace std;


int main()
{
    
    sparse_matrix::SparseMatrix<int> mat;
    mat.allocate(3,3,3);

    // cout<<mat.n<<endl;

    mat.insert_entry(0,0,1);
    mat.insert_entry(0,1,2);
    mat.insert_entry(0,2,3);
    // print_sparse_matrix(mat);


    for(auto e: mat.iterate_row(0))
    {
        cout<<e.value<<endl;
        // auto ptr = &e;
        cout<<&e<<endl;

        e.value = 42;
    }

    cout<<endl;


    for(auto& e: mat.iterate_row(0))
    {
        cout<<e.value<<endl;
        cout<<&e<<endl;
        e.value = 42;
    }


    cout<<endl;

    for(auto e: mat.iterate_row_ptr(0)){
        cout<<e->value<<endl;
        cout<<e<<endl;
    }


    // int sum = accumulate(mat.iterate_row(0).begin(),mat.iterate_row(0).end(),0);

    // cout<<sum<<endl;

    return 0;


}