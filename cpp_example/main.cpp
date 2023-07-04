#include <iostream>
#include "sparse_matrix.hpp"
#include "sparse_matrix_util.hpp"
// #include "ssf.hpp"


using namespace std;


int main()
{
    
    sparse_matrix::SparseMatrix<int> mat;
    mat.allocate(3,3,3);

    cout<<mat.n<<endl;

    // mat.insert_entry(0,0,1);
    // print_sparse_matrix(mat);

    return 0;


}