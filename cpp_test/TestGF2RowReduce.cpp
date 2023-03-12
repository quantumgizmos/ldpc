#include <gtest/gtest.h>
#include "gf2sparse.hpp"
#include "sparse_matrix_util.hpp"
#include "rapidcsv.h"
#include <iostream>
#include <set>
#include "io.hpp"
#include <string>

using namespace gf2sparse;

TEST(RowReduce, init1){

    
    auto matrix = GF2Sparse(3,3);
    for(int i =0; i<3; i++) matrix.insert_entry(i,i);
    matrix.insert_entry(1,0);

    auto rr = RowReduce(&matrix);


}

TEST(RowReduce, destructor){

    auto matrix = new GF2Sparse(5,5);
    for(int i = 0; i<5; i++){
        matrix->insert_entry(i,i);
    }
    auto rr = new RowReduce(matrix);
    delete matrix;
    delete rr;

}

TEST(RowReduce, destructor2){

    auto matrix = new GF2Sparse(5,5);
    for(int i = 0; i<5; i++){
        matrix->insert_entry(i,i);
    }
    auto rr = new RowReduce(matrix);
    rr->set_column_row_orderings();
    rr->initiliase_LU();
    delete matrix;
    delete rr;

}

TEST(RowReduce, initiliase_LU){

    
    auto matrix = GF2Sparse(5,5);
    for(int i =0; i<5; i++) matrix.insert_entry(i,i);
    matrix.insert_entry(1,0);

    auto rr = RowReduce(&matrix);
    rr.initiliase_LU();
    auto U = rr.U;
    auto p1 = print_sparse_matrix(*U,true).str();
    auto p2 = print_sparse_matrix(matrix, true).str();
    ASSERT_EQ(p1,p2);

}

TEST(RowReduce, set_column_row_orderings){

    
    auto matrix = GF2Sparse(4,4);
    for(int i =0; i<4; i++) matrix.insert_entry(i,i);
    matrix.insert_entry(1,0);

    auto rr = RowReduce(&matrix);
    rr.initiliase_LU();
    auto U = rr.U;
    auto p1 = print_sparse_matrix(*U,true).str();
    auto p2 = print_sparse_matrix(matrix, true).str();
    ASSERT_EQ(p1,p2);

    vector<int> rows = {0,1,2,3};
    vector<int> cols = {0,1,2,3};

    rr.set_column_row_orderings();

    for(int i = 0; i<4; i++){
        ASSERT_EQ(rr.rows[i],rows[i]);
        ASSERT_EQ(rr.cols[i], cols[i]);
    }

}

TEST(RowReduce, set_column_row_orderings2){

    
    auto matrix = GF2Sparse(4,4);
    for(int i =0; i<4; i++) matrix.insert_entry(i,i);
    matrix.insert_entry(1,0);

    auto rr = RowReduce(&matrix);
    rr.initiliase_LU();
    auto U = rr.U;
    auto p1 = print_sparse_matrix(*U,true).str();
    auto p2 = print_sparse_matrix(matrix, true).str();
    ASSERT_EQ(p1,p2);

    vector<int> rows = {0,1,3,2};
    vector<int> cols = {1,0,3,2};

    rr.set_column_row_orderings(cols,rows);

    for(int i = 0; i<4; i++){
        ASSERT_EQ(rr.rows[i],rows[i]);
        ASSERT_EQ(rr.cols[i], cols[i]);
    }

}


TEST(RowReduce, rref){

    
    auto matrix = GF2Sparse(3,3);
    for(int i =0; i<3; i++) matrix.insert_entry(i,i);
    matrix.insert_entry(1,0);

    auto rr = RowReduce(&matrix);
    rr.rref();


}


int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}