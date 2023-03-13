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

    
    auto matrix = GF2Sparse<>::New(3,3);
    for(int i =0; i<3; i++) matrix->insert_entry(i,i);
    matrix->insert_entry(1,0);
    auto rr = RowReduce(matrix);


}

TEST(RowReduce, initiliase_LU){

    
    auto matrix = GF2Sparse<>::New(5,5);
    for(int i =0; i<5; i++) matrix->insert_entry(i,i);
    matrix->insert_entry(1,0);

    auto rr = RowReduce(matrix);
    rr.initiliase_LU();
    auto U = rr.U;
    auto p1 = print_sparse_matrix(*U,true).str();
    auto p2 = print_sparse_matrix(*matrix, true).str();
    ASSERT_EQ(p1,p2);

}

TEST(RowReduce, set_column_row_orderings){

    
    auto matrix = GF2Sparse<>::New(4,4);
    for(int i =0; i<4; i++) matrix->insert_entry(i,i);
    matrix->insert_entry(1,0);

    auto rr = RowReduce(matrix);
    rr.initiliase_LU();
    auto U = rr.U;
    auto p1 = print_sparse_matrix(*U,true).str();
    auto p2 = print_sparse_matrix(*matrix, true).str();
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

    
    auto matrix = GF2Sparse<>::New(4,4);
    for(int i =0; i<4; i++) matrix->insert_entry(i,i);
    matrix->insert_entry(1,0);

    auto rr = RowReduce(matrix);
    rr.initiliase_LU();
    auto U = rr.U;
    auto p1 = print_sparse_matrix(*U,true).str();
    auto p2 = print_sparse_matrix(*matrix, true).str();
    ASSERT_EQ(p1,p2);

    vector<int> rows = {0,1,3,2};
    vector<int> cols = {1,0,3,2};

    rr.set_column_row_orderings(cols,rows);

    for(int i = 0; i<4; i++){
        ASSERT_EQ(rr.rows[i],rows[i]);
        ASSERT_EQ(rr.cols[i], cols[i]);
    }

}


TEST(RowReduce, rref1){

    
    auto matrix = GF2Sparse<>::New(3,3);
    for(int i =0; i<3; i++) matrix->insert_entry(i,i);
    matrix->insert_entry(1,0);
    matrix->insert_entry(2,0);

    auto rr = RowReduce(matrix);
    auto U = rr.rref();
    auto B = rr.L->matmul(matrix);
    auto I = gf2_identity(3);
    ASSERT_EQ(B->gf2_equal(I),true);
    // print_sparse_matrix(*B);

}

TEST(RowReduce, rref2){

    auto matrix = gf2_identity(3);
    matrix->insert_entry(0,1);
    auto rr = RowReduce(matrix);
    auto U = rr.rref(true);
    auto B = rr.L->matmul(matrix);
    auto I = gf2_identity(3);
    ASSERT_EQ(B->gf2_equal(I),true);

}

TEST(GF2Sparse, rref_batch){

    auto csv_path = io::getFullPath("cpp_test/test_inputs/gf2_invert_test.csv");
    rapidcsv::Document doc(csv_path, rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(';'));


    int row_count = doc.GetColumn<string>(0).size();

    for(int i = 0; i<row_count; i++){

        std::vector<string> row = doc.GetRow<string>(i);

        int m = stoi(row[0]);
        int n = stoi(row[1]);
        auto input_csr_vector = io::string_to_csr_vector(row[2]);

        auto matrix = GF2Sparse<>::New(m,n);
        matrix->csr_insert(input_csr_vector);

        auto I = gf2_identity(n);

        auto rr = new RowReduce(matrix);
        rr->rref(true);
        auto B = rr->L->matmul(matrix);

        ASSERT_EQ(B->gf2_equal(I),true);


        }

    }

TEST(GF2Sparse, reverse_cols){

    auto mat1 = GF2Sparse<>::New(3,3);
    auto mat2 = GF2Sparse<>::New(3,3);

    for(int i = 0; i<3; i++){
        mat1->insert_entry(i,3-1-i);
    }

    auto rr = RowReduce(mat1);
    vector<int> cols = {2,1,0};
    rr.rref(false,false,cols);

    ASSERT_EQ(mat1->gf2_equal(rr.U),true);

}

TEST(GF2Sparse, lu1){

    auto mat1 = GF2Sparse<>::New(3,3);

    for(int i = 0; i<3; i++){
        mat1->insert_entry(i,3-1-i);
    }

    mat1->insert_entry(1,0);
    mat1->insert_entry(0,1);
    auto rr = RowReduce(mat1);
    rr.rref(false,true);
    auto LU = rr.L->matmul(rr.U);
    mat1->reorder_rows(rr.rows);
    ASSERT_EQ(mat1->gf2_equal(LU),true);

}




int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}