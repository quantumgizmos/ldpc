#include <gtest/gtest.h>
#include "gf2sparse.hpp"
#include "sparse_matrix_util.hpp"
#include "rapidcsv.h"
#include <iostream>
#include <set>
#include "io.hpp"
#include <string>

bool TEST_WITH_CSR(GF2Sparse<GF2Entry> matrix, vector<vector<int>>& csr_matrix){

    int i = 0;
    for(vector<int> row: csr_matrix){
        int j = 0;
        set<int> row_set;
        for(auto col : row) row_set.insert(col);
        for(auto e: matrix.iterate_row(i)){

            if(!e->value==1) return false;
            if(!e->row_index==i) return false;
            if(row_set.contains(e->col_index)){
                row_set.erase(e->col_index); //this ensures there a no duplicates.
            }
            else return false;

            j++;
        }
    
        if(row_set.size()!=0) return false; //this checks that there no entries in the CSR matrix that aren't in the matrix.
        i++;
    }

    return true; 

}


TEST(GF2Sparse, csr_insert){

    auto matrix = GF2Sparse(3,3);
    vector<vector<int>> csr_mat = {{0},{1,2},{2}};
    matrix.csr_insert(csr_mat);
    // print_sparse_matrix(matrix);

    ASSERT_EQ(matrix.get_entry(0,0)->value, 1);
    ASSERT_EQ(matrix.get_entry(1,1)->value, 1);
    ASSERT_EQ(matrix.get_entry(1,2)->value, 1);
    ASSERT_EQ(matrix.get_entry(2,2)->value, 1);
    ASSERT_EQ(matrix.get_entry(2,0)->value, 0);

    //test using the TEST_WITH_CSR function
    ASSERT_EQ(TEST_WITH_CSR(matrix,csr_mat),true);

    vector<vector<int>> csr_mat2 = {{0,4},{1,2},{2}};
    ASSERT_EQ(TEST_WITH_CSR(matrix,csr_mat2),false);

    //check the entry counts
    ASSERT_EQ(matrix.entry_count(),4);

}

TEST(GF2Sparse, add_rows){
    auto csv_path = io::getFullPath("cpp_test/test_inputs/gf2_add_test.csv");
    rapidcsv::Document doc(csv_path, rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(';'));


        int row_count = doc.GetColumn<string>(0).size();

    for(int i = 0; i<row_count; i++){

        std::vector<string> row = doc.GetRow<string>(i);


        int m = stoi(row[0]);
        int n = stoi(row[1]);
        auto input_csr_vector = io::string_to_csr_vector(row[2]);
        auto target_row = stoi(row[3]);
        auto add_row = stoi(row[4]);
        auto output_csr_vector = io::string_to_csr_vector(row[5]);

        auto matrix = GF2Sparse(m,n);
        matrix.csr_insert(input_csr_vector);

        int node_count_initial = matrix.entry_count();

        ASSERT_EQ(TEST_WITH_CSR(matrix,input_csr_vector),true);


        matrix.add_rows(target_row,add_row);

        ASSERT_EQ(TEST_WITH_CSR(matrix,output_csr_vector),true);
        // ASSERT_EQ(TEST_WITH_CSR(matrix,input_csr_vector),false);

        //Adding the rows again should reverse the initial operation.
        matrix.add_rows(target_row,add_row);
        ASSERT_EQ(TEST_WITH_CSR(matrix,input_csr_vector),true);

        int node_count_final = matrix.entry_count();

        ASSERT_EQ(node_count_initial,node_count_final);


    }
    
    // ASSERT_EQ(1,1);

}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}