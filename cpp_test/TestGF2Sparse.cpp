#include <gtest/gtest.h>
#include "gf2sparse.hpp"
#include "sparse_matrix_util.hpp"
#include "rapidcsv.h"
#include <iostream>
#include <set>
#include "io.hpp"
#include <string>

bool TEST_WITH_CSR(GF2Sparse<GF2Entry>& matrix, vector<vector<int>> csr_matrix){

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

TEST(GF2Sparse, init){
    auto matrix = GF2Sparse(100,100);
    matrix.add_rows(1,4);
    print_sparse_matrix(matrix);
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

TEST(GF2Sparse, csr_insert_empty){

    auto matrix = GF2Sparse(15,24);
    vector<vector<int>> csr_mat = {{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}};
    matrix.csr_insert(csr_mat);
    // print_sparse_matrix(matrix);

    ASSERT_EQ(matrix.get_entry(0,0)->value, 0);
    ASSERT_EQ(matrix.get_entry(1,1)->value, 0);
    ASSERT_EQ(matrix.get_entry(1,2)->value, 0);
    ASSERT_EQ(matrix.get_entry(2,2)->value, 0);
    ASSERT_EQ(matrix.get_entry(2,0)->value, 0);

    //test using the TEST_WITH_CSR function
    ASSERT_EQ(TEST_WITH_CSR(matrix,csr_mat),true);

    vector<vector<int>> csr_mat2 = {{},{},{},{},{},{100},{},{},{},{},{},{},{},{},{}};
    ASSERT_EQ(TEST_WITH_CSR(matrix,csr_mat2),false);

    //check the entry counts
    ASSERT_EQ(matrix.entry_count(),0);

}

TEST(GF2Sparse, add_rows){

    auto matrix = GF2Sparse(3,3);
    vector<vector<int>> csr_mat = {{0},{1,2},{2}};
    matrix.csr_insert(csr_mat);
    auto p1 = print_sparse_matrix(matrix,true);
    matrix.add_rows(1,0);

    ASSERT_EQ(matrix.get_entry(0,0)->value, 1);
    ASSERT_EQ(matrix.get_entry(1,1)->value, 1);
    ASSERT_EQ(matrix.get_entry(1,2)->value, 1);
    ASSERT_EQ(matrix.get_entry(2,2)->value, 1);
    ASSERT_EQ(matrix.get_entry(1,0)->value, 1);


    vector<vector<int>> csr_mat2 = {{0},{0,1,2},{2}};
    ASSERT_EQ(TEST_WITH_CSR(matrix,csr_mat2),true);

    matrix.add_rows(1,0);
    ASSERT_EQ(TEST_WITH_CSR(matrix,csr_mat),true);
    auto p2 = print_sparse_matrix(matrix,true);
    ASSERT_EQ(p1.str(),p2.str());

}

TEST(GF2Sparse, string_io){

    auto matrix = GF2Sparse(3,3);
    string csr_string = " {{0},{1, 2}, {}} ";
    // cout<<csr_string<<endl;
    vector<vector<int>> csr_input = io::string_to_csr_vector(csr_string);
    
    matrix.csr_insert(csr_input);
    ASSERT_EQ(matrix.get_entry(0,0)->value, 1);
    ASSERT_EQ(matrix.get_entry(1,1)->value, 1);
    ASSERT_EQ(matrix.get_entry(1,2)->value, 1);
    ASSERT_EQ(matrix.get_entry(2,2)->value, 0);
    ASSERT_EQ(matrix.get_entry(2,0)->value, 0);

}

TEST(GF2Sparse, string_io2){

    auto matrix = GF2Sparse(1,1);

    string csr_string = "{{0}}";
    // cout<<csr_string<<endl;
    vector<vector<int>> csr_input = io::string_to_csr_vector(csr_string);
    // for(auto a: csr_input) print_vector(a);
    vector<vector<int>> csr_test = {{0}};

    ASSERT_EQ(csr_input.size(),1);
    ASSERT_EQ(matrix.n,1);
    ASSERT_EQ(matrix.m,1);
    matrix.csr_insert(csr_input);
    ASSERT_EQ(TEST_WITH_CSR(matrix,csr_test),true);
    ASSERT_EQ(TEST_WITH_CSR(matrix,csr_input),true);

    matrix.add_rows(0,0);

    csr_test = {{}};
    ASSERT_EQ(TEST_WITH_CSR(matrix,csr_test),true);


    // print_sparse_matrix(matrix);


}

// TEST(GF2Sparse, string_io3){

//     auto matrix = GF2Sparse(26,25);

//     string csr_string = "{{5,8},{},{13,20},{10,17},{2},{9,12},{9,17},{},{7,20,24},{},{},{},{},{},{19,23,24},{10,14},{13,20},{16},{},{3,5},{},{12},{},{20},{18},{16,24}}";
//     // cout<<csr_string<<endl;
//     vector<vector<int>> csr_input = io::string_to_csr_vector(csr_string);
//     // for(auto a: csr_input) print_vector(a);
//     vector<vector<int>> csr_test = {{5,8},{},{13,20},{10,17},{2},{9,12},{9,17},{},{7,20,24},{},{},{},{},{},{19,23,24},{10,14},{13,20},{16},{},{3,5},{},{12},{},{20},{18},{16,24}};

//     ASSERT_EQ(matrix.m,26);
//     ASSERT_EQ(matrix.n,25);
//     ASSERT_EQ(csr_input.size(),matrix.m);
//     matrix.csr_insert(csr_input);
//     cout<<matrix.entry_count()<<endl;
//     // print_sparse_matrix(matrix);
//     // ASSERT_EQ(TEST_WITH_CSR(matrix,csr_test),true);
//     // ASSERT_EQ(TEST_WITH_CSR(matrix,csr_input),true);

//     matrix.add_rows(0,0);



//     // print_sparse_matrix(matrix);


// }



// TEST(GF2Sparse, add_rows_batch){
//     auto csv_path = io::getFullPath("cpp_test/test_inputs/gf2_add_test.csv");
//     rapidcsv::Document doc(csv_path, rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(';'));


//     int row_count = doc.GetColumn<string>(0).size();

//     for(int i = 1; i<2; i++){

//         cout<<i<<endl;
//         std::vector<string> row = doc.GetRow<string>(i);

//         int m = stoi(row[0]);
//         int n = stoi(row[1]);
//         auto input_csr_vector = io::string_to_csr_vector(row[2]);
//         auto target_row = stoi(row[3]);
//         auto add_row = stoi(row[4]);
//         auto output_csr_vector = io::string_to_csr_vector(row[5]);

//         ASSERT_EQ(input_csr_vector.size(),m);
//         ASSERT_EQ(output_csr_vector.size(),m);
//         auto matrix = GF2Sparse(m,n);

//         cout<<row[2]<<endl;
//         for(auto a: input_csr_vector) print_vector(a);

//         print_sparse_matrix(matrix);


//         matrix.csr_insert(input_csr_vector);

        

//         print_sparse_matrix(matrix);

//         ASSERT_EQ(TEST_WITH_CSR(matrix,input_csr_vector),true);

//         int node_count_initial = matrix.entry_count();

//         matrix.add_rows(target_row,add_row);

//         ASSERT_EQ(TEST_WITH_CSR(matrix,output_csr_vector),true);

//         // //Adding the rows again should reverse the initial operation.
//         if(target_row!=add_row){
//             matrix.add_rows(target_row,add_row);
//             ASSERT_EQ(TEST_WITH_CSR(matrix,input_csr_vector),true);
//             int node_count_final = matrix.entry_count();
//             ASSERT_EQ(node_count_initial,node_count_final);
//         }
//     }

// }

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}