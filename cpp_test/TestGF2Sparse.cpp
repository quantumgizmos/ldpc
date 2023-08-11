#include <gtest/gtest.h>
#include "gf2sparse.hpp"
#include "sparse_matrix_util.hpp"
#include "rapidcsv.h"
#include <iostream>
#include <set>
#include "io.hpp"
#include <string>
#include "sparse_matrix.hpp"
#include "gf2codes.hpp"

using namespace gf2sparse;

bool TEST_WITH_CSR(GF2Sparse<GF2Entry>& matrix, vector<vector<int>> csr_matrix){

    int i = 0;
    for(vector<int> row: csr_matrix){
        int j = 0;
        set<int> row_set;
        for(auto col : row) row_set.insert(col);
        for(auto e: matrix.iterate_row_ptr(i)){

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
    // print_sparse_matrix(matrix);
}

TEST(GF2Sparse, csr_insert){

    auto matrix = GF2Sparse(3,3);
    vector<vector<int>> csr_mat = {{0},{1,2},{2}};
    matrix.csr_insert(csr_mat);
    // print_sparse_matrix(matrix);

    ASSERT_EQ(matrix.get_entry(0,0).at_end(), false);
    ASSERT_EQ(matrix.get_entry(1,1).at_end(), false);
    ASSERT_EQ(matrix.get_entry(1,2).at_end(), false);
    ASSERT_EQ(matrix.get_entry(2,2).at_end(), false);
    ASSERT_EQ(matrix.get_entry(2,0).at_end(), true);

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

    ASSERT_EQ(matrix.get_entry(0,0).at_end(), true);
    ASSERT_EQ(matrix.get_entry(1,1).at_end(), true);
    ASSERT_EQ(matrix.get_entry(1,2).at_end(), true);
    ASSERT_EQ(matrix.get_entry(2,2).at_end(), true);
    ASSERT_EQ(matrix.get_entry(2,0).at_end(), true);

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

    ASSERT_EQ(matrix.get_entry(0,0).at_end(), false);
    ASSERT_EQ(matrix.get_entry(1,1).at_end(), false);
    ASSERT_EQ(matrix.get_entry(1,2).at_end(), false);
    ASSERT_EQ(matrix.get_entry(2,2).at_end(), false);
    ASSERT_EQ(matrix.get_entry(1,0).at_end(), false);


    vector<vector<int>> csr_mat2 = {{0},{0,1,2},{2}};
    ASSERT_EQ(TEST_WITH_CSR(matrix,csr_mat2),true);

    matrix.add_rows(1,0);
    ASSERT_EQ(TEST_WITH_CSR(matrix,csr_mat),true);
    auto p2 = print_sparse_matrix(matrix,true);
    ASSERT_EQ(p1.str(),p2.str());

}

TEST(GF2Sparse, string_io){

    auto matrix = GF2Sparse(3,3);
    string csr_string = " [[0],[1, 2], []] ";
    // cout<<csr_string<<endl;
    vector<vector<int>> csr_input = io::string_to_csr_vector(csr_string);
    
    matrix.csr_insert(csr_input);
    ASSERT_EQ(matrix.get_entry(0,0).at_end(), false);
    ASSERT_EQ(matrix.get_entry(1,1).at_end(), false);
    ASSERT_EQ(matrix.get_entry(1,2).at_end(), false);
    ASSERT_EQ(matrix.get_entry(2,2).at_end(), true);
    ASSERT_EQ(matrix.get_entry(2,0).at_end(), true);

    vector<vector<int>> non_zero_row_coords = matrix.nonzero_rows();
    // for(auto row: non_zero_row_coords){
    //     print_vector(row);
    // }
    auto matrix2=GF2Sparse(3,3);
    matrix2.csr_insert(non_zero_row_coords);
    ASSERT_EQ(TEST_WITH_CSR(matrix2,csr_input),true);

    auto test_io = io::csr_vector_to_string(non_zero_row_coords);
    // cout<<test_io<<endl;
    auto test_io2 = io::string_to_csr_vector(test_io);
    auto matrix3 = GF2Sparse(3,3);
    matrix3.csr_insert(test_io2);
    ASSERT_EQ(TEST_WITH_CSR(matrix3,csr_input),true);



}

TEST(GF2Sparse, string_io2){

    auto matrix = GF2Sparse(1,1);

    string csr_string = "[[0]]";
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



TEST(GF2Sparse, string_io3){

    auto matrix = GF2Sparse(26,25);

    string csr_string = "[[5,8],[],[13,20],[10,17],[2],[9,12],[9,17],[],[7,20,24],[],[],[],[],[],[19,23,24],[10,14],[13,20],[16],[],[3,5],[],[12],[],[20],[18],[16,24]]";
    vector<vector<int>> csr_input = io::string_to_csr_vector(csr_string);
    vector<vector<int>> csr_test = {{5,8},{},{13,20},{10,17},{2},{9,12},{9,17},{},{7,20,24},{},{},{},{},{},{19,23,24},{10,14},{13,20},{16},{},{3,5},{},{12},{},{20},{18},{16,24}};

    ASSERT_EQ(matrix.m,26);
    ASSERT_EQ(matrix.n,25);
    ASSERT_EQ(csr_input.size(),matrix.m);
    matrix.csr_insert(csr_input);
    // // cout<<matrix.entry_count()<<endl;
    // print_sparse_matrix(matrix);
    ASSERT_EQ(TEST_WITH_CSR(matrix,csr_test),true);
    ASSERT_EQ(TEST_WITH_CSR(matrix,csr_input),true);

    vector<vector<int>> non_zero_row_coords = matrix.nonzero_rows();
    // for(auto row: non_zero_row_coords){
    //     print_vector(row);
    // }
    auto matrix2=GF2Sparse(26,25);
    matrix2.csr_insert(non_zero_row_coords);
    ASSERT_EQ(TEST_WITH_CSR(matrix2,csr_input),true);

    auto test_io = io::csr_vector_to_string(non_zero_row_coords);
    // cout<<test_io<<endl;
    auto test_io2 = io::string_to_csr_vector(test_io);
    auto matrix3 = GF2Sparse(26,25);
    matrix3.csr_insert(test_io2);
    ASSERT_EQ(TEST_WITH_CSR(matrix3,csr_input),true);

    ASSERT_EQ(1,1);


}


TEST(GF2Sparse, string_io4){

    auto matrix = GF2Sparse(26,25);

    string csr_string = "[[5,8],[],[13,20],[10,17],[2],[9,12],[9,17],[],[7,20,24],[],[],[],[],[],[19,23,24],[10,14],[13,20],[16],[],[3,5],[],[12],[],[20],[18],[16,24]]";
    vector<vector<int>> csr_input = io::string_to_csr_vector(csr_string);
    vector<vector<int>> csr_test = {{5,8},{},{13,20},{10,17},{2},{9,12},{9,17},{},{7,20,24},{},{},{},{},{},{19,23,24},{10,14},{13,20},{16},{},{3,5},{},{12},{},{20},{18},{16,24}};

    ASSERT_EQ(matrix.m,26);
    ASSERT_EQ(matrix.n,25);
    ASSERT_EQ(csr_input.size(),matrix.m);
    matrix.csr_insert(csr_input);
    // // cout<<matrix.entry_count()<<endl;
    // print_sparse_matrix(matrix);
    ASSERT_EQ(TEST_WITH_CSR(matrix,csr_test),true);
    ASSERT_EQ(TEST_WITH_CSR(matrix,csr_input),true);





    ASSERT_EQ(1,1);


}

TEST(GF2Sparse, gf2_equal1){

    auto mat1 = GF2Sparse(4,3);
    auto mat2 = GF2Sparse(3,4);
    ASSERT_EQ(mat1.gf2_equal(mat2),false);
    ASSERT_EQ(mat1==mat2,false);

}

TEST(GF2Sparse, gf2_equal2){

    auto mat1 = GF2Sparse(3,3);
    auto mat2 = GF2Sparse(3,3);
    ASSERT_EQ(mat1.gf2_equal(mat2),true);
    ASSERT_EQ(mat1==mat2,true);

    for(int i = 0; i<3; i++){
        mat1.insert_entry(i,i);
        mat2.insert_entry(i,i);
    }

    ASSERT_EQ(mat1.gf2_equal(mat2),true);
    ASSERT_EQ(mat1==mat2,true);
    mat2.insert_entry(1,0);
    ASSERT_EQ(mat1.gf2_equal(mat2),false);
    ASSERT_EQ(mat1==mat2,false);
    mat1.insert_entry(1,0);
    ASSERT_EQ(mat1.gf2_equal(mat2),true);
    ASSERT_EQ(mat1==mat2,true);

}


TEST(GF2Sparse, add_rows_batch){
    auto csv_path = io::getFullPath("cpp_test/test_inputs/gf2_add_test.csv");
    rapidcsv::Document doc(csv_path, rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(';'));


    int row_count = doc.GetColumn<string>(0).size();

    for(int i = 1; i<row_count; i++){

        std::vector<string> row = doc.GetRow<string>(i);

        int m = stoi(row[0]);
        int n = stoi(row[1]);
        auto input_csr_vector = io::string_to_csr_vector(row[2]);
        auto target_row = stoi(row[3]);
        auto add_row = stoi(row[4]);
        auto output_csr_vector = io::string_to_csr_vector(row[5]);

        ASSERT_EQ(input_csr_vector.size(),m);
        ASSERT_EQ(output_csr_vector.size(),m);
        auto matrix = GF2Sparse(m,n);

        matrix.csr_insert(input_csr_vector);

        ASSERT_EQ(TEST_WITH_CSR(matrix,input_csr_vector),true);

        int node_count_initial = matrix.entry_count();

        matrix.add_rows(target_row,add_row);

        ASSERT_EQ(TEST_WITH_CSR(matrix,output_csr_vector),true);

        // //Adding the rows again should reverse the initial operation.
        if(target_row!=add_row){
            matrix.add_rows(target_row,add_row);
            ASSERT_EQ(TEST_WITH_CSR(matrix,input_csr_vector),true);
            int node_count_final = matrix.entry_count();
            ASSERT_EQ(node_count_initial,node_count_final);
        }
    }

}

TEST(GF2Sparse, mulvec_batch){

    auto csv_path = io::getFullPath("cpp_test/test_inputs/gf2_mulvec_test.csv");
    rapidcsv::Document doc(csv_path, rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(';'));


    int row_count = doc.GetColumn<string>(0).size();

    for(int i = 1; i<row_count; i++){

        std::vector<string> row = doc.GetRow<string>(i);

        int m = stoi(row[0]);
        int n = stoi(row[1]);
        auto input_csr_vector = io::string_to_csr_vector(row[2]);
        auto input_vector = io::binaryStringToVector(row[3]);
        auto actual_output_vector = io::binaryStringToVector(row[4]);

        ASSERT_EQ(input_vector.size(),n);
        ASSERT_EQ(actual_output_vector.size(),m);

        auto matrix = GF2Sparse(m,n);
        matrix.csr_insert(input_csr_vector);

        vector<uint8_t> output_vector;
        output_vector.resize(m);
        matrix.mulvec(input_vector,output_vector);

        bool equal = true;
        for(int j = 0; j<m; j++){
            if(output_vector[j]!=actual_output_vector[j]){
                equal = false;
                break;
            }
        }

        ASSERT_EQ(equal, true);

    }

}

// TEST(DISABLED_GF2Sparse,mulvec_timing){
    
//     //Make sure to run this test in release mode.

//     // cout<<"Hello"<<endl;

//     auto matrix = GF2Sparse(100,100);
//     for(int i = 0; i<100;i++) matrix.insert_entry(i,i);
//     vector<uint8_t> input_vector;
//     vector<uint8_t> output_vector;
//     input_vector.resize(matrix.n,0);
//     output_vector.resize(matrix.m,0);

//     const auto start_time = std::chrono::high_resolution_clock::now();

//     for(int i = 0; i<100000; i++){
//         input_vector[2]^=1;
//         matrix.mulvec(input_vector,output_vector);
//     }

//     auto end_time = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

//     cout<<"Duration orig: "<<duration<<endl;


//     input_vector.resize(matrix.n,0);


//     const auto start_time2 = std::chrono::high_resolution_clock::now(); 

//     for(int i = 0; i<1000000; i++){
//         input_vector[2]^=1;
//         auto output = matrix.mulvec2(input_vector);
//     }

//     const auto end_time2 = std::chrono::high_resolution_clock::now();
//     auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time2 - start_time2).count();

//     cout<<"Duration new: "<<duration2<<endl;

//     SUCCEED();


// }


// TEST(GF2Sparse, matmul){
//     auto csv_path = io::getFullPath("cpp_test/test_inputs/gf2_matmul_test.csv");
//     rapidcsv::Document doc(csv_path, rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(';'));

//     class EntryTest: public sparse_matrix_base::EntryBase<EntryTest>{
//         public:
//             double extra_variable;
//             uint8_t value;
//     };


//     int row_count = doc.GetColumn<string>(0).size();

//     for(int i = 0; i<row_count; i++){

//         std::vector<string> row = doc.GetRow<string>(i);

//         int m1 = stoi(row[0]);
//         int n1 = stoi(row[1]);
//         auto s1 = io::string_to_csr_vector(row[2]);
//         auto matrix1 = GF2Sparse<>::New(m1,n1);
//         matrix1->csr_insert(s1);

//         // print_sparse_matrix(matrix1);

//         int m2 = stoi(row[3]);
//         int n2 = stoi(row[4]);
//         auto s2 = io::string_to_csr_vector(row[5]);
//         auto matrix2 = GF2Sparse<EntryTest>::New(m2,n2);
//         matrix2->csr_insert(s2);

//         // cout<<endl;
//         // print_sparse_matrix(matrix2);


//         int m3 = stoi(row[6]);
//         int n3 = stoi(row[7]);
//         auto s3 = io::string_to_csr_vector(row[8]);
//         auto matrix3 = GF2Sparse<>::New(m3,n3);
//         matrix3->csr_insert(s3);

//         auto actual_matrix3 = matrix1->matmul(matrix2);

//         ASSERT_EQ(print_sparse_matrix(*matrix3,true).str(), print_sparse_matrix(*actual_matrix3,true).str());
//         ASSERT_EQ(TEST_WITH_CSR(*actual_matrix3,s3),true);



//     }


// }


// TEST(GF2Sparse, matmul_stack){
//     auto csv_path = io::getFullPath("cpp_test/test_inputs/gf2_matmul_test.csv");
//     rapidcsv::Document doc(csv_path, rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(';'));

//     class EntryTest: public sparse_matrix_base::EntryBase<EntryTest>{
//         public:
//             double extra_variable;
//             uint8_t value;
//     };


//     int row_count = doc.GetColumn<string>(0).size();

//     for(int i = 0; i<row_count; i++){

//         std::vector<string> row = doc.GetRow<string>(i);

//         int m1 = stoi(row[0]);
//         int n1 = stoi(row[1]);
//         auto s1 = io::string_to_csr_vector(row[2]);
//         auto matrix1 = GF2Sparse<>(m1,n1);
//         matrix1.csr_insert(s1);

//         // print_sparse_matrix(matrix1);

//         int m2 = stoi(row[3]);
//         int n2 = stoi(row[4]);
//         auto s2 = io::string_to_csr_vector(row[5]);
//         auto matrix2 = GF2Sparse<EntryTest>(m2,n2);
//         matrix2.csr_insert(s2);

//         // cout<<endl;
//         // print_sparse_matrix(matrix2);


//         int m3 = stoi(row[6]);
//         int n3 = stoi(row[7]);
//         auto s3 = io::string_to_csr_vector(row[8]);
//         auto matrix3 = GF2Sparse<>(m3,n3);
//         matrix3.csr_insert(s3);

//         auto actual_matrix3 = matrix1.matmul(matrix2);

//         ASSERT_EQ(print_sparse_matrix(matrix3,true).str(), print_sparse_matrix(actual_matrix3,true).str());
//         ASSERT_EQ(TEST_WITH_CSR(actual_matrix3,s3),true);

//     }


// }


// TEST(GF2Sparse,hstack){
//     auto m1 = gf2codes::hamming_code(3);
//     auto m2 = gf2codes::hamming_code(3);

//     auto mats = vector<decltype(m1)>{m1,m2};

//     auto m3 = gf2sparse::hstack(mats);

//     ASSERT_EQ(m3->m,3);
//     ASSERT_EQ(m3->n,14);

//     // print_sparse_matrix(*m3);
// }

// TEST(GF2Sparse,vstack){
//     auto m1 = gf2codes::hamming_code(3);
//     auto m2 = gf2codes::hamming_code(3);

//     auto mats = vector<decltype(m1)>{m1,m2};

//     auto m3 = gf2sparse::vstack(mats);

//     ASSERT_EQ(m3->m,6);
//     ASSERT_EQ(m3->n,7);

//     // print_sparse_matrix(*m3);
// }

// TEST(GF2Sparse, kron){
//     auto m1 = gf2sparse::gf2_identity<bp::BpEntry>(100);
//     auto m2 = gf2codes::hamming_code(5);

//     auto m3 = gf2sparse::kron(m1,m2);

//     ASSERT_EQ(m3->m,m1->m*m2->m);
//     ASSERT_EQ(m3->n,m1->n*m2->n);

//     // print_sparse_matrix(*m3);

// }


int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}