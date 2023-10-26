#include <gtest/gtest.h>
#include "bp.hpp"
#include "ldpc.hpp"
#include "flip.hpp"

using namespace std;

TEST(TestFlipDecoder, single_bit_errors) {

    auto pcm = ldpc::bp::BpSparse(4, 8);
    pcm.insert_entry(0, 1), pcm.insert_entry(0, 2), pcm.insert_entry(0, 3), pcm.insert_entry(0, 4),
    pcm.insert_entry(1, 0), pcm.insert_entry(1, 2), pcm.insert_entry(1, 3), pcm.insert_entry(1, 5),
    pcm.insert_entry(2, 0), pcm.insert_entry(2, 1), pcm.insert_entry(2, 3), pcm.insert_entry(2, 6),
    pcm.insert_entry(3, 0), pcm.insert_entry(3, 1), pcm.insert_entry(3, 2), pcm.insert_entry(3, 7);

    ldpc::sparse_matrix_util::print_sparse_matrix(pcm);

    auto flipD = new ldpc::flip::FlipDecoder(pcm, 8);


    cout<<endl;

    for(int i = 2; i<8; i++){

        auto error = vector<uint8_t>(pcm.n,0);
        error[i] = 1;

        vector<uint8_t> syndrome(pcm.m, 0);
        pcm.mulvec(error,syndrome);

        auto decoding = flipD->decode(syndrome);

        cout<<"Error: ";
        ldpc::sparse_matrix_util::print_vector(error);
        cout<<"Syndrome: ";
        ldpc::sparse_matrix_util::print_vector(syndrome);
        cout<<"Decoding: ";
        ldpc::sparse_matrix_util::print_vector(decoding);
        cout<<"Decoding syndrome: ";
        vector<uint8_t> syndrome2(pcm.m, 0);
        pcm.mulvec(decoding,syndrome2);
        ldpc::sparse_matrix_util::print_vector(syndrome2);
        cout<<"Converged: "<<flipD->converge<<endl;
        cout<<"Iterations: "<<flipD->iterations<<endl;

        cout<<endl;

        for(int j = 0; j<pcm.m; j++){
            ASSERT_EQ(syndrome[j],syndrome2[j]);
        }
    
    }


}

TEST(TestFlipDecoder, two_bit_errors) {

    auto pcm = ldpc::bp::BpSparse(4, 8);
    pcm.insert_entry(0, 1), pcm.insert_entry(0, 2), pcm.insert_entry(0, 3), pcm.insert_entry(0, 4),
    pcm.insert_entry(1, 0), pcm.insert_entry(1, 2), pcm.insert_entry(1, 3), pcm.insert_entry(1, 5),
    pcm.insert_entry(2, 0), pcm.insert_entry(2, 1), pcm.insert_entry(2, 3), pcm.insert_entry(2, 6),
    pcm.insert_entry(3, 0), pcm.insert_entry(3, 1), pcm.insert_entry(3, 2), pcm.insert_entry(3, 7);

    ldpc::sparse_matrix_util::print_sparse_matrix(pcm);

    auto flipD = new ldpc::flip::FlipDecoder(pcm, 8);

    vector<vector<int>> error_locations;
    for(int i = 0; i<pcm.n; i++){
        for(int j = i+1; j<pcm.n; j++){
            error_locations.push_back({i,j});
        }
    }

    cout<<endl;

    for(auto error_indices: error_locations){

        auto error = vector<uint8_t>(pcm.n,0);
        for(int idx: error_indices){
            error[idx] = 1;
        }

        vector<uint8_t> syndrome(pcm.m, 0);
        pcm.mulvec(error,syndrome);

        auto decoding = flipD->decode(syndrome);

        cout<<"Error: ";
        ldpc::sparse_matrix_util::print_vector(error);
        cout<<"Syndrome: ";
        ldpc::sparse_matrix_util::print_vector(syndrome);
        cout<<"Decoding: ";
        ldpc::sparse_matrix_util::print_vector(decoding);
        cout<<"Decoding syndrome: ";
        vector<uint8_t> syndrome2(pcm.m, 0);
        pcm.mulvec(decoding,syndrome2);
        ldpc::sparse_matrix_util::print_vector(syndrome2);
        cout<<"Converged: "<<flipD->converge<<endl;
        cout<<"Iterations: "<<flipD->iterations<<endl;

        cout<<endl;

        for(int j = 0; j<pcm.m; j++){
            ASSERT_EQ(syndrome[j],syndrome2[j]);
        }
    
    }


}




int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}