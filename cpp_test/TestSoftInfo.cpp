#include <gtest/gtest.h>
#include "bp.hpp"
#include "ldpc.hpp"

#include "osd.hpp"

using namespace std;

TEST(SoftInfoDecoder, above_cutoff_same_as_ms) {

    /*The soft into decoder should give the same result
    as min-sum BP if the syndrome is below the cutoff.*/

    auto length = 3;
    //Setup ring code
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(length);
    int N = 3;
    auto pcm2 = ldpc::bp::BpSparse(length-1, length+2);
    // 2 rows, 5 cols
    pcm2.insert_entry(0, 0);
    pcm2.insert_entry(0, 1);
    pcm2.insert_entry(1, 1);
    pcm2.insert_entry(1, 2);
    pcm2.insert_entry(0, 3);
    pcm2.insert_entry(1, 4);


    vector<double> error_channel(pcm.n, 0.1);
    vector<double> error_channel2(pcm2.n, 0.1);
    error_channel2[pcm2.n-2] = -3.28474883;
    error_channel2[pcm2.n-1] =  6.56949766;

    //could we come up with a sum product formulation for this?
    auto bpd = ldpc::bp::BpDecoder(pcm, error_channel, 15);
    auto bpd2 = ldpc::bp::BpDecoder(pcm2, error_channel2, 15);

    vector<double> soft_syndrome(pcm.m, 2.0);
    soft_syndrome[0] = -1.0;
    double cutoff = 5;
    //print soft_syndrome vector
    std::cout << "soft_syndrome before: ";
    for(auto s: soft_syndrome){
        cout << static_cast<double>(s) << " ";
    }
    std::cout << std::endl;
    auto res = bpd.soft_info_decode_serial(soft_syndrome, cutoff, 0.780304146072379);
    // print res
    std::cout << "decoding: " << std::endl;
    for (auto bit: res) {
        cout << static_cast<int>(bit) << " ";
    }
    std::cout << std::endl;
    auto hs = std::vector<uint8_t>{1,0};
    auto res2 = bpd2.decode(hs);
    // print res
    std::cout << "ATD decoding: " << std::endl;
    for (auto bit2: res2) {
        cout << static_cast<int>(bit2) << " ";
    }
}


// TEST(SoftInfoDecoder, errored_close_to_zero) {

//     /*In this test, I will attempt to decode a 3 qubit ring code with an
//     errored zero syndrome. Ie. all syndromes are close to zero. I expect
//     the decoder to return the 000 codeword (corresponding to the 000 syndrome)*/


//     //Setup repetition code
//     int N = 3;
//     auto pcm = ldpc::bp::BpSparse::New(N, N);
//     for(int i = 0; i<N; i++){
//         pcm->insert_entry(i, i);
//         pcm->insert_entry(i,(i+1)%N);
//     }

//     // ldpc::sparse_matrix_util::print_sparse_matrix(*pcm);

//     vector<double> error_channel(pcm->n,0.1);
//     //could we come up with a sum product formulation for this?
//     auto bpd = new ldpc::bp::BpDecoder(pcm,error_channel,N,1,1.0,1);

//     vector<double> soft_syndrome(pcm->m,2);
//     soft_syndrome[0]=-1;
//     soft_syndrome[1]=1; //syndrome is incorrect, but only just
//     soft_syndrome[2]=-1;
//     double cutoff = 10;
//     vector<uint8_t> soft_decoding;
//     bpd->soft_info_decode_serial(soft_syndrome,cutoff);
//     for(auto bit: bpd->decoding) soft_decoding.push_back(bit);
//     for(int i = 0; i<pcm->n; i++){
//         ASSERT_EQ(soft_decoding[i],0);
//     }


// }


// TEST(SoftInfoDecoder, one_errored_syndrome_bit) {

//     /*In this test, I will attempt to decode a 3 qubit ring code with an
//     errored syndrome. The second syndrome is set to 5 (ie. no syndrome).
//     However, it is below the cutoff. I expect the decoder to flip the second syndrome
//     bit and return the decoding 100.*/


//     //Setup repetition code
//     int N = 3;
//     auto pcm = ldpc::bp::BpSparse::New(N, N);
//     for(int i = 0; i<N; i++){
//         pcm->insert_entry(i, i);
//         pcm->insert_entry(i,(i+1)%N);
//     }

//     // ldpc::sparse_matrix_util::print_sparse_matrix(*pcm);

//     vector<double> error_channel(pcm->n,0.1);
//     //could we come up with a sum product formulation for this?
//     auto bpd = new ldpc::bp::BpDecoder(pcm,error_channel,N,1,1.0,1);

//     vector<double> soft_syndrome = {-20,2,20};
//     double cutoff = 10;
//     auto soft_decoding =bpd->soft_info_decode_serial(soft_syndrome,cutoff,0.3);
//     vector<uint8_t> expected_decoding = {0,1,0};

//     // ldpc::sparse_matrix_util::print_vector(soft_decoding);

//     for(int i = 0; i<pcm->n; i++){
//         ASSERT_EQ(soft_decoding[i],expected_decoding[i]);
//     }


// }



// TEST(SoftInfoDecoder, long_rep_code) {

//     /*Same as the above test but with a longer repetition code.*/


//     //Setup repetition code
//     int N = 20;
//     auto pcm = ldpc::bp::BpSparse::New(N, N);
//     for(int i = 0; i<N; i++){
//         pcm->insert_entry(i, i);
//         pcm->insert_entry(i,(i+1)%N);
//     }

//     // ldpc::sparse_matrix_util::print_sparse_matrix(*pcm);

//     vector<double> error_channel(pcm->n,0.1);
//     //could we come up with a sum product formulation for this?
//     auto bpd = new ldpc::bp::BpDecoder(pcm,error_channel,N,1,1.0,1);

//     vector<double> soft_syndrome(pcm->m,100);
//     soft_syndrome[0]=-100;
//     soft_syndrome[7]=1;
//     double cutoff = 10;
//     auto soft_decoding =bpd->soft_info_decode_serial(soft_syndrome,cutoff);
//     vector<uint8_t> expected_decoding = {0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

//     // ldpc::sparse_matrix_util::print_vector(soft_decoding);

//     for(int i = 0; i<pcm->n; i++){
//         ASSERT_EQ(soft_decoding[i],expected_decoding[i]);
//     }


// }




int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}