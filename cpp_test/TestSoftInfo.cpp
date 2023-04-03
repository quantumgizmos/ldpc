#include <gtest/gtest.h>
#include "bp.hpp"
#include "gf2sparse.hpp"
#include "gf2sparse_linalg.hpp"
#include "sparse_matrix_util.hpp"
#include "osd.hpp"

TEST(OsdDecoder, above_cutoff_same_as_ms) {

    /*The soft into decoder should give the same result
    as min-sum BP if the syndrome is below the cutoff.*/


    //Setup repetition code
    int N = 4;
    auto pcm = bp::BpSparse::New(N-1, N);
    for(int i = 0; i<N-1; i++){
        pcm->insert_entry(i, i);
        pcm->insert_entry(i,i+1);
    }

    vector<double> error_channel(pcm->n,0.1);
    //could we come up with a sum product formulation for this?
    auto bpd = new bp::BpDecoder(pcm,error_channel,N,1,1.0,1);

    for (int j = 0; j<(N-1); j++){
        

        vector<double> soft_syndrome(pcm->m,2);
        soft_syndrome[j]*=-1;
        vector<uint8_t> hard_syndrome(pcm->m,0);
        hard_syndrome[j] = 1;


        // print_vector(soft_syndrome);
        // print_sparse_matrix(*pcm);




        vector<uint8_t> normal_decoding;
        bpd->decode(hard_syndrome);
        for(auto bit: bpd->decoding) normal_decoding.push_back(bit);

        double cutoff = 1;
        vector<uint8_t> soft_decoding;
        bpd->soft_info_decode_serial(soft_syndrome,cutoff);
        // bpd->decode(hard_syndrome);

        for(auto bit: bpd->decoding) soft_decoding.push_back(bit);

        // cout<<"Decoding: "<<j<<endl;
        // print_vector(normal_decoding);
        // print_vector(soft_decoding);

        for(int i = 0; i<pcm->n; i++){
            ASSERT_EQ(normal_decoding[i],soft_decoding[i]);
        }

    }

}


TEST(OsdDecoder, errored_syndrome) {

    /*In this test, I will attempt to decode a 3 qubit ring code with an
    errored 1-qubit syndrome. The errored qubit is assigned as soft-syndrome
    magnitude below the cutoff.*/


    //Setup repetition code
    int N = 3;
    auto pcm = bp::BpSparse::New(N, N);
    for(int i = 0; i<N; i++){
        pcm->insert_entry(i, i);
        pcm->insert_entry(i,(i+1)%N);
    }

    print_sparse_matrix(*pcm);

    vector<double> error_channel(pcm->n,0.1);
    //could we come up with a sum product formulation for this?
    auto bpd = new bp::BpDecoder(pcm,error_channel,N,1,1.0,1);

    vector<double> soft_syndrome(pcm->m,2);
    soft_syndrome[0]=0.01;
    soft_syndrome[1]=0.01; //syndrome is incorrect, but only just

    double cutoff = 0.1;
    vector<uint8_t> soft_decoding;
    bpd->soft_info_decode_serial(soft_syndrome,cutoff);
    for(auto bit: bpd->decoding) soft_decoding.push_back(bit);

    cout<<"Decoding: "<<endl;
    print_vector(soft_decoding);

    // for(int i = 0; i<pcm->n; i++){
    //     ASSERT_EQ(normal_decoding[i],soft_decoding[i]);
    // }

    

}




int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}