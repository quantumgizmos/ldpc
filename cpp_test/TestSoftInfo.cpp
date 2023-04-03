#include <gtest/gtest.h>
#include "bp.hpp"
#include "gf2sparse.hpp"
#include "gf2sparse_linalg.hpp"
#include "sparse_matrix_util.hpp"
#include "osd.hpp"

TEST(OsdDecoder, above_cutoff_same_as_ms) {

    //Setup repetition code
    int N = 4;
    auto pcm = bp::BpSparse::New(N-1, N);
    for(int i = 0; i<N-1; i++){
        pcm->insert_entry(i, i);
        pcm->insert_entry(i,i+1);
    }

    vector<double> soft_syndrome(pcm->m,2);
    soft_syndrome[0]*=-1;
    vector<uint8_t> hard_syndrome(pcm->m,0);
    hard_syndrome[0] = 1;
    vector<double> error_channel(pcm->n,0.1);

    // print_vector(soft_syndrome);
    // print_sparse_matrix(*pcm);


    //could we come up with a sum product formulation for this?
    auto bpd = new bp::BpDecoder(pcm,error_channel,N,1,1.0);

    auto hard_decoding = bpd->decode(hard_syndrome);

    double cutoff = 1;
    auto soft_decoding = bpd->soft_info_decode_serial(soft_syndrome,cutoff);

    for(int i = 0; i<pcm->n; i++){
        ASSERT_EQ(hard_decoding[i],soft_decoding[i]);
    }




    

}




int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}