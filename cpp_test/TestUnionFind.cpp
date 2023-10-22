#include <gtest/gtest.h>
#include "udlr.hpp"
#include "gf2codes.hpp"
#include "union_find.hpp"
#include "util.hpp"

using namespace std;
using namespace uf;

TEST(UfDecoder, single_bit_error) {

    auto pcm1 = gf2codes::ring_code(10);
    // udlr::sparse_matrix_util::print_sparse_matrix(pcm1);

    auto ufd = UfDecoder(pcm1);

    for(int i=0; i<pcm1.n; i++){

        auto syndrome = vector<uint8_t>(pcm1.n,0);
        syndrome[i % pcm1.n] = 1;
        syndrome[(i+1) % pcm1.n] = 1;

        auto decoding = ufd.peel_decode(syndrome);

        auto expected_decoding = vector<uint8_t>(pcm1.n,0);
        expected_decoding[(i+1) % pcm1.n] = 1;

        ASSERT_EQ(decoding,expected_decoding);

    }

}


TEST(UfDecoder, weighted_cluster_growth) {

    auto pcm1 = gf2codes::ring_code(7);

    auto ufd = UfDecoder(pcm1);

    auto syndrome = vector<uint8_t>(pcm1.n,0);
    syndrome[0] = 1;
    syndrome[1] = 1;

    auto bit_weights = vector<double>{-1,100,-1,-1,-1,-1,-1};
    auto decoding = ufd.peel_decode(syndrome, bit_weights);
    auto expected_decoding = vector<uint8_t>{1,0,1,1,1,1,1};
    ASSERT_EQ(decoding,expected_decoding);
  

}

TEST(UfDecoder, HammingCode){

    int m = 5;

    auto pcm = gf2codes::hamming_code(m);

    auto ufd = UfDecoder(pcm);

    // auto syndrome = vector<uint8_t>(pcm.n,0);

    for(int i = 0; i < std::pow(2,m); i++){

        // udlr::sparse_matrix_util::print_vector(util::decimal_to_binary(i,m));

        auto syndrome = util::decimal_to_binary(i,m);
        auto decoding = ufd.matrix_decode(syndrome);
        auto decoding_syndrome = pcm.mulvec(decoding);
        ASSERT_EQ(decoding_syndrome,syndrome);

    }

}







int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}