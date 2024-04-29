#include <gtest/gtest.h>
#include "bp.hpp"
#include "ldpc.hpp"
#include "osd.hpp"
#include "gf2codes.hpp"
#include "bp_k.hpp"
#include "sparse_matrix_util.hpp"

using namespace std;

TEST(Kruskal, rep_code){
    {
        auto rep_code = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(3);
        auto bit_weights = std::vector<int>{0,1,2};
        auto stb = ldpc::bpk::find_weighted_spanning_tree(rep_code, bit_weights);
        // ldpc::sparse_matrix_util::print_vector(stb);
        auto expected = std::vector<int>{0,1};
        ASSERT_EQ(stb, expected);
    }

    {
        auto rep_code = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(3);
        auto bit_weights = std::vector<int>{1,2,0};
        auto stb = ldpc::bpk::find_weighted_spanning_tree(rep_code, bit_weights);
        // ldpc::sparse_matrix_util::print_vector(stb);
        auto expected = std::vector<int>{1,2};
        ASSERT_EQ(stb, expected);
    }


}

// TEST(OsdDecoder, rep_code_test1){

//     auto rep_code = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(3);
//     auto error = vector<uint8_t>{0, 0, 1};
//     auto syndrome = rep_code.mulvec(error);
//     auto error_channel = vector<double>{0.1,0.1,0.1};
//     auto lbr = vector<double>{0, 0, 0};
//     auto decoder = ldpc::osd::OsdDecoder(rep_code,ldpc::osd::OSD_0,0,error_channel);
//     auto decoding = decoder.decode(syndrome,lbr);
//     auto syndrome2 = rep_code.mulvec(decoding);
//     ASSERT_TRUE(syndrome2 == syndrome);

// }



int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}