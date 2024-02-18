#include <gtest/gtest.h>
#include "ldpc.hpp"
#include "bp.hpp"
#include "gf2codes.hpp"
// #include "gd.hpp"

using namespace std;


TEST(BpGdDecoder, BpGdInit){
    
    auto pcm = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(6);
    auto error = std::vector<uint8_t>(6,0);
    error[0] = 1;
    error[1] = 1;
    error[2] = 1;
    // error[3] = 1;

    auto syndrome = pcm.mulvec(error);
    auto error_channel = std::vector<double>(6,0.1);

// Define optional input parameters
    ldpc::bp::BpMethod bp_method = ldpc::bp::MINIMUM_SUM;
    double min_sum_scaling_factor = 0.625;
    ldpc::bp::BpSchedule bp_schedule = ldpc::bp::PARALLEL;

    // Initialize decoder using input arguments and optional parameters
    auto decoder = ldpc::bp::BpDecoder(pcm, error_channel, 6, bp_method, bp_schedule, min_sum_scaling_factor);
    
    decoder.bp_decode_parallel(syndrome);
    ASSERT_FALSE(decoder.converge);

    decoder.guided_decimatation_decode(syndrome,6);
    ASSERT_TRUE(decoder.converge);


}


int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}