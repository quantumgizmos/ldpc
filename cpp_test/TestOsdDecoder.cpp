#include <gtest/gtest.h>
#include "bp.hpp"
#include "gf2sparse.hpp"
#include "gf2sparse_linalg.hpp"
#include "sparse_matrix_util.hpp"
#include "osd.hpp"

TEST(OsdDecoder, PrintsCorrectly) {
    auto pcm = bp::BpSparse::New(4, 4);
    
    for(int i = 0; i<4; i++) pcm->insert_entry(i, i);
    pcm->insert_entry(1, 0);
    pcm->insert_entry(2, 0);

    auto error = vector<uint8_t>{0, 1, 1, 1};
    auto syndrome = vector<uint8_t>{0, 0, 0, 0};
    auto error_channel = vector<double>{0.1,0.1,0.1,0.1};
    pcm->mulvec(error, syndrome);
    auto lbr = vector<double>{0, 0, 0, 0};
    for(int i = 0; i<4; i++) lbr[i] = log((1-error_channel[i])/(error_channel[i]));
    auto osdD = new osd::OsdDecoder(pcm,0,0,error_channel);
    auto decoding = osdD->decode(syndrome, lbr);
    auto syndrome2 = vector<uint8_t>{0, 0, 0, 0};
    pcm->mulvec(decoding, syndrome2);
    for(int i = 0; i<4; i++) ASSERT_EQ(syndrome[i], syndrome2[i]);

    delete osdD;
    // osdD->decode()

}


TEST(OsdDecoder, AllZeroSyndrome) {
    auto pcm = bp::BpSparse::New(4, 4);
    for(int i = 0; i<4; i++) pcm->insert_entry(i, i);
    pcm->insert_entry(1, 0);
    pcm->insert_entry(2, 0);

    auto error = vector<uint8_t>{0, 1, 1, 1};
    auto syndrome = vector<uint8_t>{0, 0, 0, 0};
    auto error_channel = vector<double>{0.1,0.1,0.1,0.1};
    auto lbr = vector<double>{0, 0, 0, 0};
    for(int i = 0; i<4; i++) lbr[i] = log((1-error_channel[i])/(error_channel[i]));
    auto osdD = new osd::OsdDecoder(pcm,0,0,error_channel);
    auto decoding = osdD->decode(syndrome, lbr);
    for(int i = 0; i<4; i++) ASSERT_EQ(decoding[i], 0);
    delete osdD;
}


TEST(OsdDecoder, VariedErrorChannel) {
    auto pcm = bp::BpSparse::New(4, 4);
    for(int i = 0; i<4; i++) pcm->insert_entry(i, i);
    pcm->insert_entry(1, 0);
    pcm->insert_entry(2, 0);

    auto error = vector<uint8_t>{0, 1, 1, 1};
    auto syndrome = vector<uint8_t>{0, 0, 0, 0};
    auto error_channel = vector<double>{0.1,0.2,0.3,0.4};
    auto lbr = vector<double>{0, 0, 0, 0};
    for(int i = 0; i<4; i++) lbr[i] = log((1-error_channel[i])/(error_channel[i]));
    auto osdD = new osd::OsdDecoder(pcm,0,0,error_channel);
    auto decoding = osdD->decode(syndrome, lbr);
    auto syndrome2 = vector<uint8_t>{0, 0, 0, 0};
    pcm->mulvec(decoding, syndrome2);
    for(int i = 0; i<4; i++) ASSERT_EQ(syndrome[i], syndrome2[i]);
    delete osdD;

}

TEST(OsdDecoder, VariedErrorChannelLargerMatrix) {
    auto pcm = bp::BpSparse::New(3, 5);
    pcm->insert_entry(0, 0);
    pcm->insert_entry(0, 1);
    pcm->insert_entry(1, 0);
    pcm->insert_entry(1, 2);
    pcm->insert_entry(1, 3);
    pcm->insert_entry(2, 1);
    pcm->insert_entry(2, 2);
    pcm->insert_entry(2, 4);

    auto error = vector<uint8_t>{1, 0, 1, 0, 1};
    auto syndrome = vector<uint8_t>{0, 0, 0};
    auto error_channel = vector<double>{0.1,0.2,0.3,0.4,0.5};
    auto lbr = vector<double>{0, 0, 0, 0, 0};
    for(int i = 0; i<5; i++) lbr[i] = log((1-error_channel[i])/(error_channel[i]));
    auto osdD = new osd::OsdDecoder(pcm,0,0,error_channel);
    auto decoding = osdD->decode(syndrome, lbr);
    auto syndrome2 = vector<uint8_t>{0, 0, 0};
    pcm->mulvec(decoding, syndrome2);
    for(int i = 0; i<3; i++) ASSERT_EQ(syndrome[i], syndrome2[i]);


    delete osdD;
}



int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}