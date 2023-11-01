#include <gtest/gtest.h>
#include "bp.hpp"
#include "ldpc.hpp"
#include "osd.hpp"
#include "gf2codes.hpp"

using namespace std;

TEST(OsdDecoder, rep_code_test1){

    auto rep_code = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(3);
    auto error = vector<uint8_t>{0, 0, 1};
    auto syndrome = rep_code.mulvec(error);
    auto error_channel = vector<double>{0.1,0.1,0.1};
    auto lbr = vector<double>{0, 0, 0};
    auto decoder = ldpc::osd::OsdDecoder(rep_code,ldpc::osd::OSD_0,0,error_channel);
    auto decoding = decoder.decode(syndrome,lbr);
    auto syndrome2 = rep_code.mulvec(decoding);
    ASSERT_TRUE(syndrome2 == syndrome);

}

TEST(OsdDecoder, ring_code_test1){

    auto rep_code = ldpc::gf2codes::ring_code<ldpc::bp::BpEntry>(3);
    auto error = vector<uint8_t>{0, 0, 1};
    auto syndrome = rep_code.mulvec(error);
    auto error_channel = vector<double>{0.1,0.1,0.1};
    auto lbr = vector<double>{0, 0, 0};
    auto decoder = ldpc::osd::OsdDecoder(rep_code,ldpc::osd::OSD_0,0,error_channel);
    auto decoding = decoder.decode(syndrome,lbr);
    auto syndrome2 = rep_code.mulvec(decoding);
    ASSERT_TRUE(syndrome2 == syndrome);

}

TEST(OsdDecoder, PrintsCorrectly) {
    auto pcm = ldpc::bp::BpSparse(4, 4);
    
    for(int i = 0; i<4; i++) pcm.insert_entry(i, i);
    pcm.insert_entry(1, 0);
    pcm.insert_entry(2, 0);

    // ldpc::sparse_matrix_util::print_sparse_matrix(pcm);

    auto error = vector<uint8_t>{0, 1, 1, 1};
    auto error_channel = vector<double>{0.1,0.1,0.1,0.1};
    auto syndrome = pcm.mulvec(error);

    // cout<<endl;
    // ldpc::sparse_matrix_util::print_vector(syndrome);

    // cout<<endl;

    auto lbr = vector<double>{0, 0, 0, 0};
    for(int i = 0; i<4; i++) lbr[i] = log((1-error_channel[i])/(error_channel[i]));
    auto osdD = new ldpc::osd::OsdDecoder(pcm,ldpc::osd::OSD_0,0,error_channel);
    auto decoding = osdD->decode(syndrome, lbr);
    auto syndrome2 = pcm.mulvec(decoding);

    // cout<<endl;
    // ldpc::sparse_matrix_util::print_vector(syndrome2);

    ASSERT_TRUE(syndrome == syndrome2);


    auto LU = ldpc::gf2sparse_linalg::RowReduce(pcm);
    LU.rref(false,true);
    LU.lu_solve(syndrome);

    auto syndrome3 = pcm.mulvec(decoding);

    // ldpc::sparse_matrix_util::print_vector(syndrome3);
    // ldpc::sparse_matrix_util::print_vector(LU.rows);

    auto LUpcm = LU.L.matmul(LU.U);

    // ldpc::sparse_matrix_util::print_sparse_matrix(LUpcm);

    // cout<<endl;

    // ldpc::sparse_matrix_util::print_sparse_matrix(LU.L);
    // cout<<endl;
    // ldpc::sparse_matrix_util::print_sparse_matrix(LU.U);

    ASSERT_TRUE(syndrome == syndrome3);


    delete osdD;

}


TEST(OsdDecoder, AllZeroSyndrome) {
    auto pcm = ldpc::bp::BpSparse(4, 4);
    for(int i = 0; i<4; i++) pcm.insert_entry(i, i);
    pcm.insert_entry(1, 0);
    pcm.insert_entry(2, 0);

    auto error = vector<uint8_t>{0, 1, 1, 1};
    auto syndrome = vector<uint8_t>{0, 0, 0, 0};
    auto error_channel = vector<double>{0.1,0.1,0.1,0.1};
    auto lbr = vector<double>{0, 0, 0, 0};
    for(int i = 0; i<4; i++) lbr[i] = log((1-error_channel[i])/(error_channel[i]));
    auto osdD = new ldpc::osd::OsdDecoder(pcm,ldpc::osd::OSD_0,0,error_channel);
    auto decoding = osdD->decode(syndrome, lbr);
    for(int i = 0; i<4; i++) ASSERT_EQ(decoding[i], 0);
    delete osdD;
}


TEST(OsdDecoder, VariedErrorChannel) {
    auto pcm = ldpc::bp::BpSparse(4, 4);
    for(int i = 0; i<4; i++) pcm.insert_entry(i, i);
    pcm.insert_entry(1, 0);
    pcm.insert_entry(2, 0);

    auto error = vector<uint8_t>{0, 1, 1, 1};
    auto syndrome = vector<uint8_t>{0, 0, 0, 0};
    auto error_channel = vector<double>{0.1,0.2,0.3,0.4};
    auto lbr = vector<double>{0, 0, 0, 0};
    for(int i = 0; i<4; i++) lbr[i] = log((1-error_channel[i])/(error_channel[i]));
    auto osdD = new ldpc::osd::OsdDecoder(pcm,ldpc::osd::OSD_0,0,error_channel);
    auto decoding = osdD->decode(syndrome, lbr);
    auto syndrome2 = vector<uint8_t>{0, 0, 0, 0};
    pcm.mulvec(decoding, syndrome2);
    for(int i = 0; i<4; i++) ASSERT_EQ(syndrome[i], syndrome2[i]);
    delete osdD;

}

TEST(OsdDecoder, VariedErrorChannelLargerMatrix) {
    auto pcm = ldpc::bp::BpSparse(3, 5);
    pcm.insert_entry(0, 0);
    pcm.insert_entry(0, 1);
    pcm.insert_entry(1, 0);
    pcm.insert_entry(1, 2);
    pcm.insert_entry(1, 3);
    pcm.insert_entry(2, 1);
    pcm.insert_entry(2, 2);
    pcm.insert_entry(2, 4);

    auto error = vector<uint8_t>{1, 0, 1, 0, 1};
    auto syndrome = vector<uint8_t>{0, 0, 0};
    auto error_channel = vector<double>{0.1,0.2,0.3,0.4,0.5};
    auto lbr = vector<double>{0, 0, 0, 0, 0};
    for(int i = 0; i<5; i++) lbr[i] = log((1-error_channel[i])/(error_channel[i]));
    auto osdD = new ldpc::osd::OsdDecoder(pcm,ldpc::osd::OSD_0,0,error_channel);
    auto decoding = osdD->decode(syndrome, lbr);
    auto syndrome2 = vector<uint8_t>{0, 0, 0};
    pcm.mulvec(decoding, syndrome2);
    for(int i = 0; i<3; i++) ASSERT_EQ(syndrome[i], syndrome2[i]);


    delete osdD;
}


TEST(OsdDecoder, DecodeHammingCode) {
    auto pcm = ldpc::bp::BpSparse(3, 7);

    // Set up the Hamming parity check matrix
    pcm.insert_entry(0, 3); pcm.insert_entry(0, 4); pcm.insert_entry(0, 5); pcm.insert_entry(0, 6);
    pcm.insert_entry(1, 1); pcm.insert_entry(1, 2); pcm.insert_entry(1, 5); pcm.insert_entry(1, 6);
    pcm.insert_entry(2, 0); pcm.insert_entry(2, 2); pcm.insert_entry(2, 4); pcm.insert_entry(2, 6);

    // ldpc::sparse_matrix_util::print_sparse_matrix(pcm);

    // Create a received codeword with a single-bit error
    auto error = vector<uint8_t>{0, 1, 0, 1, 0, 1, 1};
    auto syndrome = vector<uint8_t>{0, 0, 0};
    pcm.mulvec(error, syndrome);

    // ldpc::sparse_matrix_util::print_vector(syndrome);

    // Create a vector of log-likelihood ratios for the received codeword
    auto error_channel = vector<double>{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    auto lbr = vector<double>(7);
    for (int i = 0; i < 7; i++) {
        lbr[i] = log((1 - error_channel[i]) / error_channel[i]);
    }

    // Decode the received codeword
    auto osdD = new ldpc::osd::OsdDecoder(pcm, ldpc::osd::OSD_OFF, 0, error_channel);

    osdD->osd_method = ldpc::osd::OSD_0;
    osdD->osd_order = 4;
    osdD->osd_setup();

    ASSERT_EQ(osdD->osd_candidate_strings.size(), 0);


    osdD->osd_method = ldpc::osd::EXHAUSTIVE;
    osdD->osd_order = 4;
    osdD->osd_setup();

    ASSERT_EQ(osdD->osd_candidate_strings.size(), 16-1);

    osdD->osd_method = ldpc::osd::COMBINATION_SWEEP;
    osdD->osd_order = 4;
    osdD->osd_setup();

    ASSERT_EQ(osdD->osd_candidate_strings.size(), 10);


    auto decoding = osdD->decode(syndrome, lbr);

    // Verify that the decoded codeword is valid by computing the syndrome
    auto syndrome2 = vector<uint8_t>{0, 0, 0};
    pcm.mulvec(decoding, syndrome2);
    // ldpc::sparse_matrix_util::print_vector(decoding);
    for (int i = 0; i < 3; i++) {
        ASSERT_EQ(syndrome[i], syndrome2[i]);
    }

    delete osdD;
}



int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}