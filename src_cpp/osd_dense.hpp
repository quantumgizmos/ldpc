//
// Created by luca on 15/01/24.
//

#ifndef CPP_TEST_OSD_DENSE_HPP
#define CPP_TEST_OSD_DENSE_HPP

#include "osd.hpp"
#include "gf2dense.hpp"

namespace ldpc::osd {

/**
 * This is the analog of the osd decoder in osd.hpp for the dense case used here.
 *
 */
    class DenseOsdDecoder {
    public:
        OsdMethod osd_method;
        int osd_order;
        int k, bit_count, check_count;
        gf2dense::CscMatrix &pcm;
        const std::vector<double> &channel_probabilities;
        std::vector<uint8_t> lsd0_solution;
        std::vector<uint8_t> osdw_decoding;
        std::vector<std::vector<uint8_t>> osd_candidate_strings;
        gf2dense::PluDecomposition plu_decomposition;

        DenseOsdDecoder(
                gf2dense::CscMatrix &parity_check_matrix,
                gf2dense::PluDecomposition &pluDecomposition,
                osd::OsdMethod osd_method,
                int osd_order,
                int n,
                int m,
                const std::vector<double> &channel_probs) :
                pcm(parity_check_matrix),
                channel_probabilities(channel_probs),
                plu_decomposition(pluDecomposition) {
            this->bit_count = n;
            this->check_count = m;
            this->k = 0;
            this->osd_order = osd_order;
            this->osd_method = osd_method;
            this->osd_setup();
        }


        ~DenseOsdDecoder() {
            this->lsd0_solution.clear();
            this->osdw_decoding.clear();
            this->osd_candidate_strings.clear();
        }


        int osd_setup() {
            int osd_candidate_string_count;

            this->osd_candidate_strings.clear();
            if (this->osd_method == osd::OSD_OFF) {
                return 0;
            }
            this->k = this->plu_decomposition.not_pivot_cols.size();

            if (this->osd_method == osd::OSD_0 || this->osd_order == 0) {
                return 1;
            }

            if (this->osd_method == osd::EXHAUSTIVE) {
                osd_candidate_string_count = pow(2, this->osd_order);
                for (auto i = 1; i < osd_candidate_string_count; i++) {
                    this->osd_candidate_strings.push_back(ldpc::util::decimal_to_binary_reverse(i, k));
                }
            }

            if (this->osd_method == osd::COMBINATION_SWEEP) {
                for (auto i = 0; i < k; i++) {
                    std::vector<uint8_t> osd_candidate;
                    osd_candidate.resize(k, 0);
                    osd_candidate[i] = 1;
                    this->osd_candidate_strings.push_back(osd_candidate);
                }

                for (auto i = 0; i < this->osd_order; i++) {
                    for (auto j = 0; j < this->osd_order; j++) {
                        if (j <= i) continue;
                        if (k > 0) {
                            std::vector<uint8_t> osd_candidate;
                            osd_candidate.resize(k, 0);
                            osd_candidate[i] = 1;
                            osd_candidate[j] = 1;
                            this->osd_candidate_strings.push_back(osd_candidate);
                        }
                    }
                }
            }
            return 1;
        }


        std::vector<uint8_t> &osd_decode(std::vector<uint8_t> &syndrome) {
            // note that we do not include column orderings as in osd.hpp since this is already done 'by construction'
            // of the clusters through the guided growth.
            this->lsd0_solution = this->osdw_decoding = plu_decomposition.lu_solve(syndrome);

            int candidate_weight, osd_min_weight;

            osd_min_weight = 0;
            for (auto i = 0; i < this->bit_count; i++) {
                if (this->lsd0_solution[i] == 1) {
                    osd_min_weight++;
                }
            }
            // reset if NaN
//            osd_min_weight = std::isnan(osd_min_weight) ? 0.0 : osd_min_weight;

            auto non_pivot_columns = this->plu_decomposition.not_pivot_cols;
            if (non_pivot_columns.empty()) {
//                std::cout << "no non-pivot columns" << std::endl;
                return this->lsd0_solution;
            }
            for (auto &candidate_string: this->osd_candidate_strings) {
                auto t_syndrome = syndrome;
                int col_index = 0;
                for (auto col: non_pivot_columns) {
                    if (candidate_string[col_index] == 1) {
                        for (auto e: this->pcm.at(col)) {
                            t_syndrome[e] ^= 1;
                        }
                    }
                    col_index++;
                }

                auto candidate_solution = plu_decomposition.lu_solve(t_syndrome);
                for (auto i = 0; i < k; i++) {
                    candidate_solution[non_pivot_columns[i]] = candidate_string[i];
                }
                candidate_weight = 0;

                auto decoded_t_syndrome = std::vector<uint8_t>(t_syndrome.size(), 0);

                for (auto i = 0; i < this->bit_count; i++) {
                    if (candidate_solution[i] == 1) {
                        for(int synd_idx: this->pcm.at(i)) {
                            decoded_t_syndrome[synd_idx] ^= 1;
                        }
                        candidate_weight++;
                    }
                }

                //we abandon this candidate solution if the solution does satisfy the input
                if(decoded_t_syndrome != syndrome){
                    std::cout<<"Hello"<<std::endl;
                    continue;
                }

                // reset if NaN
//                candidate_weight = std::isnan(candidate_weight) ? 0.0 : candidate_weight;
                if (candidate_weight < osd_min_weight) {
//                    std::cout << "found lower weight solution" << std::endl;
                    osd_min_weight = candidate_weight;
                    this->osdw_decoding = candidate_solution;
                }
            }

            return this->osdw_decoding;
        }
    };
}
#endif //CPP_TEST_OSD_DENSE_HPP