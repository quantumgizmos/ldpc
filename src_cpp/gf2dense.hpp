#ifndef GF2DENSE_H
#define GF2DENSE_H

#include <vector>
#include <iterator>
#include <chrono>
#include <climits>
#include <random>

#include "util.hpp"
#include "gf2sparse.hpp"
#include "sparse_matrix_util.hpp"
#include "rng.hpp"

namespace ldpc {
    namespace gf2dense {

        template<class T>
        int vector_find(std::vector<T> vec, T value) {
            int index = 0;
            for (auto val: vec) {
                if (val == value) return index;
                index++;
            }
            return -1;
        }

        enum PluMethod {
            SPARSE_ELIMINATION = 0,
            DENSE_ELIMINATION = 1
        };

        std::vector<int> NULL_INT_VECTOR = {};
        typedef std::vector<std::vector<int>> CscMatrix;
        typedef std::vector<std::vector<int>> CsrMatrix;


        CsrMatrix csc_to_csr(CscMatrix csc_mat) {
            int row_count = -1;
            for (auto &col: csc_mat) {
                for (int entry: col) {
                    if (entry > row_count) {
                        row_count = entry;
                    }
                }
            }

            auto csr_mat = CsrMatrix(row_count + 1, std::vector<int>{});

            for (int col_index = 0; col_index < csc_mat.size(); col_index++) {
                for (int row_index: csc_mat[col_index]) {
                    csr_mat[row_index].push_back(col_index);
                }
            }
            return csr_mat;
        }


        void print_csc(int row_count, int col_count, std::vector<std::vector<int>> csc_mat) {
            CsrMatrix csr_matrix;
            for (int i = 0; i < row_count; i++) {
                csr_matrix.push_back(std::vector<int>{});
            }

            int col_index = 0;
            for (auto col: csc_mat) {
                for (auto entry: col) {
                    csr_matrix[entry].push_back(col_index);
                }
                col_index++;
            }

            for (auto row: csr_matrix) {
                auto row_dense = std::vector<int>(col_count, 0);
                for (auto entry: row) {
                    row_dense[entry] = 1;
                }
                ldpc::sparse_matrix_util::print_vector(row_dense);
            }
        }

        /**
         * A class to represent the PLU decomposition.
         * That is for a matrix A, we have P*A = L*U, where P is a permutation matrix.
         */
        class PluDecomposition {
        private:
            CscMatrix &csc_mat; // csc_mat[column][row]
        public:
            CscMatrix L;
            CscMatrix U;
            CscMatrix P;
            int matrix_rank{};
            int row_count{};
            int col_count{};
            std::vector<int> rows; // todo is this needed?
            std::vector<int> swap_rows;
            std::vector<int> pivot_cols;
            std::vector<int> not_pivot_cols;

            PluDecomposition(int row_count, int col_count, std::vector<std::vector<int>> &csc_mat)
                    : row_count(row_count), col_count(col_count), csc_mat(csc_mat), matrix_rank(0) {}

            PluDecomposition(PluDecomposition &plu) = default;

            ~PluDecomposition() =
            default;

            /**
             * Reset all internal information.
             */
            void reset() {
                this->matrix_rank = 0;
                this->rows.clear();
                this->swap_rows.clear();
                this->pivot_cols.clear();
                this->not_pivot_cols.clear();

                for (auto &col: this->L) {
                    col.clear();
                }
                this->L.clear();

                for (auto &col: this->U) {
                    col.clear();
                }
                this->U.clear();

                for (auto &col: this->P) {
                    col.clear();
                }
                this->P.clear();
            }

            /**
             * Compute the reduced row-echelon form
             * The algorithm operates column, wise. The original matrix, csc_mat is not modified.
             * Instead, rr_col is used to represent the column currently eliminated and row operations, e.g., swaps
             * and addition is done column per column.
             * First, all previous row operations are applied to the current column.
             * Then pivoting is done for the current column and corresponding swaps are applied.
             * @param construct_U
             */
            void rref(const bool construct_U = true) {
                this->reset();
                for (auto i = 0; i < this->row_count; i++) {
                    this->rows.push_back(i);
                }
                std::vector<std::size_t> rr_col;
                rr_col.resize(this->row_count, 0);
                auto max_rank = std::min(this->row_count, this->col_count);

                for (auto col = 0; col < this->col_count; col++) {
                    std::fill(rr_col.begin(), rr_col.end(), 0);
                    for (auto row_index: this->csc_mat[col]) {
                        rr_col[row_index] = 1;
                    }
                    // apply previous operations to current column
                    for (auto i = 0; i < this->matrix_rank; i++) {
                        std::swap(rr_col[i], rr_col[this->swap_rows[i]]);
                        if (rr_col[i] == 1) {
                            // if row elem is one, do elimination for current column below the pivot
                            // elimination operations to apply are stored in columns of L,
                            // start with +1 offset since we do not need to eliminate the diagonal entry
                            for (auto it = this->L[i].begin() + 1; it != this->L[i].end(); it++) {
                                auto add_row = *it;
                                rr_col[add_row] ^= 1;
                            }
                        }
                    }
                    bool PIVOT_FOUND = false;
                    for (auto i = this->matrix_rank; i < this->row_count; i++) {
                        if (rr_col[i] == 1) {
                            PIVOT_FOUND = true;
                            this->swap_rows.push_back(i);
                            this->pivot_cols.push_back(col);
                            break;
                        }
                    }
                    // if no pivot was found, we go to next column
                    if (!PIVOT_FOUND) {
                        this->not_pivot_cols.push_back(col);
                        continue;
                    }

                    std::swap(rr_col[this->matrix_rank], rr_col[this->swap_rows[this->matrix_rank]]);
                    std::swap(this->rows[this->matrix_rank], this->rows[this->swap_rows[this->matrix_rank]]);
                    this->L.push_back(std::vector<int>{this->matrix_rank});

                    for (auto i = this->matrix_rank + 1; i < this->row_count; i++) {
                        if (rr_col[i] == 1) {
                            // rr_col[i] ^= 1; //we don't actually need to eliminate here since we throw away rr_col
                            this->L[this->matrix_rank].push_back(i);
                        }
                    }
                    this->matrix_rank++;

                    if (construct_U) {
                        this->U.push_back(std::vector<int>{});
                        for (auto i = 0; i < matrix_rank; i++) {
                            if (rr_col[i] == 1) {
                                this->U[col].push_back(i);
                            }
                        }
                    }
                    if (this->matrix_rank == max_rank) {
                        break;
                    }
                }
            }

            /**
             * Apply the stored operations to a vector.
             * The vector is not changed.
             * @param vec
             */
            std::vector<int> apply_stored_operations_to_csc_column(const std::vector<int> &col) {
                auto result = col;
                for (auto i = 0; i < this->matrix_rank; i++) {
                    std::swap(result[i], result[this->swap_rows[i]]);
                    if (result[i] == 1) {
                        for (auto it = this->L[i].begin() + 1; it != this->L[i].end(); it++) {
                            auto add_row = *it;
                            result[add_row] ^= 1;
                        }
                    }
                }
                return result;
            }

            bool vector_is_in_image(const std::vector<int> &vec) {
                auto result = true;
                for (auto &e: vec) {
                    if (e >= this->matrix_rank) {
                        result = false;
                        break;
                    }
                }
                return result;
            }

            /**
             * Compute the reduced row-echelon form starting from a specific column.
             * Assumes that columns with indices starting from start_col_idx are present in this->csc_mat.
             * Assumes stores operations have been applied already.
             * @param col_idx
             * @param construct_U
             */
            void partial_rref(const std::size_t start_col_idx,
                              const bool construct_U = true) {
                std::vector<std::size_t> rr_col;
                rr_col.resize(this->row_count, 0);
                auto max_rank = std::min(this->row_count, this->col_count);

                for (auto col = start_col_idx; col < this->col_count; col++) {
                    std::fill(rr_col.begin(), rr_col.end(), 0);
                    for (auto row_index: this->csc_mat[col]) {
                        rr_col[row_index] = 1;
                    }
                    for (auto i = 0; i < this->matrix_rank; i++) {
                        std::swap(rr_col[i], rr_col[this->swap_rows[i]]);
                        if (rr_col[i] == 1) {
                            for (auto it = this->L[i].begin() + 1; it != this->L[i].end(); it++) {
                                auto add_row = *it;
                                rr_col[add_row] ^= 1;
                            }
                        }
                    }
                    bool PIVOT_FOUND = false;
                    for (auto i = this->matrix_rank; i < this->row_count; i++) {
                        if (rr_col[i] == 1) {
                            PIVOT_FOUND = true;
                            this->swap_rows.push_back(i);
                            this->pivot_cols.push_back(col);
                            break;
                        }
                    }
                    if (!PIVOT_FOUND) {
                        this->not_pivot_cols.push_back(col);
                        continue;
                    }

                    std::swap(rr_col[this->matrix_rank], rr_col[this->swap_rows[this->matrix_rank]]);
                    std::swap(this->rows[this->matrix_rank], this->rows[this->swap_rows[this->matrix_rank]]);
                    this->L.push_back(std::vector<int>{this->matrix_rank});

                    for (auto i = this->matrix_rank + 1; i < this->row_count; i++) {
                        if (rr_col[i] == 1) {
                            // rr_col[i] ^= 1; //we don't actually need to eliminate here since we throw away rr_col
                            this->L[this->matrix_rank].push_back(i);
                        }
                    }
                    this->matrix_rank++;

                    if (construct_U) {
                        this->U.emplace_back();
                        for (auto i = 0; i < matrix_rank; i++) {
                            if (rr_col[i] == 1) {
                                this->U[col].push_back(i);
                            }
                        }
                    }
                    if (this->matrix_rank == max_rank) {
                        break;
                    }
                }
            }

            /**
             * Adds new, not eliminated columns and rows to the matrix.
             * Updates row_count and col_count
             * @param new_matrix
             */
            void update_matrix(const CscMatrix &new_matrix) {
                if (this->row_count == new_matrix.size() or this->col_count == new_matrix[0].size()) {
                    return;
                }
                for (auto i = this->col_count; i < new_matrix.size(); i++) {
                    this->csc_mat.push_back(new_matrix[i]);
                    auto max_elem = std::max_element(new_matrix[i].begin(), new_matrix[i].end());
                    if (*max_elem > this->row_count) {
                        this->row_count = *max_elem;
                    }
                }
                this->col_count = new_matrix.size();
            }
        };


        int rank(int row_count, int col_count, CscMatrix &csc_mat) {
            auto plu = PluDecomposition(row_count, col_count, csc_mat);
            plu.rref(false);
            return plu.matrix_rank;
        }

        CscMatrix kernel(int row_count, int col_count, CsrMatrix &csr_mat) {

            // To compute the kernel, we need to do PLU decomposition on the transpose of the matrix.
            // The CSR representation of mat is the CSC representation of mat.transpose().

            auto plu = PluDecomposition(col_count, row_count, csr_mat);
            plu.rref(false);

            std::vector<size_t> rr_col(col_count, 0);
            std::vector<std::vector<int>> ker;

            for (int j = 0; j < col_count; j++) {

                ker.push_back(std::vector<int>{});

                std::fill(rr_col.begin(), rr_col.end(), 0);

                rr_col[j] = 1;

                for (int i = 0; i < plu.matrix_rank; i++) {

                    std::swap(rr_col[i], rr_col[plu.swap_rows[i]]);
                    if (rr_col[i] == 1) {
                        for (auto it = plu.L[i].begin() + 1; it != plu.L[i].end(); ++it) {
                            int add_row = *it;
                            rr_col[add_row] ^= 1;
                        }
                    }
                }

                for (int i = plu.matrix_rank; i < col_count; i++) {
                    if (rr_col[i] == 1) {
                        ker[j].push_back(i - plu.matrix_rank);
                    }
                }
            }
            return ker; // returns the kernel as a csc matrix
        }

        std::vector<int> pivot_rows(int row_count, int col_count, CsrMatrix &csr_mat) {
            auto plu = PluDecomposition(col_count, row_count, csr_mat);
            plu.rref(false);
            return plu.pivot_cols;
        }

        CsrMatrix row_span(int row_count, int col_count, CsrMatrix &csr_mat) {
            int row_permutations = std::pow(2, row_count);
            CsrMatrix row_span;

            for (int i = 0; i < row_permutations; i++) {

                std::vector<size_t> current_row(col_count, 0);
                auto row_add_indices = ldpc::util::decimal_to_binary_sparse(i, row_count);
                for (auto row_index: row_add_indices) {
                    for (auto col_index: csr_mat[row_index]) {
                        current_row[col_index] ^= 1;
                    }
                }
                std::vector<int> current_row_sparse;
                for (int j = 0; j < col_count; j++) {
                    if (current_row[j] == 1) {
                        current_row_sparse.push_back(j);
                    }
                }
                row_span.push_back(current_row_sparse);
            }
            return row_span;
        }

        struct DistanceStruct {
            int min_distance = INT_MAX;
            int samples_searched = 0;
            std::vector<std::vector<int>> min_weight_words;
        };

        DistanceStruct estimate_minimum_linear_row_combination(int row_count, int col_count, CsrMatrix &csr_mat,
                                                               double timeout_seconds = 0,
                                                               int number_of_words_to_save = 100) {

            DistanceStruct distance_struct;
            distance_struct.min_weight_words.resize(number_of_words_to_save, std::vector<int>{});
            int max_weight_saved_word = INT_MAX;
            int cc = 0;

            for (const auto &word: csr_mat) {
                int word_size = word.size();
                if (word_size < max_weight_saved_word) {
                    int max1 = -10;
                    int max2 = -10;
                    int replace_word_index;
                    int count_index = 0;

                    for (const auto &saved_word: distance_struct.min_weight_words) {
                        int saved_word_size = saved_word.size();
                        if (saved_word_size == 0) {
                            replace_word_index = count_index;
                            break;
                        } else if (saved_word_size > max1) {
                            max1 = saved_word_size;
                            replace_word_index = count_index;
                        } else if (saved_word_size > max2) {
                            max2 = saved_word_size;
                        }
                        count_index++;
                    }

                    distance_struct.min_weight_words[replace_word_index] = word;
                    if (word_size > max2) {
                        max_weight_saved_word = word_size;
                    } else {
                        max_weight_saved_word = max2;
                    }

                }

            }


            double sample_prob = 2.0 / double(row_count);


            int row_permutations = std::pow(2, row_count);
            auto rand_gen = ldpc::rng::RandomNumberGenerator();

            auto start = std::chrono::high_resolution_clock::now();

            int count = 0;
            while (true) {
                count++;

                // auto rand = rand_gen.random_int(row_permutations-1);
                // auto row_add_indices = ldpc::util::decimal_to_binary_sparse(rand,row_count);

                std::vector<int> row_add_indices;
                for (int i = 0; i < row_count; i++) {
                    if (rand_gen.random_double() < sample_prob) {
                        row_add_indices.push_back(i);
                    }
                }

                std::vector<size_t> current_row(col_count, 0);
                for (auto row_index: row_add_indices) {
                    for (auto col_index: csr_mat[row_index]) {
                        current_row[col_index] ^= 1;
                    }
                }


                std::vector<int> current_row_sparse;
                for (int j = 0; j < col_count; j++) {
                    if (current_row[j] == 1) {
                        current_row_sparse.push_back(j);
                    }
                }

                int current_row_size = current_row_sparse.size();

                if (current_row_size == 0) continue;

                if (current_row_size < distance_struct.min_distance) {
                    distance_struct.min_distance = current_row_size;
                }

                // std::cout<<max_weight_saved_word<<std::endl;
                // ldpc::sparse_matrix_util::print_vector(current_row_sparse);

                if (current_row_size <= max_weight_saved_word) {

                    int max1 = -10;
                    int max2 = -10;
                    int replace_word_index;
                    int count_index = 0;
                    for (auto word: distance_struct.min_weight_words) {

                        int word_size = word.size();

                        if (word_size == 0) {
                            replace_word_index = count_index;
                            break;
                        } else if (word_size > max1) {
                            max1 = word_size;
                            replace_word_index = count_index;
                        } else if (word_size > max2) {
                            max2 = word_size;
                        }
                        count_index++;
                    }

                    distance_struct.min_weight_words[replace_word_index] = current_row_sparse;
                    if (current_row_size > max2) {
                        max_weight_saved_word = current_row_size;
                    } else {
                        max_weight_saved_word = max2;
                    }
                }

                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() / 1000.0;
                if (elapsed >= timeout_seconds) {
                    break;
                }

            }

            // std::cout<<"count: "<<count<<std::endl;
            // // std::cout<<std::pow(2,row_count)<<std::endl;

            distance_struct.samples_searched = count;

            return distance_struct;
        }

        DistanceStruct
        estimate_code_distance(int row_count, int col_count, CsrMatrix &csr_mat, double timeout_seconds = 0,
                               int number_of_words_to_save = 100) {

            auto ker = ldpc::gf2dense::kernel(row_count, col_count, csr_mat);
            // convert to csr_matrix
            int max_row = -1;
            for (int col_index = 0; col_index < ker.size(); col_index++) {
                for (int row_index: ker[col_index]) {
                    if (row_index > max_row) max_row = row_index;
                }
            }

            CsrMatrix ker_csr = CscMatrix(max_row + 1, std::vector<int>{});

            for (int col_index = 0; col_index < ker.size(); col_index++) {
                for (int row_index: ker[col_index]) {
                    ker_csr[row_index].push_back(col_index);
                }
            }

            return ldpc::gf2dense::estimate_minimum_linear_row_combination(max_row + 1, col_count, ker_csr,
                                                                           timeout_seconds,
                                                                           number_of_words_to_save);

        }

        int compute_exact_code_distance(int row_count, int col_count, CsrMatrix &csr_mat) {

            CscMatrix ker = ldpc::gf2dense::kernel(row_count, col_count, csr_mat);


            CsrMatrix ker_csr = csc_to_csr(ker);

            int row_permutations = std::pow(2, ker_csr.size());

            int distance = col_count;

            for (int i = 1; i < row_permutations; i++) {

                std::vector<size_t> current_row(col_count, 0);

                auto row_add_indices = ldpc::util::decimal_to_binary_sparse(i, ker_csr.size());

                int row_count = 0;
                for (auto row_index: row_add_indices) {
                    for (auto col_index: ker_csr[row_index]) {
                        if (current_row[col_index] == 0) {
                            row_count++;
                        } else {
                            row_count--;
                        }
                    }
                }

                // std::cout<<row_count<<std::endl;

                if (row_count < distance) {
                    distance = row_count;
                }

            }

            return distance;

        }


    }//end namespace gf2dense
}//end namespace ldpc

#endif