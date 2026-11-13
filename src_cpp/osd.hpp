#ifndef BPOSD_H
#define BPOSD_H

#include <vector>
#include <memory>
#include <iterator>
#include <cmath> 
#include <limits>
#include <cstdint>
#include <bit>

#include "bp.hpp"
#include "sort.hpp"
#include "util.hpp"
#include "gf2sparse_linalg.hpp"

namespace ldpc::osd{


enum OsdMethod{
    OSD_OFF,
    OSD_0,
    EXHAUSTIVE,
    COMBINATION_SWEEP
};


class OsdDecoder{
    public:
        OsdMethod osd_method;
        int osd_order;
        int k, bit_count, check_count;
        ldpc::bp::BpSparse& pcm;
        std::vector<double>& channel_probabilities;
        std::vector<uint8_t> osd0_decoding;
        std::vector<uint8_t> osdw_decoding;
        std::vector<std::vector<uint8_t>> osd_candidate_strings;
        std::vector<int> column_ordering;
        ldpc::gf2sparse_linalg::RowReduce<ldpc::bp::BpEntry>* LuDecomposition;
        
        OsdDecoder(
            ldpc::bp::BpSparse& parity_check_matrix,
            OsdMethod osd_method,
            int osd_order,
            std::vector<double>& channel_probs):
            pcm(parity_check_matrix),
            channel_probabilities(channel_probs)    
        {

            this->bit_count = this->pcm.n;
            this->check_count = this->pcm.m;

            this->osd_order = osd_order;
            this->osd_method = osd_method;

            this->osd_setup();


        }

        int osd_setup(){

            this->osd_candidate_strings.clear();
            
            if(this->osd_method == OSD_OFF) return 0;

            this->LuDecomposition = new ldpc::gf2sparse_linalg::RowReduce<ldpc::bp::BpEntry>(this->pcm);
            this->column_ordering.resize(this->pcm.n);
            int osd_candidate_string_count;
            // OSD-0 / order 0: decode() solves with fast_solve, which rebuilds its LU from the parity-check matrix, and
            // no candidate strings are needed, so the rank (and the full row reduction it costs) is never used.
            if(this->osd_method == OSD_0 || this->osd_order==0){
                this->k = -1;
                return 1;
            }
            if (this->osd_method == COMBINATION_SWEEP && this->dense_fits()) {
                this->k = this->pcm.n - this->dense_rank();            // rank only: same value as the sparse rref
            } else {
                this->LuDecomposition->rref(false,true);
                this->k = this->pcm.n - this->LuDecomposition->rank;
            }

            if(this->osd_method == OSD_0 || this->osd_order==0){
                return 1;
            }

            if(this->osd_method == EXHAUSTIVE){
                osd_candidate_string_count = pow(2,this->osd_order);
                for(int i=1; i<osd_candidate_string_count; i++){
                    this->osd_candidate_strings.push_back(ldpc::util::decimal_to_binary_reverse(i,k));
                }
            }

            if(this->osd_method == COMBINATION_SWEEP){
                for(int i=0; i<k; i++) {
                    std::vector<uint8_t> osd_candidate;
                    osd_candidate.resize(k,0);
                    osd_candidate[i]=1; 
                    this->osd_candidate_strings.push_back(osd_candidate);
                }

                // pairs among the first osd_order non-pivot columns (at most k of them: the candidates have length k)
                const int pair_order = std::min(this->osd_order, this->k);
                for(int i = 0; i<pair_order;i++){
                    for(int j = 0; j<pair_order; j++){
                        if(j<=i) continue;
                        std::vector<uint8_t> osd_candidate;
                        osd_candidate.resize(k,0);
                        osd_candidate[i]=1;
                        osd_candidate[j]=1; 
                        this->osd_candidate_strings.push_back(osd_candidate);
                    }
                }

            }
            return 1;
        }

        ~OsdDecoder(){
            delete this->LuDecomposition;
        };


        // The dense paths store H as an m x n bit matrix. Beyond this size (256 MB) the sparse LU is used instead.
        static constexpr size_t dense_max_words = (size_t)1 << 25;
        bool dense_fits() const {
            return (size_t)this->pcm.m * (size_t)((this->pcm.n + 1 + 63) / 64) <= dense_max_words;
        }

        // ---- fast exact combination sweep -------------------------------------------------------------
        // Every candidate of the sweep solves H x = s ^ (H columns of its flipped non-pivot bits) with the LU factors of
        // this call.  lu_solve is GF(2)-linear, so all single-flip candidates are obtained at once by running ldpc's own
        // forward/back substitution on bit-packed rows (bit j = candidate j).  Approximate weights shortlist the
        // candidates that can reach the minimum; those are re-weighed exactly as ldpc does (log(1/p) summed in index
        // order), and the first candidate attaining the minimum exact weight is returned - the result of ldpc's
        // sequential `candidate_weight < osd_min_weight` scan.
        int dense_rank() {
            const int n = this->pcm.n, m = this->pcm.m, Wn = (n + 63) / 64;
            std::vector<uint64_t> M((size_t)m * Wn, 0);
            for (int i = 0; i < m; i++) for (auto& e : this->pcm.iterate_row(i)) M[(size_t)i * Wn + (e.col_index >> 6)] ^= (uint64_t)1 << (e.col_index & 63);
            int R = 0;
            for (int p = 0; p < n && R < m; p++) {
                const int w0 = p >> 6; const uint64_t bit = (uint64_t)1 << (p & 63);
                int r = -1;
                for (int i = R; i < m; i++) if (M[(size_t)i * Wn + w0] & bit) { r = i; break; }
                if (r < 0) continue;
                if (r != R) for (int w = w0; w < Wn; w++) std::swap(M[(size_t)r * Wn + w], M[(size_t)R * Wn + w]);
                const uint64_t* pr = &M[(size_t)R * Wn];
                for (int i = R + 1; i < m; i++) {
                    uint64_t* ri = &M[(size_t)i * Wn];
                    if (!(ri[w0] & bit)) continue;
                    for (int w = w0; w < Wn; w++) ri[w] ^= pr[w];
                }
                R++;
            }
            return R;
        }

        int full_rank_cache = -1;

        // OSD-0 without the sparse LU. ldpc's fast_solve picks as pivots the greedy independent column set in the
        // soft-decision order and returns the unique solution supported on it. Greedy pivots depend only on the
        // prefix of the order, so eliminate a dense bit-packed prefix and double it until the rank reaches rank(H)
        // (cached); then back-substitute. As in fast_solve, elimination stops once the reduced syndrome is zero below
        // the current pivot row: the remaining pivot bits of the unique solution are then zero. The result is checked against H x = s; on failure (syndrome outside the
        // column space), or if H is too large for the dense matrix, the caller falls back to fast_solve, so the
        // output is always ldpc's.
        bool osd0_dense(std::vector<uint8_t>& syndrome) {
            const int n = this->pcm.n, m = this->pcm.m;
            if (!this->dense_fits()) return false;
            std::vector<int> pos(n);
            for (int p = 0; p < n; p++) pos[this->column_ordering[p]] = p;
            const int max_rank = std::min(m, n);
            int P = std::min(n, std::max(1024, 2 * m));
            std::vector<uint64_t> M;
            std::vector<int> piv_pos;
            while (true) {
                const int Wn = (P + 1 + 63) / 64;                    // column P holds the syndrome
                M.assign((size_t)m * Wn, 0);
                for (int i = 0; i < m; i++) {
                    uint64_t* row = &M[(size_t)i * Wn];
                    for (auto& e : this->pcm.iterate_row(i)) {
                        const int p = pos[e.col_index];
                        if (p < P) row[p >> 6] ^= (uint64_t)1 << (p & 63);
                    }
                    if (syndrome[i]) row[P >> 6] ^= (uint64_t)1 << (P & 63);
                }
                piv_pos.clear();
                const int sw = P >> 6; const uint64_t sbit = (uint64_t)1 << (P & 63);
                int live = 0;                                        // rows >= R with a nonzero reduced syndrome bit
                for (int i = 0; i < m; i++) if (syndrome[i]) live++;
                int R = 0;
                for (int p = 0; p < P && R < max_rank && live > 0; p++) {
                    const int w0 = p >> 6; const uint64_t bit = (uint64_t)1 << (p & 63);
                    int r = -1;
                    for (int i = R; i < m; i++) if (M[(size_t)i * Wn + w0] & bit) { r = i; break; }
                    if (r < 0) continue;
                    if (r != R) for (int w = w0; w < Wn; w++) std::swap(M[(size_t)r * Wn + w], M[(size_t)R * Wn + w]);
                    const uint64_t* pr = &M[(size_t)R * Wn];
                    const bool ps = pr[sw] & sbit;
                    for (int i = R + 1; i < m; i++) {
                        uint64_t* ri = &M[(size_t)i * Wn];
                        if (!(ri[w0] & bit)) continue;
                        if (ps) live += (ri[sw] & sbit) ? -1 : 1;
                        for (int w = w0; w < Wn; w++) ri[w] ^= pr[w];
                    }
                    if (ps) live--;
                    piv_pos.push_back(p);
                    R++;
                }
                if (live > 0 && P < n) {
                    if (this->full_rank_cache < 0) this->full_rank_cache = this->dense_rank();
                    if (R < this->full_rank_cache) { P = std::min(n, 2 * P); continue; }
                }
                // back substitution on the echelon form
                std::vector<uint8_t> xp(R, 0);
                this->osd0_decoding.assign(n, 0);
                for (int r = R - 1; r >= 0; r--) {
                    const uint64_t* row = &M[(size_t)r * Wn];
                    uint8_t b = (row[P >> 6] >> (P & 63)) & 1;
                    for (int q = r + 1; q < R; q++) {
                        const int pq = piv_pos[q];
                        if (xp[q] && ((row[pq >> 6] >> (pq & 63)) & 1)) b ^= 1;
                    }
                    xp[r] = b;
                    if (b) this->osd0_decoding[this->column_ordering[piv_pos[r]]] = 1;
                }
                for (int i = 0; i < m; i++) {
                    uint8_t par = 0;
                    for (auto& e : this->pcm.iterate_row(i)) par ^= this->osd0_decoding[e.col_index];
                    if (par != (syndrome[i] ? 1 : 0)) return false;
                }
                return true;
            }
        }

        double exact_weight(const std::vector<uint8_t>& v) {
            double w = 0;
            for (int i = 0; i < this->pcm.n; i++) {
                if (v[i] == 1) {
                    w += log(1 / this->channel_probabilities[i]);
                }
            }
            return w;
        }

        // Dense bit-packed Gauss-Jordan elimination in the soft-decision column order.  The pivot columns are the
        // greedy independent set in that order, exactly the set ldpc's rref finds, and the OSD-0 solution and every
        // sweep candidate are unique given that set, so this reproduces ldpc's results without its sparse LU.
        // Returns false, leaving the decodings untouched, if H is too large for the dense matrix or the syndrome is
        // not in the column space of H (there is no solution then); the caller then takes the LU path, so the output
        // is still ldpc's.
        bool dense_combination_sweep(std::vector<uint8_t>& syndrome) {
            if (!this->dense_fits()) return false;
            const int n = this->pcm.n, m = this->pcm.m;
            const int Wn = (n + 1 + 63) / 64;                       // column n holds the syndrome
            std::vector<int> pos(n);                                 // column -> position in the soft-decision order
            for (int p = 0; p < n; p++) pos[this->column_ordering[p]] = p;
            std::vector<uint64_t> M((size_t)m * Wn, 0);
            for (int i = 0; i < m; i++) {
                uint64_t* row = &M[(size_t)i * Wn];
                for (auto& e : this->pcm.iterate_row(i)) { int p = pos[e.col_index]; row[p >> 6] ^= (uint64_t)1 << (p & 63); }
                if (syndrome[i]) row[n >> 6] ^= (uint64_t)1 << (n & 63);
            }
            const int max_rank = std::min(m, n);
            std::vector<int> piv_pos; piv_pos.reserve(max_rank);
            std::vector<uint8_t> is_piv(n, 0);
            std::vector<int> rows_with;
            int R = 0;
            for (int p = 0; p < n && R < max_rank; p++) {
                const int w0 = p >> 6; const uint64_t bit = (uint64_t)1 << (p & 63);
                int r = -1;
                for (int i = R; i < m; i++) if (M[(size_t)i * Wn + w0] & bit) { r = i; break; }
                if (r < 0) continue;
                if (r != R) for (int w = w0; w < Wn; w++) std::swap(M[(size_t)r * Wn + w], M[(size_t)R * Wn + w]);
                const uint64_t* pr = &M[(size_t)R * Wn];
                for (int i = 0; i < m; i++) {
                    if (i == R) continue;
                    uint64_t* ri = &M[(size_t)i * Wn];
                    if (!(ri[w0] & bit)) continue;
                    for (int w = w0; w < Wn; w++) ri[w] ^= pr[w];
                }
                piv_pos.push_back(p); is_piv[p] = 1; R++;
            }
            for (int i = R; i < m; i++) if ((M[(size_t)i * Wn + (n >> 6)] >> (n & 63)) & 1) return false;
            // column orderings exactly as ldpc's rref leaves them: pivots, then non-pivots, each in soft-decision order
            std::vector<int> N; N.reserve(n - R);
            for (int p = 0; p < n; p++) if (!is_piv[p]) N.push_back(this->column_ordering[p]);
            const int K = (int)N.size();
            std::vector<int> np_index(n, -1);                        // position -> non-pivot index
            for (int p = 0, j = 0; p < n; p++) if (!is_piv[p]) np_index[p] = j++;
            // OSD-0: pivot column of row r takes the reduced syndrome bit of row r
            this->osd0_decoding.assign(n, 0);
            for (int r = 0; r < R; r++) if ((M[(size_t)r * Wn + (n >> 6)] >> (n & 63)) & 1) this->osd0_decoding[this->column_ordering[piv_pos[r]]] = 1;
            if (K == 0) {                                            // full column rank: no candidates to sweep
                this->osdw_decoding = this->osd0_decoding;
                return true;
            }
            double w0sum = this->exact_weight(this->osd0_decoding);
            // X[r] : row r of the reduced matrix on the non-pivot columns, re-indexed by non-pivot index
            const int W = (K + 63) / 64;
            std::vector<uint64_t> X((size_t)R * W, 0);
            for (int r = 0; r < R; r++) {
                const uint64_t* row = &M[(size_t)r * Wn];
                uint64_t* xr = &X[(size_t)r * W];
                for (int w = 0; w < Wn; w++) {
                    uint64_t v = row[w];
                    while (v) {
                        int t = std::countr_zero(v); v &= v - 1;
                        int p = (w << 6) + t;
                        if (p >= n || is_piv[p]) continue;
                        int j = np_index[p];
                        xr[j >> 6] |= (uint64_t)1 << (j & 63);
                    }
                }
            }
            std::vector<int> pivcol(R);
            for (int r = 0; r < R; r++) pivcol[r] = this->column_ordering[piv_pos[r]];
            this->sweep_from(X, W, R, K, N, pivcol, w0sum);
            return true;
        }

        void sweep_from(const std::vector<uint64_t>& X, int W, int R, int K, const std::vector<int>& N,
                        const std::vector<int>& pivcol, double osd0_weight) {
            const int n = this->pcm.n;
            // approximate weights of all single-flip candidates
            std::vector<double> wcol(n);
            for (int i = 0; i < n; i++) wcol[i] = log(1 / this->channel_probabilities[i]);
            std::vector<double> approx(K);
            for (int j = 0; j < K; j++) approx[j] = osd0_weight + wcol[N[j]];
            for (int r = 0; r < R; r++) {
                int c = pivcol[r];
                double sig = this->osd0_decoding[c] ? -wcol[c] : wcol[c];
                const uint64_t* xr = &X[(size_t)r * W];
                for (int w = 0; w < W; w++) {
                    uint64_t v = xr[w];
                    while (v) { int t = std::countr_zero(v); approx[(w << 6) + t] += sig; v &= v - 1; }
                }
            }
            auto single = [&](int j) {
                std::vector<uint8_t> v = this->osd0_decoding;
                for (int r = 0; r < R; r++) if ((X[(size_t)r * W + (j >> 6)] >> (j & 63)) & 1) v[pivcol[r]] ^= 1;
                v[N[j]] = 1;
                return v;
            };
            // exact pass over the candidate sequence: osd0, singles 0..K-1, pairs (i < j < osd_order)
            double amin = osd0_weight;
            for (int j = 0; j < K; j++) amin = std::min(amin, approx[j]);
            const double tol = 1e-4 + 1e-9 * std::fabs(amin);
            double best = osd0_weight;
            this->osdw_decoding = this->osd0_decoding;
            for (int j = 0; j < K; j++) {
                if (approx[j] > amin + tol) continue;
                auto v = single(j);
                double w = exact_weight(v);
                if (w < best) { best = w; this->osdw_decoding = v; }
            }
            // pairs among the first osd_order non-pivot columns; with fewer than osd_order non-pivot columns, pairs
            // reaching past them add no candidate the sweep has not already tried
            const int order = std::min(this->osd_order, K);
            for (int i = 0; i < order; i++) {
                for (int j = i + 1; j < order; j++) {
                    std::vector<uint8_t> v = this->osd0_decoding;
                    for (int r = 0; r < R; r++) {
                        int bi = (X[(size_t)r * W + (i >> 6)] >> (i & 63)) & 1, bj = (X[(size_t)r * W + (j >> 6)] >> (j & 63)) & 1;
                        if (bi ^ bj) v[pivcol[r]] ^= 1;
                    }
                    v[N[i]] = 1; v[N[j]] = 1;
                    double w = exact_weight(v);
                    if (w < best) { best = w; this->osdw_decoding = v; }
                }
            }
        }

        std::vector<uint8_t>& decode(std::vector<uint8_t>& syndrome, std::vector<double>& log_prob_ratios) {

            ldpc::sort::soft_decision_col_sort(log_prob_ratios, this->column_ordering,bit_count);

            if(this->osd_order == 0){
                if (this->osd0_dense(syndrome)) {
                    this->osdw_decoding = this->osd0_decoding;
                    return this->osd0_decoding;
                }
                this->osd0_decoding = this->osdw_decoding =  this->LuDecomposition->fast_solve(syndrome,this->column_ordering);
                return this->osd0_decoding;
            }

            if (this->osd_method == COMBINATION_SWEEP && this->dense_combination_sweep(syndrome)) {
                return this->osdw_decoding;
            }
            //row reduce the matrix according to the new column ordering
            this->LuDecomposition->rref(false,true,this->column_ordering);

            // find the OSD0 solution
            this->osd0_decoding = this->osdw_decoding = LuDecomposition->lu_solve(syndrome);

            // this->osd0_decoding = this->LuDecomposition->fast_solve(syndrome,this->column_ordering);

            // if(osd_order==0){
            //     return this->osd0_decoding;
            // }

            double candidate_weight, osd_min_weight;

            osd_min_weight=0;
            for(int i=0; i<this->pcm.n; i++){
                if(this->osd0_decoding[i]==1){
                    osd_min_weight+=log(1/this->channel_probabilities[i]);
                }
            }

            std::vector<int> non_pivot_columns;
            std::vector<ldpc::gf2sparse::GF2Entry*> delete_entries;
            for(int i = this->LuDecomposition->rank; i<this->pcm.n; i++){
                int col = this->LuDecomposition->cols[i];
                non_pivot_columns.push_back(col);
                
                for(auto& e: this->LuDecomposition->U.iterate_column(col)){
                    delete_entries.push_back(&e);
                }
            }

            for (auto e: delete_entries){
                this->LuDecomposition->U.remove(*e);
            }
            
            for(auto& candidate_string: this->osd_candidate_strings){

                auto t_syndrome = syndrome;
                int col_index = 0;
                for(int col: non_pivot_columns){
                    if(candidate_string[col_index]==1){
                        for(auto& e: this->pcm.iterate_column(col)){
                            t_syndrome[e.row_index] ^= 1;
                        }
                    }
                    col_index++;
                }

                auto candidate_solution = this->LuDecomposition->lu_solve(t_syndrome);
                for(int i=0; i<k; i++){
                    candidate_solution[non_pivot_columns[i]]=candidate_string[i];
                }
                candidate_weight=0;
                for(int i=0; i<this->pcm.n; i++){
                    if(candidate_solution[i]==1){
                        candidate_weight+=log(1/this->channel_probabilities[i]);
                    }
                }
                if(candidate_weight<osd_min_weight){
                    osd_min_weight = candidate_weight;
                    
                    this->osdw_decoding = candidate_solution;

                }

            }

            return this->osdw_decoding;

        }

};

}//end osd namespace


#endif