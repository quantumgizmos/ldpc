#ifndef Gf2Codes_hpp
#define Gf2Codes_hpp

#include <vector>
#include <memory>

#include "gf2sparse.hpp"
#include "bp.hpp"
namespace ldpc::gf2codes{

/**
 * Creates the parity check matrix of a repetition code of length n.
 *
 * @tparam T The type of the entries in the sparse matrix. Default is BpEntry.
 * @param n The length of the repetition code.
 * @return A shared pointer to a GF2Sparse<T> matrix representing the parity check matrix.
 */
template <typename T = ldpc::gf2sparse::GF2Entry>
ldpc::gf2sparse::GF2Sparse<T> rep_code(int n){
    // Create a shared pointer to a new GF2Sparse<T> matrix with n-1 rows and n columns.
    auto pcm = ldpc::gf2sparse::GF2Sparse<T>(n-1, n);
    // Fill in the entries of the matrix corresponding to the repetition code.
    for(int i=0; i<n-1; i++){
        pcm.insert_entry(i, i);    // Insert a 1 in the diagonal position.
        pcm.insert_entry(i, i+1);  // Insert a 1 in the position to the right of the diagonal.
    }
    // Return the shared pointer to the matrix.
    return pcm;
}


/**
 * Creates the parity check matrix of a cyclic repetition code of length n.
 *
 * @tparam T The type of the entries in the sparse matrix. Default is BpEntry.
 * @param n The length of the cyclic repetition code.
 * @return A shared pointer to a GF2Sparse<T> matrix representing the parity check matrix.
 */
template <typename T = ldpc::gf2sparse::GF2Entry>
ldpc::gf2sparse::GF2Sparse<T> ring_code(int n) {
    // Create a shared pointer to a new GF2Sparse<T> matrix with n-1 rows and n columns.
    auto pcm = ldpc::gf2sparse::GF2Sparse<T>(n, n);
    // Fill in the entries of the matrix corresponding to the cyclic repetition code.
    for (int i = 0; i < n; i++) {
        pcm.insert_entry(i, i);    // Insert a 1 in the diagonal position.
        pcm.insert_entry(i, (i + 1) % n);  // Insert a 1 in the position to the right of the diagonal, with wraparound.
    }
    // Return the shared pointer to the matrix.
    return pcm;
}

/**
 * Creates the parity check matrix of a Hamming code with given rank.
 *
 * @tparam T The type of the entries in the sparse matrix. Default is BpEntry.
 * @param r The rank of the Hamming code, where the block length is 2^r - 1.
 * @return A shared pointer to a GF2Sparse<T> matrix representing the parity check matrix.
 */
template <typename T = ldpc::gf2sparse::GF2Entry>
ldpc::gf2sparse::GF2Sparse<T> hamming_code(int r) {
    // Calculate the block length and the number of data bits.
    int n = (1 << r) - 1;
    int k = n - r;

    // Create a shared pointer to a new GF2Sparse<T> matrix with r rows and n columns.
    auto pcm = ldpc::gf2sparse::GF2Sparse<T>(r, n);

    // Fill in the entries of the matrix corresponding to the Hamming code.
    for (int i = 0; i < n; i++) {
        int binary = i + 1;
        for (int j = 0; j < r; j++) {
            if (binary & (1 << j)) {
                pcm.insert_entry(j, i);
            }
        }
    }

    // Return the shared pointer to the matrix.
    return pcm;
}



} //end namespace gf2codes

#endif /* Gf2Codes_hpp */
