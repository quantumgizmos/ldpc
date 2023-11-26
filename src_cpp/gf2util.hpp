#include <iostream>
#include <vector>
#include <cmath>

namespace ldpc{
namespace gf2util{

    /**
     * @brief Calculate the number of bits required to represent an integer in binary.
     *
     * @param num Integer whose bit-length is to be calculated.
     * @return int Number of bits required to represent the integer in binary.
     */
    int number_of_bits_binary_rep(int num) {
        if (num == 0) {
            return 1;  // Special case: '0' is represented by 1 bit
        }
        if (num < 0) {
            // Negative numbers are represented in two's complement form
            // Here we only consider the magnitude for simplicity
            num = -num;
        }

        return static_cast<int>(std::floor(std::log2(num))) + 1;
    }



    std::vector<int> int_to_sparse_binary(int num) {
        std::vector<int> binaryVector;

        // Special case: When num is zero, return a single zero in the vector
        if (num == 0) {
            return std::vector<int>{};
        }

        // Loop to fill the vector with binary representation
        int index = number_of_bits_binary_rep(num)-1;
        while (num > 0) {
            binaryVector.push_back(num % 2);
            num /= 2;
            index--;
        }

        return binaryVector;
    }

} //end namespace gf2util
} //end namespace ldpc

