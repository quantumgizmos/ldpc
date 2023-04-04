/**
 * @file rng.hpp
 * @brief Defines the rng::RandomNumberGenerator class, which generates random double values between 0 and 1
 */

#ifndef RNG_HPP
#define RNG_HPP

#include <random>
#include <chrono> // for std::chrono::system_clock

namespace rng {

    /**
     * @brief Generates random double values between 0 and 1 using a Mersenne Twister random number generator
     *
     * The RandomNumberGenerator class creates a Mersenne Twister random number generator, and provides a method
     * for generating random double values between 0 and 1 using a uniform distribution.
     */
    class RandomNumberGenerator {
    public:
        /**
         * @brief Constructs a new RandomNumberGenerator object with an optional seed
         *
         * If no seed is specified, the generator is seeded using the system clock.
         *
         * @param seed The seed value to use for the random number generator
         */
        RandomNumberGenerator(int seed = 0) : gen(seed) {
            if (seed == 0) {
                gen.seed(std::chrono::system_clock::now().time_since_epoch().count());
            }
        }

        ~RandomNumberGenerator() = default;
        
        /**
         * @brief Generates a new random double value between 0 and 1
         *
         * @return A random double value between 0 and 1
         */
        double random_double() {
            // Create a uniform distribution between 0 and 1
            std::uniform_real_distribution<double> dis(0.0, 1.0);
            
            // Generate a random number between 0 and 1
            double random_num = dis(gen);
            
            return random_num;
        }
        
    private:
        std::mt19937 gen; /**< The Mersenne Twister random number generator used by the class */
    };

} // namespace rng

#endif // RNG_HPP
