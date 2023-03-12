# Set the minimum version of CMake required
cmake_minimum_required(VERSION 3.10)

### Project Configuration
# Define the main project
project(main)

# Set the C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)

# Set the C standard to C11
set (CMAKE_C_STANDARD 11)

# Enable OpenMP support for parallel programming
SET(CMAKE_CXX_FLAGS  "-fopenmp")

# Enable C++11 features required by RapidCSV library
set(CMAKE_CXX_STANDARD_REQUIRED ON)

### Library Configuration
# Define an interface library named ldpc2 that only includes headers
add_library(ldpc2 INTERFACE)

# Add the src_cpp and include directories to the include path for any target that links with ldpc2
target_include_directories(ldpc2 INTERFACE src_cpp src_cpp/include/rapidcsv src_cpp/include/robin_map)

### Executable Configuration
# Add an executable named main, which depends on the main.cpp file
add_executable(main cpp_example/main.cpp)

# Link the ldpc2 library to the main executable
target_link_libraries(main ldpc2)

## Install GoogleTest
### Library Configuration
# Download the GoogleTest library as a subdirectory of the current project
include(FetchContent)
FetchContent_Declare(
  GoogleTest
  URL ${CMAKE_CURRENT_SOURCE_DIR}/cpp_test/google_test.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Make the GoogleTest library available to the project
FetchContent_MakeAvailable(GoogleTest)

### Test Configuration
# Define the cpp_test project
project(cpp_test)

# Include the GoogleTest library in the project
include(GoogleTest)

# Add test executables for SparseMatrix, SparseMatrixUtil, GF2Sparse, and GF2RowReduce
add_executable(TestSparseMatrix cpp_test/TestSparseMatrix.cpp)
add_executable(TestSparseMatrixUtil cpp_test/TestSparseMatrixUtil.cpp)
add_executable(TestGF2Sparse cpp_test/TestGF2Sparse.cpp)
add_executable(TestGF2RowReduce cpp_test/TestGF2RowReduce.cpp)

# Link the ldpc2 and gtest_main libraries to the test executables
target_link_libraries(TestSparseMatrix gtest_main ldpc2)
target_link_libraries(TestSparseMatrixUtil gtest_main ldpc2)
target_link_libraries(TestGF2Sparse gtest_main ldpc2)
target_link_libraries(TestGF2RowReduce gtest_main ldpc2)

# Discover and run tests using GoogleTest
enable_testing()
gtest_discover_tests(TestSparseMatrix)
gtest_discover_tests(TestSparseMatrixUtil)
gtest_discover_tests(TestGF2Sparse)
gtest_discover_tests(TestGF2RowReduce)
