# Set the minimum version of CMake required to build this project
cmake_minimum_required(VERSION 3.10)

# Enable the testing functionality provided by CMake
enable_testing()

# Define the main project, its description, and the languages used
project(main
    DESCRIPTION "ldpc - Software for classical and quantum low density parity check codes"
    LANGUAGES CXX)

# Set the C++ standard to C++20 for this project
set(CMAKE_CXX_STANDARD 20)

# Enable OpenMP support for parallel programming
# SET(CMAKE_CXX_FLAGS  "-fopenmp")

# Ensure that the necessary C++11 features are available, as required by the RapidCSV library
set(CMAKE_CXX_STANDARD_REQUIRED ON)

### Library Configuration Section ###

# Define an interface library named 'ldpc' which only includes headers
add_library(ldpc INTERFACE)

# Add specific directories to the include path for any target that links with the 'ldpc' library
target_include_directories(ldpc INTERFACE src_cpp include/robin_map include/rapidcsv)

### Executable Configuration Section ###

# Add an executable target named 'main', which is built from the 'main.cpp' source file
add_executable(main cpp_example/main.cpp)

# Link the 'ldpc' library to the 'main' executable target
target_link_libraries(main ldpc)

### GoogleTest Configuration Section ###

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

### Test Configuration Section ###

# Define a separate project for the tests, its description, and the languages used
project(cpp_test
    DESCRIPTION "Tests for ldpc"
    LANGUAGES CXX)

# Include the GoogleTest library in the project
include(GoogleTest)

# List of test executables
set(TEST_EXECUTABLES
    TestBPDecoder
    TestOsdDecoder
    TestSoftInfo
    TestFlipDecoder
    TestGf2Codes
    TestLsd
    TestSparseMatrix
    TestSparseMatrixUtil
    TestGF2Sparse
    TestGF2RowReduce
    TestGf2Linalg
    TestGf2Dense
    TestGf2DenseApplications
    TestUtil
    TestRng
    TestUnionFind
)

# Loop through the list of test executables to:
# - Add each executable target
# - Link the necessary libraries (gtest_main and ldpc)
# - Discover and register the tests with CTest
foreach(TEST ${TEST_EXECUTABLES})
    add_executable(${TEST} cpp_test/${TEST}.cpp)  # Add the test executable
    target_link_libraries(${TEST} gtest_main ldpc)  # Link the libraries
    gtest_discover_tests(${TEST})  # Discover and register the tests
endforeach()
