cmake_minimum_required(VERSION 3.2)
set(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_FLAGS  "-fopenmp")
set (CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

### The following two lines define an interface library (A header-only library)
add_library(ldpc2 INTERFACE)
target_include_directories(ldpc2 INTERFACE src_cpp include/robin_map)  

project(main)
add_executable(main cpp_example/main.cpp)
target_link_libraries(main ldpc2) 

project(cpp_test)

# Include GoogleTest
include(GoogleTest)
find_package(GTest REQUIRED)

# Add test executables
add_executable(TestSparseMatrix cpp_test/TestSparseMatrix.cpp)
add_executable(TestSparseMatrixUtil cpp_test/TestSparseMatrixUtil.cpp)
add_executable(TestGF2Sparse cpp_test/TestGF2Sparse.cpp)

target_link_libraries(TestSparseMatrix GTest::GTest GTest::Main ldpc2)
target_link_libraries(TestSparseMatrixUtil GTest::GTest GTest::Main ldpc2)
target_link_libraries(TestGF2Sparse GTest::GTest GTest::Main ldpc2)

# Discover and run tests
enable_testing()
gtest_discover_tests(TestSparseMatrix)
gtest_discover_tests(TestSparseMatrixUtil)
gtest_discover_tests(TestGF2Sparse)

