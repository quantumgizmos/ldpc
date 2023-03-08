cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_FLAGS  "-fopenmp")
set (CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

# set(CMAKE_BUILD_TYPE "-O3")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

### The following two lines define an interface library (A header-only library)
add_library(ldpc2 INTERFACE)
target_include_directories(ldpc2 INTERFACE src_cpp include/robin_map include/rapidcsv)  

project(main)
add_executable(main cpp_example/main.cpp)
target_link_libraries(main ldpc2) 


## Install GoogleTest
include(FetchContent)
FetchContent_Declare(
  GoogleTest
  URL ${CMAKE_CURRENT_SOURCE_DIR}/cpp_test/google_test.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(GoogleTest)


project(cpp_test)

# Include GoogleTest
include(GoogleTest)

# Add test executables
add_executable(TestSparseMatrix cpp_test/TestSparseMatrix.cpp)
add_executable(TestSparseMatrixUtil cpp_test/TestSparseMatrixUtil.cpp)
add_executable(TestGF2Sparse cpp_test/TestGF2Sparse.cpp)

target_link_libraries(TestSparseMatrix gtest_main ldpc2)
target_link_libraries(TestSparseMatrixUtil gtest_main ldpc2)
target_link_libraries(TestGF2Sparse gtest_main ldpc2)

# Discover and run tests
enable_testing()
gtest_discover_tests(TestSparseMatrix)
gtest_discover_tests(TestSparseMatrixUtil)
gtest_discover_tests(TestGF2Sparse)


