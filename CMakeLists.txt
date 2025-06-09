# Set the minimum version of CMake required to build this project
cmake_minimum_required(VERSION 3.12...3.30)

# Define the main project, its description, and the languages used
project(ldpc
    DESCRIPTION "ldpc - Software for classical and quantum low density parity check codes"
    LANGUAGES CXX)

# Enable OpenMP support for parallel programming
# SET(CMAKE_CXX_FLAGS  "-fopenmp")

# Ensure that the necessary C++11 features are available, as required by the RapidCSV library
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# check if this is the master project or used via add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
	set(LDPC_MASTER_PROJECT ON)
else()
	set(LDPC_MASTER_PROJECT OFF)
endif()
option(BUILD_LDPC_TESTS "Also build tests for the LDPC project" ${LDPC_MASTER_PROJECT})

### Library Configuration Section ###

# Define an interface library named 'ldpc' which only includes headers
add_library(ldpc INTERFACE)

# Add specific directories to the include path for any target that links with the 'ldpc' library
target_include_directories(ldpc INTERFACE src_cpp include/robin_map include/rapidcsv)

# Set the required C++ standard
target_compile_features(ldpc INTERFACE cxx_std_20)

if (LDPC_MASTER_PROJECT)
	add_subdirectory(cpp_example)
endif()

if(BUILD_LDPC_TESTS)
	# Add test code
	enable_testing()
	include(GoogleTest)
	add_subdirectory(cpp_test)
endif()
