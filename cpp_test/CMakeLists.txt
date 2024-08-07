### GoogleTest Configuration Section ###

# Download the GoogleTest library as a subdirectory of the current project
include(FetchContent)
FetchContent_Declare(
	GoogleTest
	URL ${CMAKE_CURRENT_SOURCE_DIR}/google_test.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Make the GoogleTest library available to the project
FetchContent_MakeAvailable(GoogleTest)

### Test Configuration Section ###

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
	add_executable(${TEST} ${TEST}.cpp)  # Add the test executable
	target_link_libraries(${TEST} PRIVATE gtest_main ldpc)  # Link the libraries
	gtest_discover_tests(${TEST})  # Discover and register the tests
endforeach()
