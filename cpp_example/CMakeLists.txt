### Executable Configuration Section ###

# Add an executable target named 'ldpc-main', which is built from the 'main.cpp' source file
add_executable(ldpc-main main.cpp)

# Link the 'ldpc' library to the 'ldpc-main' executable target
target_link_libraries(ldpc-main PRIVATE ldpc)
