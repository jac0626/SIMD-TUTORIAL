

add_executable(sse_load_instructions load_instructions.cpp)
target_compile_options(sse_load_instructions PRIVATE -msse -msse2 -msse4.1)

add_executable(sse_store_instructions store_instructions.cpp)
target_compile_options(sse_store_instructions PRIVATE -msse -msse2 -msse4.1)