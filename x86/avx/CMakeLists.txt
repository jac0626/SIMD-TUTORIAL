

add_executable(avx_memory_operations memory_operations.cpp)
target_compile_options(avx_memory_operations PRIVATE -mavx)
