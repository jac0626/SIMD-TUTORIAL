

add_executable(avx2_memory_operations memory_operations.cpp)
target_compile_options(avx2_memory_operations PRIVATE -mavx2)
