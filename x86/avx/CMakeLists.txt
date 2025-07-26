add_executable(avx_load_store load_store.cpp)
target_compile_options(avx_load_store PRIVATE -mavx)