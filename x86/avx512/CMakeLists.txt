add_executable(avx512_load_store load_store.cpp)
target_compile_options(avx512_load_store PRIVATE -mavx512f  -mavx512dq -mavx512cd -mavx512bw -mavx512vl)