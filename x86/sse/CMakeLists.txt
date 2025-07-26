add_executable(sse_load_store load_store.cpp)
target_compile_options(sse_load_store PRIVATE -msse)