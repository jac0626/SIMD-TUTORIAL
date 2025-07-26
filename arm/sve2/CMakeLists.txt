add_executable(sve2_example sve2_example.cpp)
target_compile_options(sve2_example PRIVATE -march=armv8-a+sve2)

add_executable(sve2_memory_load_store memory_load_store.cpp)
target_compile_options(sve2_memory_load_store PRIVATE -march=armv8-a+sve2)
