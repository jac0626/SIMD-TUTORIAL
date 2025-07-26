add_executable(sve_example sve_example.cpp)
target_compile_options(sve_example PRIVATE -march=armv8-a+sve)

add_executable(sve_memory_load_store memory_load_store.cpp)
target_compile_options(sve_memory_load_store PRIVATE -march=armv8-a+sve2)
