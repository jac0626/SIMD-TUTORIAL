add_executable(sve_example sve_example.cpp)
target_compile_options(sve_example PRIVATE -march=armv8-a+sve)
