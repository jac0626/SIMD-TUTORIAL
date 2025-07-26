cmake_minimum_required(VERSION 3.10)
project(SIMD_Tutorial)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(common)

add_subdirectory(sse)
add_subdirectory(avx)
add_subdirectory(avx2)
add_subdirectory(avx512)