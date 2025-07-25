cmake_minimum_required(VERSION 3.10)
project(arm-simd-tutorial CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(neon)
add_subdirectory(sve)
add_subdirectory(sve2)
