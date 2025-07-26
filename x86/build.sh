#!/bin/bash

# Create a build directory if it doesn't exist
mkdir -p build

# Navigate into the build directory
cd build

# Run CMake to configure the project and generate Makefiles
c_compiler=/usr/bin/gcc
cxx_compiler=/usr/bin/g++
cmake -DCMAKE_C_COMPILER=${c_compiler} -DCMAKE_CXX_COMPILER=${cxx_compiler} ..

# Run make to build the project
make -j$(nproc)
