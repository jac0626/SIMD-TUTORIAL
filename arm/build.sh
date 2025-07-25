#!/bin/bash

# Create a build directory if it doesn't exist
mkdir -p build
cd build

# Run CMake with the toolchain file
cmake .. -DCMAKE_TOOLCHAIN_FILE=../aarch64-linux-gnu.cmake

# Build the project
make
