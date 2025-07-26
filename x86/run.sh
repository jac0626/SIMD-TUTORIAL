#!/bin/bash

# Check if an example name is provided
if [ -z "$1" ]; then
    echo "Usage: ./run.sh <example_name>"
    echo "Example: ./run.sh sse_load_store"
    exit 1
fi

EXAMPLE_NAME=$1
EXAMPLE_PATH=""

# Find the example path
if [ -f "./build/sse/$EXAMPLE_NAME" ]; then
    EXAMPLE_PATH="./build/sse/$EXAMPLE_NAME"
fi

if [ -f "./build/avx/$EXAMPLE_NAME" ]; then
    EXAMPLE_PATH="./build/avx/$EXAMPLE_NAME"
fi

if [ -f "./build/avx2/$EXAMPLE_NAME" ]; then
    EXAMPLE_PATH="./build/avx2/$EXAMPLE_NAME"
fi

if [ -f "./build/avx512/$EXAMPLE_NAME" ]; then
    EXAMPLE_PATH="./build/avx512/$EXAMPLE_NAME"
fi

if [ -z "$EXAMPLE_PATH" ]; then
    echo "Error: Example '$EXAMPLE_NAME' not found in ./build/{sse,avx,avx2,avx512}/"
    exit 1
fi

# Run the example with QEMU, specifying a CPU with AVX512 support
./sde-external-9.58.0-2025-06-16-lin/sde  -icl -- $EXAMPLE_PATH