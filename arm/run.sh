#!/bin/bash

# Check if an example name is provided
if [ -z "$1" ]; then
    echo "Usage: ./run.sh <example_name>"
    echo "Example: ./run.sh neon_example"
    exit 1
fi

EXAMPLE_NAME=$1
EXAMPLE_PATH=""

# Find the example path
if [ -f "./build/neon/$EXAMPLE_NAME" ]; then
    EXAMPLE_PATH="./build/neon/$EXAMPLE_NAME"
fi

if [ -f "./build/sve/$EXAMPLE_NAME" ]; then
    EXAMPLE_PATH="./build/sve/$EXAMPLE_NAME"
fi

if [ -f "./build/sve2/$EXAMPLE_NAME" ]; then
    EXAMPLE_PATH="./build/sve2/$EXAMPLE_NAME"
fi

if [ -z "$EXAMPLE_PATH" ]; then
    echo "Error: Example '$EXAMPLE_NAME' not found."
    exit 1
fi

# Run the example with QEMU
QEMU_LD_PREFIX=/usr/aarch64-linux-gnu qemu-aarch64-static -cpu max $EXAMPLE_PATH
