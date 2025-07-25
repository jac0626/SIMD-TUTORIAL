#!/bin/bash

# Update package lists
sudo apt-get update

# Install essential build tools
sudo apt-get install -y build-essential cmake

# Install cross-compilation toolchain for ARM64
sudo apt-get install -y crossbuild-essential-arm64

# Install QEMU for running ARM64 binaries
sudo apt-get install -y qemu-user-static
sudo apt-get install -y qemu-user

echo "Setup complete. You can now cross-compile for ARM64."
 