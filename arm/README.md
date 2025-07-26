# ARM SIMD Programming Tutorial

This project contains C++ examples for learning SIMD (Single Instruction, Multiple Data) programming on the ARM architecture. It covers NEON, SVE, and SVE2 instruction sets.

## Directory Structure

- `neon/`: Contains examples using the NEON instruction set.
- `sve/`: Contains examples using the Scalable Vector Extension (SVE).
- `sve2/`: Contains examples using the Scalable Vector Extension 2 (SVE2).
- `build/`: This directory is created by the build script and contains all the compiled binaries.

## How to Build

A convenience script `build.sh` is provided to compile all the examples.

```bash
# Make sure the script is executable
chmod +x build.sh

# Run the build script
./build.sh
```

This will create the `build` directory and place the compiled executables inside the corresponding subdirectories (e.g., `build/neon/`, `build/sve/`).

## How to Run

After a successful build, you can run the individual examples. For instance, to run the NEON memory load/store example:

```bash
./build/neon/neon_memory_load_store
```

To run the SVE example:

```bash
./build/sve/sve_memory_load_store
```
