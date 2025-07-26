# SIMD C++ Tutorial

This project provides a collection of C++ examples for learning SIMD (Single Instruction, Multiple Data) programming on both x86 and ARM architectures.

## Architectures

*   **x86**: Covers SSE, AVX, AVX2, and AVX512 instruction sets.
*   **ARM**: Covers NEON, SVE, and SVE2 instruction sets.

## Directory Structure

```
.
├── arm/         # ARM SIMD examples (NEON, SVE, SVE2)
├── x86/         # x86 SIMD examples (SSE, AVX, AVX2, AVX512)
├── LICENSE      # Project license
└── README.md    # This file
```

## How to Build

Each architecture has its own build script. Navigate to the respective directory and run the `build.sh` script. To simulate arm instructions on x86（I have test on ubuntu-22.04）, you need run arm/setup_env.sh before building

### x86

```bash
cd x86
./build.sh
```

### ARM

```bash
cd arm
./build.sh
```

## How to Run

Similarly, each architecture has its own script to run the compiled examples.

### x86

The `run.sh` script in the `x86` directory uses the Intel Software Development Emulator (SDE) to run the examples.

```bash
cd x86

# Run the SSE load/store example
./run.sh sse_load_store

# Run the AVX load/store example
./run.sh avx_load_store
```

### ARM

The `run.sh` script in the `arm` directory uses QEMU to emulate an AArch64 environment.

```bash
cd arm

# Run the NEON example
./run.sh neon_example

# Run the SVE example
./run.sh sve_example
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
