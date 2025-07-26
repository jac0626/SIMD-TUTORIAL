# x86 SIMD Tutorial

This project is a tutorial for learning x86 SIMD programming using C++ with SSE, AVX, AVX2, and AVX512 instruction sets.

---

# x86 SIMD 教程

本项目是一个用于学习 x86 SIMD C++ 编程的教程，涵盖 SSE、AVX、AVX2 和 AVX512 指令集。

## Directory Structure (目录结构)

```
.
├── sse/         # Code for SSE examples (SSE 示例代码)
├── avx/         # Code for AVX examples (AVX 示例代码)
├── avx2/        # Code for AVX2 examples (AVX2 示例代码)
├── avx512/      # Code for AVX-512 examples (AVX-512 示例代码)
├── common/      # Common utility functions (通用工具函数)
├── build.sh     # Script to build all examples (编译所有示例的脚本)
├── run.sh       # Script to run an example with QEMU (使用 QEMU 运行示例的脚本)
├── CMakeLists.txt # Main CMake configuration file (主 CMake 配置文件)
└── README.md    # This file (本文件)
```

## How to Build (如何编译)

To build all the examples, simply run the `build.sh` script. This script will create a `build` directory, run CMake to generate the Makefiles, and then compile the code.

要编译所有示例，只需运行 `build.sh` 脚本。该脚本会自动创建 `build` 目录，运行 CMake 生成 Makefile，然后编译代码。

```bash
./build.sh
```

## How to Run (如何运行)

Use the `run.sh` script followed by the name of the executable you want to run. The script will find the executable in the `build` directory and run it using QEMU.

使用 `run.sh` 脚本并跟上你想要运行的可执行文件的名称。该脚本会在 `build` 目录中查找并使用 QEMU 运行它。

**Example (示例):**

```bash
# Run the SSE load/store example
# 运行 SSE 的加载/存储示例
./run.sh sse_load_store

# Run the AVX load/store example
# 运行 AVX 的加载/存储示例
./run.sh avx_load_store
```
