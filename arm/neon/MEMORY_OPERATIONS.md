# Understanding NEON Load and Store Operations

Efficiently moving data between memory and NEON registers is fundamental to SIMD programming. NEON provides a rich set of intrinsics for this purpose. The key to mastering them is to recognize the consistent patterns in their naming conventions.

All load and store intrinsics follow a general structure:

`v[ld|st][N]q?_[lane?]_[type]_[xM]`

-   **`v`**: Standard prefix for all NEON intrinsics.
-   **`ld` or `st`**: Specifies the operation type: **load** from memory or **store** to memory.
-   **`[N]`**: A number (2, 3, or 4) indicating the number of vectors to de-interleave or interleave. If absent, it's a standard operation on a single vector structure.
-   **`q`**: An optional flag. If present, it signifies a 128-bit operation (`uint8x16_t`). If absent, it's a 64-bit operation (`uint8x8_t`).
-   **`_lane`**: An optional modifier indicating that the operation targets a single lane within a vector.
-   **`_dup`**: (Load only) An optional modifier that loads a single value and duplicates it across all lanes of a vector.
-   **`_[type]`**: The data type being handled (e.g., `u8`, `f32`, `s16`).
-   **`_xM`**: (Load/Store `vld1`/`vst1` only) An optional suffix indicating a multi-vector operation, loading or storing `M` entire vectors contiguously.

---

## Core Concepts & Examples

Below are the primary patterns for memory operations. Understanding these will allow you to interpret almost any load/store intrinsic.

### 1. Basic Load/Store (`vld1` / `vst1`)

This is the most common operation. It loads a full vector's worth of data from a contiguous memory block or stores a full vector back to memory.

-   **`vld1q_u8`**: Loads 16 `uint8_t` values from memory into a `uint8x16_t` register.
-   **`vst1q_u8`**: Stores 16 `uint8_t` values from a `uint8x16_t` register into memory.

**Use Case**: Standard processing of arrays and data buffers.

### 2. Load and Duplicate (`vld1_dup`)

This loads a *single* element from memory and broadcasts (duplicates) it to every lane of the destination vector.

-   **`vld1q_dup_u8`**: Reads one `uint8_t` and creates a `uint8x16_t` where all 16 lanes contain that value.

**Use Case**: Preparing a constant value for arithmetic operations (e.g., multiplying an entire vector by a single number loaded from memory).

### 3. Load/Store a Single Lane (`vld1_lane` / `vst1_lane`)

This reads a single value from memory into one specific lane of an *existing* vector, or stores a single lane from a vector out to memory.

-   **`vld1q_lane_u8`**: Loads one `uint8_t` into a specified lane of a `uint8x16_t` without affecting the other 15 lanes.
-   **`vst1q_lane_u8`**: Stores one lane from a `uint8x16_t` to a single `uint8_t` memory location.

**Use Case**: Gathering scattered data into a single vector, or scattering results back to disparate memory locations.

### 4. De-interleaving Load / Interleaving Store (`vldN` / `vstN`)

These are powerful instructions for handling structured data, like multi-channel images (RGB, RGBA).

-   **`vld3q_u8`**: Reads from an interleaved memory buffer (e.g., `R1, G1, B1, R2, G2, B2, ...`) and separates the data into three distinct `uint8x16_t` vectors: one for all R values, one for all Gs, and one for all Bs.
-   **`vst3q_u8`**: Performs the reverse, taking three separate vectors (R, G, B) and writing them to memory in an interleaved format.

**Use Case**: Essential for processing multi-channel data. It allows you to perform operations on each channel independently (e.g., adjusting the brightness of the R channel only) and then combine them back.

### 5. Multi-Vector Load/Store (`vld1_xM` / `vst1_xM`)

This is a simple bulk operation. It loads or stores multiple *entire* vectors from/to a contiguous block of memory.

-   **`vld1q_u8_x2`**: Loads 32 bytes from memory into two separate `uint8x16_t` vectors.
-   **`vst1q_u8_x2`**: Stores two `uint8x16_t` vectors into a contiguous 32-byte block of memory.

**Use Case**: Loading or storing larger blocks of data when you need to operate on several vectors at once.

**Key Distinction**: Do not confuse `vld2q_u8` (de-interleave) with `vld1q_u8_x2` (multi-vector load). `vld2` separates `A1, B1, A2, B2, ...` into two vectors `{A...}` and `{B...}`. `vld1_x2` simply loads `{A1..A16}` and `{B1..B16}` from a contiguous block.
