# AVX Memory Operations

This document covers the essential memory operations introduced or enhanced by the AVX (Advanced Vector Extensions) instruction set. AVX extends SIMD capabilities to 256-bit YMM registers and introduces more powerful and flexible instructions.

---

## Key Enhancements from SSE

- **256-bit Vectors**: YMM registers hold twice the data of XMM registers (e.g., 8 floats or 4 doubles).
- **Three-Operand Instructions**: Most AVX instructions can be non-destructive, specifying two source operands and one destination operand (`dest = src1 op src2`), improving code clarity and efficiency.
- **Masked Operations**: AVX introduces a more robust and explicit way to conditionally load or store elements using a mask register.

---

## Load and Store Instructions

### 1. Aligned vs. Unaligned Access

Similar to SSE, AVX provides instructions for both aligned and unaligned memory access. For AVX, alignment is 32 bytes for YMM registers.

- **Aligned**: `_mm256_load_ps`, `_mm256_store_pd`, `_mm256_load_si256`, etc.
- **Unaligned**: `_mm256_loadu_ps`, `_mm256_storeu_pd`, `_mm256_loadu_si256`, etc.

**Example:**
```cpp
// Aligned access requires a 32-byte boundary
alignas(32) float data[8];
__m256 vec = _mm256_load_ps(data);

// Unaligned access works on any address
_mm256_storeu_ps(&data[1], vec);
```

### 2. Broadcast Operations

AVX provides a rich set of broadcast instructions to load a single data element from memory and duplicate it across all elements of a YMM register.

- **Broadcast a single element**: `_mm256_broadcast_ss` (float), `_mm256_broadcast_sd` (double).
- **Broadcast a 128-bit lane**: `_mm256_broadcast_ps` or `_mm256_broadcast_pd` can load a 128-bit XMM vector from memory and duplicate it into both the lower and upper halves of a 256-bit YMM vector.

**Example:**
```cpp
double val = 42.0;
// vec becomes [42.0, 42.0, 42.0, 42.0]
__m256d vec = _mm256_broadcast_sd(&val);

alignas(16) float f_vals[4] = {1.f, 2.f, 3.f, 4.f};
// f_vec becomes [1,2,3,4,1,2,3,4]
__m256 f_vec = _mm256_broadcast_ps((__m128*)f_vals);
```

### 3. Masked Load and Store

This is a major feature of AVX. It allows you to conditionally load or store elements based on a mask. In the mask register, the most significant bit of each element acts as a selector (1 = perform operation, 0 = do not).

- **Masked Load**: `_mm256_maskload_ps`, `_mm256_maskload_pd`
- **Masked Store**: `_mm256_maskstore_ps`, `_mm256_maskstore_pd`

**Example:**
```cpp
alignas(32) double data[4] = {10, 20, 30, 40};
alignas(32) double dest[4] = {0};

// Create a mask to load/store the 1st and 4th elements
__m256i mask = _mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF, 0, 0, 0xFFFFFFFFFFFFFFFF);

// Load only the elements specified by the mask; others are zeroed.
// loaded_vec will be [10.0, 0.0, 0.0, 40.0]
__m256d loaded_vec = _mm256_maskload_pd(data, mask);

// Store only the elements from loaded_vec where the mask is set.
// dest will be [10.0, 0.0, 0.0, 40.0]
_mm256_maskstore_pd(dest, mask, loaded_vec);
```

### 4. Streaming (Non-Temporal) Stores

Similar to SSE, AVX provides streaming stores to write data directly to memory, bypassing the cache hierarchy. This is a performance hint for data that is not expected to be read again soon.

- **All types**: `_mm256_stream_ps`, `_mm256_stream_pd`, `_mm256_stream_si256`

**Example:**
```cpp
__m256d vec = _mm256_set1_pd(99.9);
alignas(32) double dest[4];
// Write to dest without polluting the cache
_mm256_stream_pd(dest, vec);
```
