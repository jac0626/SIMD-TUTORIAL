# SSE Memory Operations

This document provides an overview of the fundamental memory load and store operations available in SSE/SSE2. Understanding these operations is crucial for efficiently moving data between memory and XMM registers.

---

## Load Instructions

Load instructions are used to move data from memory into XMM registers.

### 1. Aligned vs. Unaligned Loads

The most fundamental distinction is between aligned and unaligned memory access. Aligned access is faster but requires the memory address to be a multiple of 16 bytes. Unaligned access is more flexible but may incur a performance penalty.

- **Aligned Load**: `_mm_load_ps` (movaps), `_mm_load_pd` (movapd), `_mm_load_si128` (movdqa)
- **Unaligned Load**: `_mm_loadu_ps` (movups), `_mm_loadu_pd` (movupd), `_mm_loadu_si128` (movdqu)
- **Special Unaligned Load**: `_mm_lddqu_si128` is a special instruction for integers that can be more efficient than `movdqu` when crossing cache line boundaries.

**Example:**
```cpp
// Aligned: Crashes if fa_data is not 16-byte aligned
alignas(16) float fa_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
__m128 vec_a = _mm_load_ps(fa_data);

// Unaligned: Works on any address
const float* fu_ptr = &some_float_array[1];
__m128 vec_u = _mm_loadu_ps(fu_ptr);
```

### 2. Broadcast / Splat Loads

These instructions load a single value from memory and duplicate it across all elements of the XMM register.

- **Float**: `_mm_load1_ps`
- **Double**: `_mm_load1_pd` (also `_mm_loaddup_pd`)

**Example:**
```cpp
float val = 3.14f;
// vec will contain [3.14, 3.14, 3.14, 3.14]
__m128 vec = _mm_load1_ps(&val);
```

### 3. Scalar Loads

Scalar loads move a single value into the lowest element of the register, leaving the other elements unmodified or zeroed.

- **Load and Zero Others**: `_mm_load_ss` (float), `_mm_load_sd` (double). Loads one value into the low bits and sets the high bits to zero.

**Example:**
```cpp
float val = 1.23f;
// vec will contain [1.23, 0.0, 0.0, 0.0]
__m128 vec = _mm_load_ss(&val);
```

### 4. Partial Loads

These instructions load data into either the lower or upper half of an XMM register, leaving the other half untouched.

- **Load to Low**: `_mm_loadl_pd` (double), `_mm_loadl_epi64` (integer)
- **Load to High**: `_mm_loadh_pd` (double)

**Example:**
```cpp
__m128d vec = _mm_set_pd(99.0, 88.0); // vec is [88.0, 99.0]
double val = 10.0;
// vec becomes [10.0, 99.0]
vec = _mm_loadl_pd(vec, &val);
```

### 5. Streaming (Non-Temporal) Load

This is a special hint to the CPU that the loaded data will not be reused soon. This allows the CPU to bypass the cache, preventing pollution of data that is more important.

- **Integer**: `_mm_stream_load_si128` (movntdqa)

**Example:**
```cpp
alignas(16) int32_t data[4] = {1, 2, 3, 4};
// Load data without putting it into the CPU caches
__m128i vec = _mm_stream_load_si128((__m128i*)data);
```

---

## Store Instructions

Store instructions are used to move data from XMM registers back into memory.

### 1. Aligned vs. Unaligned Stores

Similar to loads, stores have aligned and unaligned versions.

- **Aligned Store**: `_mm_store_ps`, `_mm_store_pd`, `_mm_store_si128`
- **Unaligned Store**: `_mm_storeu_ps`, `_mm_storeu_pd`, `_mm_storeu_si128`

**Example:**
```cpp
__m128 vec = _mm_set1_ps(5.0f);
alignas(16) float dest[4];
_mm_store_ps(dest, vec);
```

### 2. Scalar and Partial Stores

These instructions store only a portion of the register.

- **Scalar Store**: `_mm_store_ss` (float), `_mm_store_sd` (double). Stores only the lowest element.
- **Partial Store**: `_mm_storeh_pd`, `_mm_storel_pd`, `_mm_storel_epi64`. Stores the high or low part of the register.

**Example:**
```cpp
__m128 vec = _mm_setr_ps(1.0f, 2.0f, 3.0f, 4.0f);
float dest = 0.0f;
// dest becomes 1.0f
_mm_store_ss(&dest, vec);
```

### 3. Masked (Conditional) Store

A powerful feature that allows per-element conditional storing without branches. The store for each byte is controlled by the most significant bit of the corresponding byte in a mask register.

- **Integer**: `_mm_maskmoveu_si128`

**Example:**
```cpp
__m128i data = _mm_setr_epi32(10, 20, 30, 40);
// Mask to store the 1st and 3rd integers
__m128i mask = _mm_setr_epi32(0xFFFFFFFF, 0, 0xFFFFFFFF, 0);
alignas(16) int32_t dest[4] = {0};
// dest becomes [10, 0, 30, 0]
_mm_maskmoveu_si128(data, mask, (char*)dest);
```

### 4. Streaming (Non-Temporal) Stores

This instructs the CPU to write data directly to memory, bypassing the cache. This is a performance optimization for data that will not be read again soon, preventing cache pollution.

- **All types**: `_mm_stream_ps`, `_mm_stream_pd`, `_mm_stream_si128`, `_mm_stream_si32`

**Example:**
```cpp
__m128 vec = _mm_set1_ps(9.9f);
alignas(16) float dest[4];
// Write to dest without polluting the cache
_mm_stream_ps(dest, vec);
```