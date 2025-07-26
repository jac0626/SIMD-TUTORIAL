# AVX2 Memory Operations

This document covers the memory operations introduced by AVX2. While AVX expanded SIMD to 256 bits, AVX2's most significant contribution is the addition of **gather** instructions and the expansion of integer operations.

---

## Key Enhancements from AVX

- **Gather Instructions**: The flagship feature of AVX2. Gather allows loading data from non-contiguous memory locations efficiently, which is a major performance boost for applications involving indirect addressing (e.g., `array[indices[i]]`).
- **Expanded Integer Support**: AVX2 extends most 128-bit integer instructions to their 256-bit counterparts.
- **Broadcast Additions**: New broadcast instructions were added for more data types.

---

## Gather Instructions

Gather is the most important memory operation introduced in AVX2. It builds a vector by loading elements from memory locations specified by a vector of indices.

The address of each element is calculated as:
`address = base_addr + (vindex * scale)`

- `base_addr`: A pointer to the beginning of the source data array.
- `vindex`: A vector register where each element is an index into the source data.
- `scale`: A compile-time constant (`1`, `2`, `4`, or `8`) that scales the indices, typically corresponding to the size of the data type being loaded.

### 1. Basic Gather

Loads data from addresses computed from the `vindex` vector.

- **Float**: `_mm256_i32gather_ps` (32-bit indices), `_mm256_i64gather_ps` (64-bit indices)
- **Double**: `_mm256_i32gather_pd`, `_mm256_i64gather_pd`
- **Integer**: `_mm256_i32gather_epi32`, `_mm256_i64gather_epi64`, etc.

**Example:**
```cpp
float source_data[8] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8};

// Indices to gather: 0, 2, 4, 6, 1, 3, 5, 7
__m256i vindex = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);

// Gather elements from source_data
// Result: [1.1, 3.3, 5.5, 7.7, 2.2, 4.4, 6.6, 8.8]
__m256 result = _mm256_i32gather_ps(source_data, vindex, sizeof(float));
```

### 2. Masked Gather

Masked gather adds a condition to the operation. It only loads elements where the corresponding element in a `mask` register has its most significant bit set. For elements where the mask is inactive, the value can either be zeroed or passed through from a source (`src`) register.

- **Syntax**: `_mm256_mask_i32gather_ps(src, base_addr, vindex, mask, scale)`

**Example:**
```cpp
__m256 src_passthru = _mm256_set1_ps(-1.0f);

// Mask to gather only the 1st, 3rd, and 5th elements
__m256i mask = _mm256_setr_epi32(0xFFFFFFFF, 0, 0xFFFFFFFF, 0, 0xFFFFFFFF, 0, 0, 0);

// Result: [1.1, -1.0, 5.5, -1.0, 2.2, -1.0, -1.0, -1.0]
__m256 result = _mm256_mask_i32gather_ps(src_passthru, source_data, vindex, mask, sizeof(float));
```

## Other Memory Operations

AVX2 also extends masked load/store operations to integer types, which were previously only available for floating-point types in AVX1.

- **Masked Integer Load**: `_mm256_maskload_epi32`, `_mm256_maskload_epi64`
- **Masked Integer Store**: `_mm256_maskstore_epi32`, `_mm256_maskstore_epi64`

**Example:**
```cpp
alignas(32) int32_t int_data[8] = {10, 20, 30, 40, 50, 60, 70, 80};
__m256i mask = _mm256_setr_epi32(0, 0xFFFFFFFF, 0, 0, 0, 0xFFFFFFFF, 0, 0);

// Result: [0, 20, 0, 0, 0, 60, 0, 0]
__m256i loaded_ints = _mm256_maskload_epi32(int_data, mask);
```
