# AVX-512 Memory Operations

AVX-512 is a major extension to x86 SIMD capabilities, introducing 512-bit ZMM registers and, more importantly, a new programming paradigm centered around mask registers (`k0-k7`). This document focuses on the new and enhanced memory operations.

---

## Key Concepts

- **512-bit Vectors**: ZMM registers hold 16 floats or 8 doubles.
- **Mask Registers (`k` registers)**: AVX-512 replaces the old vector-based masking with dedicated 8/16/32/64-bit mask registers. A `__mmask16` for a float vector has 16 bits, where each bit directly corresponds to an element, making masking more intuitive and powerful.
- **Scatter, Compress, Expand**: Powerful new instructions for handling non-contiguous and sparse data.

---

## 1. The New Masking Paradigm

Masking is no longer done with SIMD vectors of all-ones or all-zeros. Instead, you use a bitmask (`__mmask8`, `__mmask16`, etc.) where each bit corresponds to a vector lane.

- **`_maskz_...` intrinsics (Zeroing)**: If the mask bit is `0`, the corresponding element in the destination is zeroed.
- **`_mask_...` intrinsics (Merging)**: If the mask bit is `0`, the corresponding element in the destination is preserved from a source (`src`) vector.

**Example:**
```cpp
__m512 vec = ...;
__m512 src = _mm512_set1_ps(-1.0f);
alignas(64) float dest[16];

// A 16-bit mask for a 16-element float vector.
__mmask16 k = 0b10101010;

// Zeroing: Elements at positions 0,2,4,6 will be loaded, others will be 0.
__m512 z_vec = _mm512_maskz_load_ps(k, &data);

// Merging: Elements at positions 1,3,5,7 will be stored from vec,
// others in dest will be untouched.
_mm512_mask_store_ps(dest, k, vec);
```

## 2. Scatter and Gather

While Gather was introduced in AVX2, AVX-512 adds its inverse: **Scatter**. Scatter writes elements from a vector to non-contiguous memory locations specified by an index vector.

- **Gather**: `_mm512_i32gather_ps` - Load from `base[indices[i]]` into a vector.
- **Scatter**: `_mm512_i32scatter_ps` - Store from a vector into `base[indices[i]]`.

**Example:**
```cpp
__m512 data_vec = ...; // Vector with data to write
__m512i indices = ...; // Vector of indices
alignas(64) float destination_array[16];

// Writes data_vec[i] to destination_array[indices[i]]
_mm512_i32scatter_ps(destination_array, indices, data_vec, 4);
```

## 3. Compress and Expand

These instructions are designed to handle sparse data by filtering and packing/unpacking elements based on a mask.

- **Compress**: Takes the active elements from a source vector (selected by a mask) and stores them **contiguously** at a memory location. This is useful for filtering a vector.
- **Expand**: Does the opposite. It takes a contiguous block of data from memory and **expands** it into a vector, placing elements into the lanes selected by a mask. Inactive lanes are left untouched.

**Example:**
```cpp
__m512 sparse_vec = _mm512_setr_ps(1, -1, 2, -1, 3, -1, 4, -1, ...);
__mmask16 k = 0b1010101010101010; // Mask to select the valid numbers
alignas(64) float dense_array[8];

// Compress: Filters out the -1 values.
// dense_array will become [1, 2, 3, 4, 5, 6, 7, 8]
_mm512_mask_compressstoreu_ps(dense_array, k, sparse_vec);

__m512 new_vec = _mm512_setzero_ps();
// Expand: Loads the dense data back into the correct lanes of a new vector.
// new_vec will become [1, 0, 2, 0, 3, 0, 4, 0, ...]
new_vec = _mm512_mask_expandloadu_ps(new_vec, k, dense_array);
```
