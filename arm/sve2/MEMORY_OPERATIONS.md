# SVE2 Memory Operations: Advanced Features

Arm SVE2 builds upon the foundational memory operations of SVE by introducing more powerful and specialized instructions. SVE2 inherits all of SVE's capabilities, so instructions like `svld1` and `svst1` are still fundamental. The key enhancements in SVE2 for memory access focus on improving performance and flexibility for non-contiguous data patterns.

This guide covers the new features demonstrated in the SVE2 examples.

## 1. The Evolution of Gather/Scatter: Non-Temporal Access

While SVE provides basic gather/scatter operations, SVE2 enhances this capability by combining it with the **non-temporal** hint. This is a critical optimization for streaming and high-performance computing workloads where you are processing large datasets that don't fit in the cache.

### `svldnt1_gather` & `svstnt1_scatter`

- **Concept:** These instructions perform the same gather/scatter function as their SVE counterparts but add a hint to the CPU that the data should not be retained in the cache. This avoids polluting the cache with data that will be used only once, preserving cache space for more frequently accessed data.
- **Use Case:** Ideal for algorithms that process large, non-contiguous datasets, such as sparse matrix operations, particle simulations, or certain types of database lookups, where the data access is scattered and unlikely to be repeated soon.
- **Example:** The `load_nt_gather_instructions.cpp` and `store_nt_scatter_instructions.cpp` files demonstrate these non-temporal gather/scatter operations.

## 2. Enhanced Data Manipulation on the Fly

Another significant SVE2 enhancement is the ability to perform data type conversions and manipulations as part of the memory access itself, saving separate instructions.

### 2.1. Load with Sign/Zero Extension

- **Concept:** SVE2 can load data of a smaller type from memory and automatically *sign-extend* or *zero-extend* it to fit the wider element size of the destination vector. For example, you can load an `int8_t` and have it automatically converted to an `int32_t` in the vector register.
- **Instruction:** `svldnt1sh_gather_...` (Load Signed Halfword) is an example. It gathers 16-bit signed values and sign-extends them to 32-bit integers during the load.
- **Use Case:** This is extremely useful when working with mixed-precision data, where you need to promote a smaller data type to a larger one for calculations.
- **Example:** `nt_gather_load_scaling_s16()` in `load_nt_gather_instructions.cpp`.

### 2.2. Store with Truncation

- **Concept:** SVE2 can also perform the reverse operation: storing data from a vector to a smaller memory type, effectively *truncating* the value.
- **Instruction:** `svstnt1h_scatter_...` (Store Halfword) is an example. It takes 32-bit elements from a vector and writes the lower 16 bits to memory.
- **Use Case:** This is common when you have performed calculations in a higher precision (e.g., 32-bit) and need to store the result back in a lower-precision format (e.g., 16-bit) to save memory or bandwidth.
- **Example:** `nt_scatter_store_truncating_s16()` in `store_nt_scatter_instructions.cpp`.

---

**Summary:** SVE2 does not replace SVE; it extends it. You should always start with the basic SVE instructions (`svld1`, `svst1`) for simple, contiguous data. When dealing with more complex, non-contiguous access patterns, SVE's gather/scatter is your next tool. Finally, for high-performance scenarios involving scattered, non-reusable data, or for on-the-fly data type conversion, SVE2's advanced instructions provide the power and flexibility you need.