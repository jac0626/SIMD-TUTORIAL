# SVE Memory Operations: A Foundational Guide

The Arm Scalable Vector Extension (SVE) provides powerful instructions for moving data between memory and the scalable vector registers. Understanding these operations is key to writing efficient SIMD code. This guide explains the fundamental memory access patterns available in SVE.

All SVE instructions are governed by a predicate (`svbool_t`). Only the vector lanes corresponding to `true` in the predicate will participate in the memory access.

## 1. Core Access Patterns

There are two primary ways to access memory in SIMD:

- **Consecutive (or Contiguous) Access:** This is the simplest and most common pattern. It involves reading or writing a solid, unbroken block of data from or to memory. Think of it as reading an array sequentially. This is generally the most performant access method.

- **Non-Consecutive Access:** This pattern involves reading or writing data elements that are scattered at different locations in memory. This is essential for more complex algorithms, such as look-up tables, processing sparse data, or working with certain data structures.

## 2. SVE Load & Store Instructions

The examples in `load_instructions_s32.cpp` and `store_instructions_u32.cpp` demonstrate these patterns.

### 2.1. Consecutive Access: `svld1` & `svst1`

- **Concept:** These are the workhorse instructions for simple memory access. `svld1` loads a contiguous block of elements into a vector, and `svst1` stores a vector's contents to a contiguous block.
- **Use Case:** Ideal for processing simple arrays, buffers, and any data that is laid out sequentially in memory.

### 2.2. Non-Consecutive Access (Gather/Scatter): `svld1_gather` & `svst1_scatter`

- **Concept:** SVE provides "gather" loads to read from scattered locations and "scatter" stores to write to them. This is achieved by providing a vector of addresses or indices.
- **Variants:**
    - **Offset-based:** (`_offset_`): You provide a base address and a vector of *byte offsets*. The final address for each lane is `base + offset_vector[lane]`.
    - **Index-based:** (`_index_`): You provide a base address and a vector of *element indices*. The hardware automatically scales the index by the element size (e.g., `sizeof(int32_t)`). The final address is `base + index_vector[lane] * element_size`. This is often more convenient than calculating byte offsets manually.
- **Use Case:** Accessing elements of a structure, implementing look-up tables, or any algorithm where data elements are not adjacent in memory.

### 2.3. Interleaved Access: `svld2`/`svst2`

- **Concept:** These instructions are a specialized form of consecutive access designed for *interleaved* data. `svld2` reads from an interleaved stream (e.g., `R1, G1, R2, G2, ...`) and de-interleaves it into two separate vectors (one for `R`, one for `G`). `svst2` does the reverse. SVE also provides `svld3` and `svld4` for 3- and 4-channel data.
- **Use Case:** Perfect for processing multi-channel data like RGB/RGBA images, stereo audio, or complex numbers.

### 2.4. Specialized Cache-Control and Fault-Handling Loads

- **Concept:** SVE includes instructions to give the programmer more control over caching and error handling.
- **Non-Temporal (`svldnt1`, `svstnt1`):** This instruction provides a hint to the CPU that the data being loaded or stored is "non-temporal" (i.e., it won't be reused soon). This can prevent the data from displacing more important, frequently-used data in the cache, a phenomenon known as "cache pollution." It's a performance optimization for streaming operations.
- **First-Faulting (`svldff1`):** This load will stop upon encountering the first invalid memory access. The predicate register is updated to show which lanes were successfully loaded before the fault occurred. This is very useful for safely traversing data structures with unknown length, like C-style strings.
- **Non-Faulting (`svldnf1`):** This load will *not* cause a program crash on an invalid memory access. Instead, it zeroes out the corresponding lanes in the destination vector and updates the predicate.

---

**Next Steps:** While SVE provides a powerful and comprehensive set of memory operations, its successor, **SVE2**, builds upon this foundation by adding even more specialized and powerful instructions, particularly for enhancing the capabilities of non-contiguous data access.