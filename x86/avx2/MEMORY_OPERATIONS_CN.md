# AVX2 内存操作

本文档涵盖了由 AVX2 引入的内存操作。虽然 AVX 将 SIMD 扩展到了 256 位，但 AVX2 最重要的贡献是增加了 **Gather**（收集）指令，并扩展了整数操作。

---

## 相对于 AVX 的主要增强

- **Gather 指令**: 这是 AVX2 的旗舰功能。Gather 允许高效地从非连续的内存地址加载数据，这对于涉及间接寻址（例如 `array[indices[i]]`）的应用是一个巨大的性能提升。
- **扩展的整数支持**: AVX2 将大多数 128 位的整数指令扩展到了对应的 256 位版本。
- **新增的广播指令**: 为更多数据类型增加了新的广播指令。

---

## Gather 指令

Gather 是 AVX2 中最重要的内存操作。它通过加载一个索引向量指定的多个内存地址上的元素来构建一个结果向量。

每个元素的地址计算方式如下：
`地址 = 基地址 + (索引 * 缩放因子)`

- `基地址 (base_addr)`: 指向源数据数组起始位置的指针。
- `索引向量 (vindex)`: 一个向量寄存器，其中每个元素都是源数据的一个索引。
- `缩放因子 (scale)`: 一个编译时常量（`1`, `2`, `4`, 或 `8`），用于缩放索引值，通常对应于所加载数据类型的大小。

### 1. 基本 Gather

从由 `vindex` 向量计算出的地址加载数据。

- **浮点**: `_mm256_i32gather_ps` (32位索引), `_mm256_i64gather_ps` (64位索引)
- **双精度**: `_mm256_i32gather_pd`, `_mm256_i64gather_pd`
- **整数**: `_mm256_i32gather_epi32`, `_mm256_i64gather_epi64` 等。

**示例:**
```cpp
float source_data[8] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8};

// 要收集的索引: 0, 2, 4, 6, 1, 3, 5, 7
__m256i vindex = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);

// 从 source_data 中收集元素
// 结果: [1.1, 3.3, 5.5, 7.7, 2.2, 4.4, 6.6, 8.8]
__m256 result = _mm256_i32gather_ps(source_data, vindex, sizeof(float));
```

### 2. 掩码 Gather

掩码 Gather 为操作增加了一个条件。它仅加载掩码寄存器（`mask`）中对应元素的最高有效位为 1 的元素。对于掩码未激活的元素，其值可以从一个源向量（`src`）中透传过来，或者被置零。

- **语法**: `_mm256_mask_i32gather_ps(src, base_addr, vindex, mask, scale)`

**示例:**
```cpp
__m256 src_passthru = _mm256_set1_ps(-1.0f);

// 用于仅收集第 1, 3, 5 个元素的掩码
__m256i mask = _mm256_setr_epi32(0xFFFFFFFF, 0, 0xFFFFFFFF, 0, 0xFFFFFFFF, 0, 0, 0);

// 结果: [1.1, -1.0, 5.5, -1.0, 2.2, -1.0, -1.0, -1.0]
__m256 result = _mm256_mask_i32gather_ps(src_passthru, source_data, vindex, mask, sizeof(float));
```

## 其他内存操作

AVX2 还将掩码加载/存储操作扩展到了整数类型，这在 AVX1 中仅适用于浮点类型。

- **掩码整数加载**: `_mm256_maskload_epi32`, `_mm256_maskload_epi64`
- **掩码整数存储**: `_mm256_maskstore_epi32`, `_mm256_maskstore_epi64`

**示例:**
```cpp
alignas(32) int32_t int_data[8] = {10, 20, 30, 40, 50, 60, 70, 80};
__m256i mask = _mm256_setr_epi32(0, 0xFFFFFFFF, 0, 0, 0, 0xFFFFFFFF, 0, 0);

// 结果: [0, 20, 0, 0, 0, 60, 0, 0]
__m256i loaded_ints = _mm256_maskload_epi32(int_data, mask);
```
