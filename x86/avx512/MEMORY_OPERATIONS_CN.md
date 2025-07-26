# AVX-512 内存操作

AVX-512 是对 x86 SIMD 能力的一次重大扩展，它引入了 512 位的 ZMM 寄存器，以及更重要的、围绕掩码寄存器（`k0-k7`）构建的全新编程范式。本文档重点介绍其新增及增强的内存操作。

---

## 核心概念

- **512 位向量**: ZMM 寄存器可以容纳 16 个浮点数或 8 个双精度数。
- **掩码寄存器（`k` 寄存器）**: AVX-512 用专用的 8/16/32/64 位掩码寄存器取代了旧的、基于向量的掩码方式。一个用于浮点向量的 `__mmask16` 拥有 16 个比特位，每个比特位直接对应一个向量元素，这使得掩码操作更直观、更强大。
- **Scatter、Compress、Expand**: 用于处理非连续和稀疏数据的强大新指令。

---

## 1. 全新的掩码范式

掩码操作不再通过全 1 或全 0 的向量来完成。取而代之的是使用一个比特掩码（`__mmask8`, `__mmask16` 等），其中每个比特位都对应一个向量通道。

- **`_maskz_...` 内在函数 (置零)**: 如果掩码位为 `0`，目标向量中对应的元素将被置零。
- **`_mask_...` 内在函数 (合并)**: 如果掩码位为 `0`，目标向量中对应的元素将从一个源（`src`）向量中保持不变（透传）。

**示例:**
```cpp
__m512 vec = ...;
__m512 src = _mm512_set1_ps(-1.0f);
alignas(64) float dest[16];

// 一个用于 16 元素浮点向量的 16 位掩码
__mmask16 k = 0b10101010;

// 置零: 位于 0,2,4,6 的元素将被加载，其他元素为 0
__m512 z_vec = _mm512_maskz_load_ps(k, &data);

// 合并: 位于 1,3,5,7 的元素将从 vec 中存储，
// dest 中的其他元素将保持不变
_mm512_mask_store_ps(dest, k, vec);
```

## 2. Scatter 和 Gather

AVX2 引入了 Gather，而 AVX-512 则增加了它的逆向操作：**Scatter**。Scatter 将一个向量中的元素写入到由一个索引向量指定的、非连续的内存地址中。

- **Gather (收集)**: `_mm512_i32gather_ps` - 从 `base[indices[i]]` 加载数据到向量中。
- **Scatter (发散)**: `_mm512_i32scatter_ps` - 将向量中的数据存储到 `base[indices[i]]`。

**示例:**
```cpp
__m512 data_vec = ...; // 包含待写入数据的向量
__m512i indices = ...; // 索引向量
alignas(64) float destination_array[16];

// 将 data_vec[i] 写入到 destination_array[indices[i]]
_mm512_i32scatter_ps(destination_array, indices, data_vec, 4);
```

## 3. Compress 和 Expand

这些指令专为处理稀疏数据而设计，它们可以根据掩码来过滤和打包/解包元素。

- **Compress (压缩)**: 从源向量中提取由掩码指定的活动元素，并将它们**连续地**存储到一个内存地址。这对于过滤向量非常有用。
- **Expand (扩展)**: 执行相反的操作。它从内存中加载一个连续的数据块，并将其**扩展**到一个向量中，只放置在由掩码指定的活动通道里。非活动通道的元素保持不变。

**示例:**
```cpp
__m512 sparse_vec = _mm512_setr_ps(1, -1, 2, -1, 3, -1, 4, -1, ...);
__mmask16 k = 0b1010101010101010; // 用于选择有效数字的掩码
alignas(64) float dense_array[8];

// 压缩: 过滤掉值为 -1 的元素
// dense_array 将变为 [1, 2, 3, 4, 5, 6, 7, 8]
_mm512_mask_compressstoreu_ps(dense_array, k, sparse_vec);

__m512 new_vec = _mm512_setzero_ps();
// 扩展: 将紧凑的数据加载回新向量的正确通道中
// new_vec 将变为 [1, 0, 2, 0, 3, 0, 4, 0, ...]
new_vec = _mm512_mask_expandloadu_ps(new_vec, k, dense_array);
```
