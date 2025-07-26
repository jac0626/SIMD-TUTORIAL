# AVX 内存操作

本文档涵盖了由 AVX（高级矢量扩展）指令集引入或增强的基本内存操作。AVX 将 SIMD 的能力扩展到 256 位的 YMM 寄存器，并引入了更强大、更灵活的指令。

---

## 相对于 SSE 的主要增强

- **256 位向量**: YMM 寄存器的数据容量是 XMM 寄存器的两倍（例如，可以容纳 8 个浮点数或 4 个双精度数）。
- **三操作数指令**: 大多数 AVX 指令可以是非破坏性的，即可以指定两个源操作数和一个目标操作数（`dest = src1 op src2`），这提高了代码的清晰度和效率。
- **掩码操作**: AVX 引入了一种更强大、更明确的方式，可以使用掩码寄存器来条件性地加载或存储元素。

---

## 加载与存储指令

### 1. 对齐与非对齐访问

与 SSE 类似，AVX 也为对齐和非对齐内存访问提供了相应的指令。对于 AVX，YMM 寄存器的对齐要求是 32 字节。

- **对齐**: `_mm256_load_ps`, `_mm256_store_pd`, `_mm256_load_si256` 等。
- **非对齐**: `_mm256_loadu_ps`, `_mm256_storeu_pd`, `_mm256_loadu_si256` 等。

**示例:**
```cpp
// 对齐访问需要 32 字节边界
alignas(32) float data[8];
__m256 vec = _mm256_load_ps(data);

// 非对齐访问适用于任何地址
_mm256_storeu_ps(&data[1], vec);
```

### 2. 广播操作

AVX 提供了一套丰富的广播指令，可以从内存加载单个数据元素，并将其复制到 YMM 寄存器的所有元素中。

- **广播单个元素**: `_mm256_broadcast_ss` (浮点), `_mm256_broadcast_sd` (双精度)。
- **广播 128 位通道**: `_mm256_broadcast_ps` 或 `_mm256_broadcast_pd` 可以从内存加载一个 128 位的 XMM 向量，并将其复制到 256 位 YMM 向量的低位和高位两个通道中。

**示例:**
```cpp
double val = 42.0;
// vec 变为 [42.0, 42.0, 42.0, 42.0]
__m256d vec = _mm256_broadcast_sd(&val);

alignas(16) float f_vals[4] = {1.f, 2.f, 3.f, 4.f};
// f_vec 变为 [1,2,3,4,1,2,3,4]
__m256 f_vec = _mm256_broadcast_ps((__m128*)f_vals);
```

### 3. 掩码加载与存储

这是 AVX 的一个主要特性。它允许你根据一个掩码来条件性地加载或存储元素。在掩码寄存器中，每个元素的最高有效位充当选择器（1 = 执行操作，0 = 不执行）。

- **掩码加载**: `_mm256_maskload_ps`, `_mm256_maskload_pd`
- **掩码存储**: `_mm256_maskstore_ps`, `_mm256_maskstore_pd`

**示例:**
```cpp
alignas(32) double data[4] = {10, 20, 30, 40};
alignas(32) double dest[4] = {0};

// 创建一个掩码来加载/存储第 1 和第 4 个元素
__m256i mask = _mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF, 0, 0, 0xFFFFFFFFFFFFFFFF);

// 仅加载掩码指定的元素；其他元素被清零。
// loaded_vec 将变为 [10.0, 0.0, 0.0, 40.0]
__m256d loaded_vec = _mm256_maskload_pd(data, mask);

// 仅存储 loaded_vec 中掩码设置了的元素。
// dest 将变为 [10.0, 0.0, 0.0, 40.0]
_mm256_maskstore_pd(dest, mask, loaded_vec);
```

### 4. 流式存储

与 SSE 类似，AVX 提供流式存储指令，将数据直接写入内存，绕过缓存层级。这是一种性能提示，适用于预期不会很快被再次读取的数据。

- **所有类型**: `_mm256_stream_ps`, `_mm256_stream_pd`, `_mm256_stream_si256`

**示例:**
```cpp
__m256d vec = _mm256_set1_pd(99.9);
alignas(32) double dest[4];
// 写入 dest 而不污染缓存
_mm256_stream_pd(dest, vec);
```
