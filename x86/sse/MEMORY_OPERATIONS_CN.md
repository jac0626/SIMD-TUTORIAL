# SSE 内存操作

本文档概述了 SSE/SSE2 中可用的基本内存加载和存储操作。理解这些操作对于在内存和 XMM 寄存器之间高效地移动数据至关重要。

---

## 加载指令

加载指令用于将数据从内存移动到 XMM 寄存器中。

### 1. 对齐与非对齐加载

最基本的区别在于对齐和非对齐内存访问。对齐访问速度更快，但要求内存地址是 16 字节的倍数。非对齐访问更灵活，但可能会带来性能损失。

- **对齐加载**: `_mm_load_ps` (movaps), `_mm_load_pd` (movapd), `_mm_load_si128` (movdqa)
- **非对齐加载**: `_mm_loadu_ps` (movups), `_mm_loadu_pd` (movupd), `_mm_loadu_si128` (movdqu)
- **特殊非对齐加载**: `_mm_lddqu_si128` 是一条特殊的整数加载指令，当数据跨越缓存行边界时，它可能比 `movdqu` 更高效。

**示例:**
```cpp
// 对齐: 如果 fa_data 不是 16 字节对齐的，程序会崩溃
alignas(16) float fa_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
__m128 vec_a = _mm_load_ps(fa_data);

// 非对齐: 适用于任何地址
const float* fu_ptr = &some_float_array[1];
__m128 vec_u = _mm_loadu_ps(fu_ptr);
```

### 2. 广播加载

这类指令从内存加载一个单一值，并将其复制到 XMM 寄存器的所有元素中。

- **浮点**: `_mm_load1_ps`
- **双精度**: `_mm_load1_pd` (也作 `_mm_loaddup_pd`)

**示例:**
```cpp
float val = 3.14f;
// vec 将包含 [3.14, 3.14, 3.14, 3.14]
__m128 vec = _mm_load1_ps(&val);
```

### 3. 标量加载

标量加载将一个单一值移动到寄存器的最低位元素，而其他元素保持不变或被清零。

- **加载并清零其他位**: `_mm_load_ss` (浮点), `_mm_load_sd` (双精度)。加载一个值到低位，并将高位清零。

**示例:**
```cpp
float val = 1.23f;
// vec 将包含 [1.23, 0.0, 0.0, 0.0]
__m128 vec = _mm_load_ss(&val);
```

### 4. 部分加载

这类指令将数据加载到 XMM 寄存器的低位或高位部分，而另一半保持不变。

- **加载到低位**: `_mm_loadl_pd` (双精度), `_mm_loadl_epi64` (整数)
- **加载到高位**: `_mm_loadh_pd` (双精度)

**示例:**
```cpp
__m128d vec = _mm_set_pd(99.0, 88.0); // vec 是 [88.0, 99.0]
double val = 10.0;
// vec 变为 [10.0, 99.0]
vec = _mm_loadl_pd(vec, &val);
```

### 5. 流式加载

这是一个给 CPU 的特殊提示，表明加载的数据不会很快被重用。这使得 CPU 可以绕过缓存，从而避免污染更重要的数据。

- **整数**: `_mm_stream_load_si128` (movntdqa)

**示例:**
```cpp
alignas(16) int32_t data[4] = {1, 2, 3, 4};
// 加载数据而不将其放入 CPU 缓存
__m128i vec = _mm_stream_load_si128((__m128i*)data);
```

---

## 存储指令

存储指令用于将数据从 XMM 寄存器移回内存。

### 1. 对齐与非对齐存储

与加载类似，存储也有对齐和非对齐版本。

- **对齐存储**: `_mm_store_ps`, `_mm_store_pd`, `_mm_store_si128`
- **非对齐存储**: `_mm_storeu_ps`, `_mm_storeu_pd`, `_mm_storeu_si128`

**示例:**
```cpp
__m128 vec = _mm_set1_ps(5.0f);
alignas(16) float dest[4];
_mm_store_ps(dest, vec);
```

### 2. 标量与部分存储

这些指令仅存储寄存器的一部分。

- **标量存储**: `_mm_store_ss` (浮点), `_mm_store_sd` (双精度)。仅存储最低位的元素。
- **部分存储**: `_mm_storeh_pd`, `_mm_storel_pd`, `_mm_storel_epi64`。存储寄存器的高位或低位部分。

**示例:**
```cpp
__m128 vec = _mm_setr_ps(1.0f, 2.0f, 3.0f, 4.0f);
float dest = 0.0f;
// dest 变为 1.0f
_mm_store_ss(&dest, vec);
```

### 3. 掩码存储

一个强大的功能，允许在没有分支的情况下进行按元素的条件存储。每个字节的存储都由掩码寄存器中对应字节的最高位来控制。

- **整数**: `_mm_maskmoveu_si128`

**示例:**
```cpp
__m128i data = _mm_setr_epi32(10, 20, 30, 40);
// 用于存储第 1 和第 3 个整数的掩码
__m128i mask = _mm_setr_epi32(0xFFFFFFFF, 0, 0xFFFFFFFF, 0);
alignas(16) int32_t dest[4] = {0};
// dest 变为 [10, 0, 30, 0]
_mm_maskmoveu_si128(data, mask, (char*)dest);
```

### 4. 流式存储

这会指示 CPU 将数据直接写入内存，绕过缓存。这是一种性能优化，适用于不会很快被再次读取的数据，可以防止缓存污染。

- **所有类型**: `_mm_stream_ps`, `_mm_stream_pd`, `_mm_stream_si128`, `_mm_stream_si32`

**示例:**
```cpp
__m128 vec = _mm_set1_ps(9.9f);
alignas(16) float dest[4];
// 写入 dest 而不污染缓存
_mm_stream_ps(dest, vec);
```
