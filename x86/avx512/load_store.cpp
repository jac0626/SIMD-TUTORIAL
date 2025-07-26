#include <iostream>
#include <immintrin.h>
#include "simd_utils.h"

int main() {
    // Aligned memory allocation for 64 bytes
    float* data = (float*)_mm_malloc(16 * sizeof(float), 64);
    for(int i=0; i<16; ++i) data[i] = i + 1.0;

    // Load data from memory
    __m512 reg = _mm512_load_ps(data);

    std::cout << "Loaded data:" << std::endl;
    print_m512(reg);

    // Store data back to memory
    float* result = (float*)_mm_malloc(16 * sizeof(float), 64);
    _mm512_store_ps(result, reg);

    std::cout << "Stored data:" << std::endl;
    __m512 loaded_result = _mm512_load_ps(result);
    print_m512(loaded_result);

    _mm_free(data);
    _mm_free(result);

    return 0;
}
