#include <iostream>
#include <immintrin.h>
#include "simd_utils.h"

int main() {
    // Aligned memory allocation for 32 bytes
    float* data = (float*)_mm_malloc(8 * sizeof(float), 32);
    for(int i=0; i<8; ++i) data[i] = i + 1.0;

    // Load data from memory
    __m256 reg = _mm256_load_ps(data);

    std::cout << "Loaded data:" << std::endl;
    print_m256(reg);

    // Store data back to memory
    float* result = (float*)_mm_malloc(8 * sizeof(float), 32);
    _mm256_store_ps(result, reg);

    std::cout << "Stored data:" << std::endl;
    __m256 loaded_result = _mm256_load_ps(result);
    print_m256(loaded_result);

    _mm_free(data);
    _mm_free(result);

    return 0;
}
