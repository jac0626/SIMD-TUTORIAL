#include <iostream>
#include <xmmintrin.h>
#include "simd_utils.h"

int main() {
    // Aligned memory allocation for 16 bytes
    float* data = (float*)_mm_malloc(4 * sizeof(float), 16);
    data[0] = 1.0;
    data[1] = 2.0;
    data[2] = 3.0;
    data[3] = 4.0;

    // Load data from memory
    __m128 reg = _mm_load_ps(data);

    std::cout << "Loaded data:" << std::endl;
    print_m128(reg);

    // Store data back to memory
    float* result = (float*)_mm_malloc(4 * sizeof(float), 16);
    _mm_store_ps(result, reg);

    std::cout << "Stored data:" << std::endl;
    __m128 loaded_result = _mm_load_ps(result);
    print_m128(loaded_result);

    _mm_free(data);
    _mm_free(result);

    return 0;
}
