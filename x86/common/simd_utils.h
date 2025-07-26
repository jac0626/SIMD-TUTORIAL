#ifndef SIMD_UTILS_H
#define SIMD_UTILS_H

#include <iostream>
#include <immintrin.h>

// SSE print function
#ifdef __SSE__
void print_m128(__m128 var) {
    alignas(16) float val[4];
    _mm_store_ps(val, var);
    std::cout << "Values: " << val[0] << " " << val[1] << " " << val[2] << " " << val[3] << std::endl;
}
#endif

// AVX/AVX2 print function
#ifdef __AVX__
void print_m256(__m256 var) {
    alignas(32) float val[8];
    _mm256_store_ps(val, var);
    std::cout << "Values: ";
    for(int i=0; i<8; ++i) std::cout << val[i] << " ";
    std::cout << std::endl;
}
#endif

// AVX512 print function
#ifdef __AVX512F__
void print_m512(__m512 var) {
    alignas(64) float val[16];
    _mm512_store_ps(val, var);
    std::cout << "Values: ";
    for(int i=0; i<16; ++i) std::cout << val[i] << " ";
    std::cout << std::endl;
}
#endif

#endif // SIMD_UTILS_H
