#ifndef SIMD_UTILS_H
#define SIMD_UTILS_H

#include <iostream>
#include <immintrin.h>
#include <cstdint>
#include <bitset>

// SSE print function for __m128 (4x float)
#ifdef __SSE__
void print_m128(__m128 var) {
    alignas(16) float val[4];
    _mm_store_ps(val, var);
    std::cout << "Values: " << val[0] << " " << val[1] << " " << val[2] << " " << val[3] << std::endl;
}
#endif

// SSE2 print function for __m128d (2x double)
#ifdef __SSE2__
void print_m128d(__m128d var) {
    alignas(16) double val[2];
    _mm_store_pd(val, var);
    std::cout << "Values: " << val[0] << " " << val[1] << std::endl;
}
#endif

// SSE2 print function for __m128i (e.g., 4x int32_t)
#ifdef __SSE2__
void print_m128i(__m128i var) {
    alignas(16) int32_t val[4];
    _mm_store_si128((__m128i*)val, var);
    std::cout << "Values: " << val[0] << " " << val[1] << " " << val[2] << " " << val[3] << std::endl;
}
#endif

// AVX print function for __m256 (8x float)
#ifdef __AVX__
void print_m256(__m256 var) {
    alignas(32) float val[8];
    _mm256_store_ps(val, var);
    std::cout << "Values: ";
    for(int i=0; i<8; ++i) std::cout << val[i] << " ";
    std::cout << std::endl;
}
#endif

// AVX print function for __m256d (4x double)
#ifdef __AVX__
void print_m256d(__m256d var) {
    alignas(32) double val[4];
    _mm256_store_pd(val, var);
    std::cout << "Values: ";
    for(int i=0; i<4; ++i) std::cout << val[i] << " ";
    std::cout << std::endl;
}
#endif

// AVX print function for __m256i (e.g., 8x int32_t)
#ifdef __AVX__
void print_m256i(__m256i var) {
    alignas(32) int32_t val[8];
    _mm256_store_si256((__m256i*)val, var);
    std::cout << "Values: ";
    for(int i=0; i<8; ++i) std::cout << val[i] << " ";
    std::cout << std::endl;
}
#endif

// AVX512 print function for __m512 (16x float)
#ifdef __AVX512F__
void print_m512(__m512 var) {
    alignas(64) float val[16];
    _mm512_store_ps(val, var);
    std::cout << "Values: ";
    for(int i=0; i<16; ++i) std::cout << val[i] << " ";
    std::cout << std::endl;
}
#endif

// AVX512 print function for __m512d (8x double)
#ifdef __AVX512F__
void print_m512d(__m512d var) {
    alignas(64) double val[8];
    _mm512_store_pd(val, var);
    std::cout << "Values: ";
    for(int i=0; i<8; ++i) std::cout << val[i] << " ";
    std::cout << std::endl;
}
#endif

// AVX512 print function for __m512i (e.g., 16x int32_t)
#ifdef __AVX512F__
void print_m512i(__m512i var) {
    alignas(64) int32_t val[16];
    _mm512_store_si512(val, var);
    std::cout << "Values: ";
    for(int i=0; i<16; ++i) std::cout << val[i] << " ";
    std::cout << std::endl;
}
#endif

// AVX512 mask printers
#ifdef __AVX512F__
void print_mask8(__mmask8 k) { std::cout << "Mask: " << std::bitset<8>(k) << std::endl; }
void print_mask16(__mmask16 k) { std::cout << "Mask: " << std::bitset<16>(k) << std::endl; }
#endif

#endif // SIMD_UTILS_H