#include <iostream>
#include <vector>
#include <immintrin.h> // AVX, AVX2
#include <cstdint>
#include "simd_utils.h"

// Helper to print a C-style array
template<typename T>
void print_array(const char* title, const T* data, int size) {
    std::cout << title;
    for (int i = 0; i < size; ++i) {
        std::cout << data[i] << (i == size - 1 ? "" : ", ");
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "--- AVX2 Memory Operations Tutorial ---" << std::endl;

    // =================================================================
    // 1. Gather Instructions
    // =================================================================
    std::cout << std::endl << "[1. Gather Instructions]" << std::endl;
    // Gather is the flagship feature of AVX2. It loads data from non-contiguous
    // memory locations. The final address for each element is calculated as:
    // address = base_addr + (vindex * scale)

    // Source data array
    float source_data[16] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.1, 12.1, 13.1, 14.1, 15.1, 16.1};
    print_array("Source Data: ", source_data, 16);

    // Vector of indices to gather from the source data
    __m256i vindex = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    std::cout << "Indices to gather: "; print_m256i(vindex);

    // Gather float values using 32-bit indices. Scale is sizeof(float).
    __m256 gathered_ps = _mm256_i32gather_ps(source_data, vindex, sizeof(float));
    std::cout << "Gathered floats (_mm256_i32gather_ps): ";
    print_m256(gathered_ps);

    // --- Masked Gather ---
    // The mask determines which elements to gather. Inactive elements can be
    // passed through from a source vector (src) or be zeroed.
    __m256i mask = _mm256_setr_epi32(0xFFFFFFFF, 0, 0xFFFFFFFF, 0, 0xFFFFFFFF, 0, 0, 0);
    __m256 src_passthru = _mm256_set1_ps(-1.0f); // Values for masked-off elements
    
    std::cout << "Mask for gather: "; print_m256i(mask);

    __m256 masked_gathered_ps = _mm256_mask_i32gather_ps(src_passthru, source_data, vindex, mask, sizeof(float));
    std::cout << "Masked gathered floats: ";
    print_m256(masked_gathered_ps);


    // =================================================================
    // 2. Masked Load/Store for Integers
    // =================================================================
    std::cout << std::endl << "[2. Masked Load/Store for Integers]" << std::endl;
    // AVX2 extends masked load/store to integer types.
    alignas(32) int32_t int_data[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    alignas(32) int32_t int_dest[8] = {0};
    __m256i int_mask = _mm256_setr_epi32(0, 0xFFFFFFFF, 0, 0xFFFFFFFF, 0, 0xFFFFFFFF, 0, 0xFFFFFFFF);
    
    std::cout << "Integer mask: "; print_m256i(int_mask);

    // Load only the elements where the mask's high bit is 1.
    __m256i masked_loaded_ints = _mm256_maskload_epi32(int_data, int_mask);
    std::cout << "Masked loaded integers: ";
    print_m256i(masked_loaded_ints);

    // Store only the elements where the mask's high bit is 1.
    _mm256_maskstore_epi32(int_dest, int_mask, masked_loaded_ints);
    print_array("Masked stored integers: ", int_dest, 8);

    return 0;
}
