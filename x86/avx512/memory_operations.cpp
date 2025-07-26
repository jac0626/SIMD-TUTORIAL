#include <iostream>
#include <vector>
#include <immintrin.h>
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
    std::cout << "--- AVX-512 Memory Operations Tutorial ---" << std::endl;

    // --- Data Preparation ---
    alignas(64) float data_ps[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    alignas(64) int32_t indices[16] = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    alignas(64) float dest_ps[16] = {0};
    alignas(64) int32_t dest_epi32[16] = {0};

    __m512i vindex = _mm512_load_si512(indices);
    __m512 vec_ps = _mm512_load_ps(data_ps);

    // =================================================================
    // 1. New Masking Paradigm
    // =================================================================
    std::cout << std::endl << "[1. New Masking Paradigm]" << std::endl;
    // AVX-512 uses dedicated mask registers (k0-k7) of type __mmask8, __mmask16, etc.
    // A mask is a bitmask, not a vector of high/low values like in AVX/SSE.
    __mmask16 k = 0b1010101010101010; // 1 = active, 0 = inactive
    print_mask16(k);

    std::cout << "Masked load (zeroing inactive):" << std::endl;
    __m512 masked_load_z = _mm512_maskz_load_ps(k, data_ps);
    print_m512(masked_load_z);

    std::cout << "Masked store:" << std::endl;
    _mm512_mask_store_ps(dest_ps, k, vec_ps);
    print_array("Result of masked store: ", dest_ps, 16);

    // =================================================================
    // 2. Scatter / Gather
    // =================================================================
    std::cout << std::endl << "[2. Scatter / Gather]" << std::endl;
    std::cout << "Gathering with reversed indices:" << std::endl;
    __m512 gathered_vec = _mm512_i32gather_ps(vindex, data_ps, 4);
    print_m512(gathered_vec);

    std::cout << "Scattering the gathered (reversed) vector:" << std::endl;
    // Scatter writes the elements of a vector to different memory locations
    // based on an index vector. It's the inverse of gather.
    _mm512_i32scatter_ps(dest_ps, vindex, gathered_vec, 4);
    print_array("Result of scatter: ", dest_ps, 16);

    // =================================================================
    // 3. Compress / Expand
    // =================================================================
    std::cout << std::endl << "[3. Compress / Expand]" << std::endl;
    // Compress: Takes active elements from a source vector and stores them
    // contiguously in a destination memory location.
    std::cout << "Compressing active elements from a vector:" << std::endl;
    print_m512(vec_ps);
    print_mask16(k);
    _mm512_mask_compressstoreu_ps(dest_ps, k, vec_ps);
    print_array("Result of compress store: ", dest_ps, 8); // Only 8 elements are active

    // Expand: The inverse of compress. It loads contiguous data from memory
    // and places it into the active lanes of a vector, leaving masked-off lanes untouched.
    __m512 expand_dest = _mm512_set1_ps(-1.0f);
    std::cout << "Expanding contiguous data into a vector:" << std::endl;
    expand_dest = _mm512_mask_expandloadu_ps(expand_dest, k, dest_ps);
    print_m512(expand_dest);

    return 0;
}
