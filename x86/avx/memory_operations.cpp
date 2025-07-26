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
    // --- Data Preparation ---
    alignas(32) float fa_data[8] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    alignas(32) double da_data[4] = {10.0, 20.0, 30.0, 40.0};
    alignas(32) int32_t ia_data[8] = {100, 200, 300, 400, 500, 600, 700, 800};

    // Unaligned data pointers
    float* fu_ptr = &fa_data[1];
    double* du_ptr = &da_data[1];
    int32_t* iu_ptr = &ia_data[1];

    // Destination memory
    alignas(32) float fa_dest[8];
    alignas(32) double da_dest[4];
    alignas(32) int32_t ia_dest[8];

    std::cout << "--- AVX Load/Store Instructions Tutorial ---" << std::endl;

    // =================================================================
    // 1. Aligned vs. Unaligned Load/Store
    // =================================================================
    std::cout << std::endl << "[1. Aligned vs. Unaligned]" << std::endl;
    __m256 vec_ps_a = _mm256_load_ps(fa_data);
    _mm256_store_ps(fa_dest, vec_ps_a);
    print_array("Aligned float load/store:   ", fa_dest, 8);

    __m256d vec_pd_u = _mm256_loadu_pd(du_ptr);
    _mm256_storeu_pd(da_dest, vec_pd_u);
    print_array("Unaligned double load/store: ", da_dest, 4);

    __m256i vec_si_a = _mm256_load_si256((__m256i*)ia_data);
    _mm256_store_si256((__m256i*)ia_dest, vec_si_a);
    print_array("Aligned integer load/store: ", ia_dest, 8);

    // =================================================================
    // 2. Broadcast Operations
    // =================================================================
    std::cout << std::endl << "[2. Broadcast]" << std::endl;
    // Broadcast a single float value to all elements of a __m256 vector
    __m256 vec_b_ss = _mm256_broadcast_ss(&fa_data[2]); // broadcast 3.0f
    print_m256(vec_b_ss);

    // Broadcast a single double value to all elements of a __m256d vector
    __m256d vec_b_sd = _mm256_broadcast_sd(&da_data[1]); // broadcast 20.0
    print_m256d(vec_b_sd);

    // Broadcast a 128-bit chunk of data to both lanes of a 256-bit vector
    __m256 vec_b_128 = _mm256_broadcast_ps((__m128*)fa_data); // broadcast {1,2,3,4}
    print_m256(vec_b_128);

    // =================================================================
    // 3. Masked Load/Store
    // =================================================================
    std::cout << std::endl << "[3. Masked Load/Store]" << std::endl;
    // Mask: high bit of each element determines if the action occurs.
    // 0xFF... -> true, 0x00... -> false
    __m256i mask_pd = _mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF, 0, 0xFFFFFFFFFFFFFFFF, 0);
    std::cout << "Masked load (doubles):" << std::endl;
    __m256d vec_mask_load = _mm256_maskload_pd(da_data, mask_pd);
    print_m256d(vec_mask_load);

    std::cout << "Masked store (doubles):" << std::endl;
    _mm256_maskstore_pd(da_dest, mask_pd, vec_pd_u);
    print_array("Result: ", da_dest, 4);

    // =================================================================
    // 4. Streaming (Non-Temporal) Store
    // =================================================================
    std::cout << std::endl << "[4. Streaming Store]" << std::endl;
    // Hint to CPU to bypass cache, useful for large data writes.
    _mm256_stream_ps(fa_dest, vec_ps_a);
    print_array("Streaming float store:  ", fa_dest, 8);

    _mm256_stream_pd(da_dest, vec_pd_u);
    print_array("Streaming double store: ", da_dest, 4);

    _mm256_stream_si256((__m256i*)ia_dest, vec_si_a);
    print_array("Streaming integer store:", ia_dest, 8);

    return 0;
}
