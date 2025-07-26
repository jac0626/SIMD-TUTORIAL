#include <iostream>
#include <vector>
#include <xmmintrin.h> // SSE
#include <emmintrin.h> // SSE2
#include <smmintrin.h> // SSE4.1
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
    // Source vectors to be stored
    __m128 vec_ps = _mm_setr_ps(1.1f, 2.2f, 3.3f, 4.4f);
    __m128d vec_pd = _mm_setr_pd(10.1, 20.2);
    __m128i vec_si = _mm_setr_epi32(101, 202, 303, 404);

    // Destination memory, aligned
    alignas(16) float fa_dest[4] = {0};
    alignas(16) double da_dest[2] = {0};
    alignas(16) int32_t ia_dest[4] = {0};

    // Destination memory, unaligned
    std::vector<float> fu_dest_vec(5, 0.0f);
    std::vector<double> du_dest_vec(3, 0.0);
    std::vector<int32_t> iu_dest_vec(5, 0);
    float* fu_dest = &fu_dest_vec[1];
    double* du_dest = &du_dest_vec[1];
    int32_t* iu_dest = &iu_dest_vec[1];

    std::cout << "--- SSE Store Instructions Tutorial ---" << std::endl;

    // =================================================================
    // 1. Aligned vs. Unaligned Stores
    // =================================================================
    std::cout << std::endl << "[1. Aligned vs. Unaligned Stores]" << std::endl;
    _mm_store_ps(fa_dest, vec_ps);
    print_array("Aligned float store (_mm_store_ps):   ", fa_dest, 4);

    _mm_storeu_ps(fu_dest, vec_ps);
    print_array("Unaligned float store (_mm_storeu_ps): ", fu_dest, 4);

    _mm_store_pd(da_dest, vec_pd);
    print_array("Aligned double store (_mm_store_pd):  ", da_dest, 2);

    _mm_storeu_pd(du_dest, vec_pd);
    print_array("Unaligned double store (_mm_storeu_pd):", du_dest, 2);

    _mm_store_si128((__m128i*)ia_dest, vec_si);
    print_array("Aligned integer store (_mm_store_si128): ", ia_dest, 4);

    _mm_storeu_si128((__m128i*)iu_dest, vec_si);
    print_array("Unaligned integer store (_mm_storeu_si128):", iu_dest, 4);

    // =================================================================
    // 2. Scalar Stores
    // =================================================================
    std::cout << std::endl << "[2. Scalar Stores]" << std::endl;
    _mm_store_ss(fa_dest, vec_ps);
    print_array("Scalar float store (_mm_store_ss):    ", fa_dest, 4);

    _mm_store_sd(da_dest, vec_pd);
    print_array("Scalar double store (_mm_store_sd):   ", da_dest, 2);

    // =================================================================
    // 3. Partial Stores
    // =================================================================
    std::cout << std::endl << "[3. Partial Stores]" << std::endl;
    _mm_storeh_pd(da_dest, vec_pd);
    print_array("Store HIGH double (_mm_storeh_pd):    ", da_dest, 1);

    _mm_storel_pd(da_dest, vec_pd);
    print_array("Store LOW double (_mm_storel_pd):     ", da_dest, 1);

    _mm_storel_epi64((__m128i*)ia_dest, vec_si);
    print_array("Store LOW 64-bits (_mm_storel_epi64):", ia_dest, 2);

    // =================================================================
    // 4. Masked (Conditional) Store
    // =================================================================
    std::cout << std::endl << "[4. Masked Store]" << std::endl;
    __m128i mask = _mm_setr_epi32(0xFFFFFFFF, 0, 0xFFFFFFFF, 0); // Mask for 1st and 3rd elements
    std::cout << "Storing with mask: "; print_m128i(mask);
    _mm_maskmoveu_si128(vec_si, mask, (char*)ia_dest);
    print_array("Masked store (_mm_maskmoveu_si128): ", ia_dest, 4);

    // =================================================================
    // 5. Streaming (Non-Temporal) Stores
    // =================================================================
    std::cout << std::endl << "[5. Streaming Stores]" << std::endl;
    // These instructions write directly to memory, bypassing caches.
    // This is a performance hint for data that won't be read again soon.
    _mm_stream_ps(fa_dest, vec_ps);
    print_array("Streaming float store (_mm_stream_ps):", fa_dest, 4);

    _mm_stream_pd(da_dest, vec_pd);
    print_array("Streaming double store (_mm_stream_pd):", da_dest, 2);

    _mm_stream_si128((__m128i*)ia_dest, vec_si);
    print_array("Streaming int store (_mm_stream_si128):", ia_dest, 4);

    _mm_stream_si32(&ia_dest[0], 999);
    print_array("Streaming single int (_mm_stream_si32):", ia_dest, 4);

    return 0;
}
