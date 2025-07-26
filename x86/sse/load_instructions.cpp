#include <iostream>
#include <vector>
#include <xmmintrin.h> // SSE
#include <emmintrin.h> // SSE2
#include <smmintrin.h> // SSE4.1 for _mm_stream_load_si128
#include "simd_utils.h"

// Helper to print a simple C-style array
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
    // Aligned data for aligned load instructions
    alignas(16) float fa_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    alignas(16) double da_data[2] = {10.0, 20.0};
    alignas(16) int32_t ia_data[4] = {100, 200, 300, 400};

    // Unaligned data (we will use pointers to create unaligned addresses)
    float fu_data[5] = {0.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    double du_data[3] = {0.0, 50.0, 60.0};
    int32_t iu_data[5] = {0, 500, 600, 700, 800};
    
    // Pointers to unaligned data
    const float* fu_ptr = &fu_data[1];
    const double* du_ptr = &du_data[1];
    const int32_t* iu_ptr = &iu_data[1];

    std::cout << "--- SSE Load Instructions Tutorial ---" << std::endl;

    // =================================================================
    // 1. Aligned vs. Unaligned Loads (movaps, movups, movdqa, movdqu)
    // =================================================================
    std::cout << std::endl << "[1. Aligned vs. Unaligned Loads]" << std::endl;
    
    // _mm_load_ps (movaps): Loads 4 floats from an aligned memory address.
    // Crash if address is not 16-byte aligned.
    std::cout << "Aligned float load (_mm_load_ps):" << std::endl;
    __m128 vec_ps_a = _mm_load_ps(fa_data);
    print_m128(vec_ps_a);

    // _mm_loadu_ps (movups): Loads 4 floats from any memory address.
    std::cout << "Unaligned float load (_mm_loadu_ps):" << std::endl;
    __m128 vec_ps_u = _mm_loadu_ps(fu_ptr);
    print_m128(vec_ps_u);

    // _mm_load_pd (movapd) / _mm_loadu_pd (movupd) for doubles
    std::cout << "Aligned double load (_mm_load_pd):" << std::endl;
    __m128d vec_pd_a = _mm_load_pd(da_data);
    print_m128d(vec_pd_a);

    std::cout << "Unaligned double load (_mm_loadu_pd):" << std::endl;
    __m128d vec_pd_u = _mm_loadu_pd(du_ptr);
    print_m128d(vec_pd_u);

    // _mm_load_si128 (movdqa) / _mm_loadu_si128 (movdqu) for integers
    std::cout << "Aligned integer load (_mm_load_si128):" << std::endl;
    __m128i vec_si_a = _mm_load_si128((__m128i*)ia_data);
    print_m128i(vec_si_a);

    std::cout << "Unaligned integer load (_mm_loadu_si128):" << std::endl;
    __m128i vec_si_u = _mm_loadu_si128((__m128i*)iu_ptr);
    print_m128i(vec_si_u);

    // _mm_lddqu_si128 (lddqu): A special unaligned load for integers that can be
    // faster than movdqu on some architectures, especially when crossing cache line boundaries.
    std::cout << "Special unaligned integer load (_mm_lddqu_si128):" << std::endl;
    __m128i vec_lddqu = _mm_lddqu_si128((__m128i*)iu_ptr);
    print_m128i(vec_lddqu);


    // =================================================================
    // 2. Broadcast / Splat Loads (movss + shufps, movsd + shufpd)
    // =================================================================
    std::cout << std::endl << "[2. Broadcast Loads]" << std::endl;

    // _mm_load1_ps (or _mm_load_ps1): Loads a single float and broadcasts it to all 4 elements.
    std::cout << "Broadcast single float to vector (_mm_load1_ps):" << std::endl;
    print_array("Source float: ", &fa_data[1], 1);
    __m128 vec_ps1 = _mm_load1_ps(&fa_data[1]); // Load 2.0f
    print_m128(vec_ps1);

    // _mm_load1_pd (or _mm_loaddup_pd): Loads a single double and broadcasts it to both elements.
    std::cout << "Broadcast single double to vector (_mm_load1_pd):" << std::endl;
    print_array("Source double: ", &da_data[1], 1);
    __m128d vec_pd1 = _mm_load1_pd(&da_data[1]); // Load 20.0
    print_m128d(vec_pd1);


    // =================================================================
    // 3. Scalar Loads (movss, movsd)
    // =================================================================
    std::cout << std::endl << "[3. Scalar Loads]" << std::endl;

    // _mm_load_ss: Loads a single float into the lowest element, and zeros the upper 3 elements.
    std::cout << "Scalar float load (_mm_load_ss):" << std::endl;
    print_array("Source float: ", &fa_data[0], 1);
    __m128 vec_ss = _mm_load_ss(&fa_data[0]);
    print_m128(vec_ss);

    // _mm_load_sd: Loads a single double into the lowest element, and zeros the upper element.
    std::cout << "Scalar double load (_mm_load_sd):" << std::endl;
    print_array("Source double: ", &da_data[0], 1);
    __m128d vec_sd = _mm_load_sd(&da_data[0]);
    print_m128d(vec_sd);


    // =================================================================
    // 4. Partial Loads (movlpd, movhpd, movlps, movhps)
    // =================================================================
    std::cout << std::endl << "[4. Partial Loads]" << std::endl;
    
    __m128d partial_vec_d = _mm_set_pd(99.0, 88.0);
    std::cout << "Initial double vector: ";
    print_m128d(partial_vec_d);

    // _mm_loadl_pd: Loads one double into the low element of the destination vector, leaving the high element unchanged.
    std::cout << "Load into LOW double (_mm_loadl_pd):" << std::endl;
    print_array("Source double: ", &da_data[0], 1);
    partial_vec_d = _mm_loadl_pd(partial_vec_d, &da_data[0]);
    print_m128d(partial_vec_d);

    // _mm_loadh_pd: Loads one double into the high element of the destination vector, leaving the low element unchanged.
    std::cout << "Load into HIGH double (_mm_loadh_pd):" << std::endl;
    print_array("Source double: ", &da_data[1], 1);
    partial_vec_d = _mm_loadh_pd(partial_vec_d, &da_data[1]);
    print_m128d(partial_vec_d);


    // =================================================================
    // 5. Reversed Loads
    // =================================================================
    std::cout << std::endl << "[5. Reversed Loads]" << std::endl;
    
    // _mm_loadr_ps: Loads 4 floats in reverse order.
    std::cout << "Reversed float load (_mm_loadr_ps):" << std::endl;
    print_array("Source floats: ", fa_data, 4);
    __m128 vec_psr = _mm_loadr_ps(fa_data);
    print_m128(vec_psr);

    // _mm_loadr_pd: Loads 2 doubles in reverse order.
    std::cout << "Reversed double load (_mm_loadr_pd):" << std::endl;
    print_array("Source doubles: ", da_data, 2);
    __m128d vec_pdr = _mm_loadr_pd(da_data);
    print_m128d(vec_pdr);


    // =================================================================
    // 6. Streaming Load (movntdqa)
    // =================================================================
    std::cout << std::endl << "[6. Streaming Load]" << std::endl;
    std::cout << "Streaming load (_mm_stream_load_si128):" << std::endl;
    // This instruction is a hint to the CPU that the data is "non-temporal"
    // and will not be reused soon, so it can bypass the cache hierarchy.
    // It's mainly used for performance tuning in very specific scenarios
    // where you are processing a large amount of data once.
    // NOTE: For this to be effective, the source memory should be of a
    // special type (Write-Combining, WC), which is beyond this basic example.
    // Here, we just demonstrate the syntax on regular memory.
    __m128i stream_vec = _mm_stream_load_si128((__m128i*)ia_data);
    print_m128i(stream_vec);


    return 0;
}
