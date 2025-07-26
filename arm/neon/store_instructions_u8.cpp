
#include <iostream>
#include <vector>
#include <arm_neon.h>
#include <cstdint>
#include <string>

// Helper function to print an array of uint8_t
void print_u8_array(const uint8_t* data, size_t size, const std::string& name) {
    std::cout << name << ": [ ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << (int)data[i] << (i == size - 1 ? " " : ", ");
    }
    std::cout << "]" << std::endl;
}

// Helper function to print a 128-bit vector of uint8_t
void print_u8_vector(uint8x16_t vec, const std::string& name) {
    alignas(16) uint8_t buffer[16];
    vst1q_u8(buffer, vec);
    print_u8_array(buffer, 16, name);
}

// Forward declare the generic template
template<typename T, int N>
struct v_type;

// Specializations for NEON's multi-vector types
template<> struct v_type<uint8_t, 2> { using type = uint8x16x2_t; };
template<> struct v_type<uint8_t, 3> { using type = uint8x16x3_t; };
template<> struct v_type<uint8_t, 4> { using type = uint8x16x4_t; };


int main() {
    std::cout << "--- NEON uint8_t Store Instruction Examples ---" << std::endl;

    // =================================================================
    // 1. Basic Store (vst1q_u8)
    // =================================================================
    std::cout << "\n[1. Basic Store (vst1q_u8)]" << std::endl;
    uint8x16_t vec1 = vmovq_n_u8(42); // Vector with all elements set to 42
    alignas(16) uint8_t dest_array[16];
    print_u8_vector(vec1, "Vector to store");
    vst1q_u8(dest_array, vec1);
    print_u8_array(dest_array, 16, "Resulting array");
    std::cout << "// vst1_u8 would store only the 8 elements from a 64-bit vector." << std::endl;

    // =================================================================
    // 2. Store a Single Lane (vst1q_lane_u8)
    // =================================================================
    std::cout << "\n[2. Store a Single Lane (vst1q_lane_u8)]" << std::endl;
    uint8x16_t vec_lane_src = vcombine_u8(vcreate_u8(0x0102030405060708), vcreate_u8(0x090A0B0C0D0E0F10));
    uint8_t dest_lane;
    print_u8_vector(vec_lane_src, "Source vector");
    std::cout << "Storing lane 7 (value 8) into a variable." << std::endl;
    vst1q_lane_u8(&dest_lane, vec_lane_src, 7);
    std::cout << "Resulting value: " << (int)dest_lane << std::endl;

    // =================================================================
    // 3. Interleaving Store (vst3q_u8)
    // =================================================================
    std::cout << "\n[3. Interleaving Store (vst3q_u8)]" << std::endl;
    std::cout << "Simulating storing separate R, G, B vectors into a single interleaved array." << std::endl;
    typename v_type<uint8_t, 3>::type rgb_vectors;
    rgb_vectors.val[0] = vmovq_n_u8(255); // R vector (all 255)
    rgb_vectors.val[1] = vmovq_n_u8(128); // G vector (all 128)
    rgb_vectors.val[2] = vmovq_n_u8(0);   // B vector (all 0)
    
    alignas(16) uint8_t interleaved_dest[16 * 3];
    
    std::cout << "Storing 3 separate vectors (R, G, B)..." << std::endl;
    vst3q_u8(interleaved_dest, rgb_vectors);
    print_u8_array(interleaved_dest, 16 * 3, "Interleaved result (R1,G1,B1, R2,G2,B2, ...)");
    std::cout << "// vst2q_u8 and vst4q_u8 work similarly for 2 and 4 vectors." << std::endl;

    // =================================================================
    // 4. Multi-Vector Store (vst1q_u8_x2)
    // =================================================================
    std::cout << "\n[4. Multi-Vector Store (vst1q_u8_x2)]" << std::endl;
    std::cout << "Storing two separate vectors into one contiguous array." << std::endl;
    typename v_type<uint8_t, 2>::type two_vectors;
    two_vectors.val[0] = vmovq_n_u8(1);
    two_vectors.val[1] = vmovq_n_u8(2);

    alignas(16) uint8_t multi_dest[32];
    vst1q_u8_x2(multi_dest, two_vectors);
    print_u8_array(multi_dest, 32, "Multi-vector store result");
    std::cout << "// This is different from vst2q_u8, which interleaves data." << std::endl;
    std::cout << "// vst1q_u8_x3 and vst1q_u8_x4 work similarly." << std::endl;

    return 0;
}
