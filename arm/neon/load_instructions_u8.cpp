
#include <iostream>
#include <vector>
#include <arm_neon.h>
#include <cstdint>
#include <string>

// Helper function to print a 128-bit vector of uint8_t
void print_u8_vector(uint8x16_t vec, const std::string& name) {
    // Create a temporary array to store the vector's data
    alignas(16) uint8_t buffer[16];
    vst1q_u8(buffer, vec);

    std::cout << name << ": [ ";
    for (int i = 0; i < 16; ++i) {
        std::cout << (int)buffer[i] << (i == 15 ? " " : ", ");
    }
    std::cout << "]" << std::endl;
}

// Forward declare the generic template
template<typename T, int N>
struct v_type;

// Specializations for NEON's multi-vector types
template<> struct v_type<uint8_t, 2> { using type = uint8x16x2_t; };
template<> struct v_type<uint8_t, 3> { using type = uint8x16x3_t; };
template<> struct v_type<uint8_t, 4> { using type = uint8x16x4_t; };


// Helper function to print a multi-vector structure (e.g., uint8x16x3_t)
template<int N>
void print_multi_vector(const typename v_type<uint8_t, N>::type& multi_vec, const std::string& base_name) {
    for (int i = 0; i < N; ++i) {
        print_u8_vector(multi_vec.val[i], base_name + "[" + std::to_string(i) + "]");
    }
}


int main() {
    std::cout << "--- NEON uint8_t Load Instruction Examples ---" << std::endl;

    // =================================================================
    // 1. Basic Load (vld1q_u8)
    // =================================================================
    std::cout << "\n[1. Basic Load (vld1q_u8)]" << std::endl;
    alignas(16) uint8_t data_array[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    std::cout << "Original Data: An array from 0 to 15" << std::endl;
    uint8x16_t vec1 = vld1q_u8(data_array);
    print_u8_vector(vec1, "vld1q_u8 result");
    std::cout << "// vld1_u8 would load only the first 8 elements into a 64-bit vector." << std::endl;


    // =================================================================
    // 2. Load and Duplicate (vld1q_dup_u8)
    // =================================================================
    std::cout << "\n[2. Load and Duplicate (vld1q_dup_u8)]" << std::endl;
    uint8_t single_value = 42;
    std::cout << "Original Value: " << (int)single_value << std::endl;
    uint8x16_t vec_dup = vld1q_dup_u8(&single_value);
    print_u8_vector(vec_dup, "vld1q_dup_u8 result");


    // =================================================================
    // 3. Load into a Single Lane (vld1q_lane_u8)
    // =================================================================
    std::cout << "\n[3. Load into a Single Lane (vld1q_lane_u8)]" << std::endl;
    uint8_t value_to_insert = 99;
    uint8x16_t vec_lane_target = vdupq_n_u8(0); // A vector of all zeros
    print_u8_vector(vec_lane_target, "Target vector (before)");
    std::cout << "Value to insert: " << (int)value_to_insert << " into lane 5" << std::endl;
    // Load `value_to_insert` into the 5th lane of `vec_lane_target`
    vec_lane_target = vld1q_lane_u8(&value_to_insert, vec_lane_target, 5);
    print_u8_vector(vec_lane_target, "Target vector (after)");


    // =================================================================
    // 4. De-interleaving Load (vld3q_u8)
    // =================================================================
    std::cout << "\n[4. De-interleaving Load (vld3q_u8)]" << std::endl;
    std::cout << "Simulating loading 16 RGB pixels (R1,G1,B1, R2,G2,B2, ...)" << std::endl;
    alignas(16) uint8_t rgb_data[16 * 3];
    for(int i = 0; i < 16; ++i) {
        rgb_data[i*3 + 0] = i * 10;      // R
        rgb_data[i*3 + 1] = i * 10 + 1;  // G
        rgb_data[i*3 + 2] = i * 10 + 2;  // B
    }
    // vld3q_u8 loads interleaved data and separates it into 3 vectors.
    uint8x16x3_t rgb_vectors = vld3q_u8(rgb_data);
    std::cout << "// vld3q_u8 separates the data into R, G, and B vectors:" << std::endl;
    print_multi_vector<3>(rgb_vectors, "RGB vectors");
    std::cout << "// vld2q_u8 and vld4q_u8 work similarly for 2 and 4 channels." << std::endl;


    // =================================================================
    // 5. Multi-Vector Load (vld1q_u8_x2)
    // =================================================================
    std::cout << "\n[5. Multi-Vector Load (vld1q_u8_x2)]" << std::endl;
    std::cout << "Loading two separate, contiguous vectors from one array." << std::endl;
    alignas(16) uint8_t two_vectors_data[32];
    for(int i=0; i<32; ++i) two_vectors_data[i] = i;
    // vld1q_u8_x2 loads two full vectors from a contiguous block of memory.
    uint8x16x2_t two_vectors = vld1q_u8_x2(two_vectors_data);
    print_multi_vector<2>(two_vectors, "Two vectors");
    std::cout << "// This is different from vld2q_u8, which de-interleaves data." << std::endl;
    std::cout << "// vld1q_u8_x3 and vld1q_u8_x4 work similarly." << std::endl;


    // =================================================================
    // 6. Note on Combined Concepts
    // =================================================================
    std::cout << "\n[6. Note on Combined Concepts]" << std::endl;
    std::cout << "The patterns can be combined. For example:" << std::endl;
    std::cout << " - vld3_dup_u8: Loads 3 values and creates 3 vectors, each duplicating one value." << std::endl;
    std::cout << " - vld4_lane_u8: Loads 4 values and inserts them into a specific lane of 4 existing vectors." << std::endl;
    std::cout << "Understanding the basic patterns allows you to infer how these combined instructions work." << std::endl;

    return 0;
}
