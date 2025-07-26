#include <iostream>
#include <vector>
#include <arm_neon.h>

// Helper function to print a vector of floats
void print_f32_vector(const float* data, size_t size, const std::string& name) {
    std::cout << name << ": ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    // --- Basic Load and Store ---
    std::cout << "--- Basic Load and Store ---" << std::endl;

    // Create an array of floats to be loaded into a NEON register.
    // alignas(16) ensures the memory is 16-byte aligned for optimal performance.
    alignas(16) float input_data[4] = {1.1f, 2.2f, 3.3f, 4.4f};
    print_f32_vector(input_data, 4, "Original data ");

    // Load the entire array into a 128-bit NEON register (float32x4_t).
    // vld1q_f32: "Vector Load 1 (quad-word) of float32"
    float32x4_t neon_vector = vld1q_f32(input_data);

    // Create an output array to store the data back from the NEON register.
    alignas(16) float output_data[4];

    // Store the contents of the NEON register into the output array.
    // vst1q_f32: "Vector Store 1 (quad-word) of float32"
    vst1q_f32(output_data, neon_vector);

    print_f32_vector(output_data, 4, "Stored data   ");

    std::cout << "\n--- Modifying data with SIMD ---" << std::endl;

    // Create another array
    alignas(16) float numbers[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    print_f32_vector(numbers, 4, "Initial numbers ");

    // Load the data
    float32x4_t numbers_vec = vld1q_f32(numbers);

    // Create a vector of constants to add
    float32x4_t constants_to_add = vdupq_n_f32(5.5f); // {5.5, 5.5, 5.5, 5.5}

    // Perform the addition
    numbers_vec = vaddq_f32(numbers_vec, constants_to_add);

    // Store the result back
    vst1q_f32(numbers, numbers_vec);
    print_f32_vector(numbers, 4, "Modified numbers");


    return 0;
}