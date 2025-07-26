#include <iostream>
#include <vector>
#include <arm_sve.h>

// Helper function to print a vector of floats
void print_f32_vector(const float* data, size_t size, const std::string& name) {
    std::cout << name << ": ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    // Get the number of 32-bit float elements that fit in a single SVE vector.
    uint64_t num_lanes = svcntw();
    std::cout << "SVE2 vector holds " << num_lanes << " float32 elements." << std::endl << std::endl;

    // --- Basic Load and Store ---
    std::cout << "--- Basic Load and Store ---" << std::endl;

    // Create an array of floats. Its size is determined by the SVE vector width.
    std::vector<float> input_data(num_lanes);
    for(uint64_t i = 0; i < num_lanes; ++i) {
        input_data[i] = 1.1f * (i + 1);
    }
    print_f32_vector(input_data.data(), num_lanes, "Original data ");

    // Create a predicate that enables all 32-bit lanes.
    svbool_t all_lanes = svptrue_b32();

    // Load the data from memory into an SVE register.
    svfloat32_t sve2_vector = svld1_f32(all_lanes, input_data.data());

    // Create an output array.
    std::vector<float> output_data(num_lanes);

    // Store the data from the SVE register back to memory.
    svst1_f32(all_lanes, output_data.data(), sve2_vector);

    print_f32_vector(output_data.data(), num_lanes, "Stored data   ");

    std::cout << "\n--- Modifying data with SIMD ---" << std::endl;

    // Create another array
    std::vector<float> numbers(num_lanes);
    for(uint64_t i = 0; i < num_lanes; ++i) {
        numbers[i] = 10.0f * (i + 1);
    }
    print_f32_vector(numbers.data(), num_lanes, "Initial numbers ");

    // Load the data
    svfloat32_t numbers_vec = svld1_f32(all_lanes, numbers.data());

    // Create a vector of constants to add
    svfloat32_t constants_to_add = svdup_n_f32(5.5f);

    // Perform the addition
    numbers_vec = svadd_f32_z(all_lanes, numbers_vec, constants_to_add);

    // Store the result back
    svst1_f32(all_lanes, numbers.data(), numbers_vec);
    print_f32_vector(numbers.data(), num_lanes, "Modified numbers");

    return 0;
}
