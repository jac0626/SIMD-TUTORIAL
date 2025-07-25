#include <iostream>
#include <vector>
#include <arm_neon.h>

void vector_add_neon(float* a, float* b, float* result, int n) {
    for (int i = 0; i < n; i += 4) {
        // Load 4 floats from memory into NEON registers
        float32x4_t vec_a = vld1q_f32(a + i);
        float32x4_t vec_b = vld1q_f32(b + i);

        // Perform vector addition
        float32x4_t vec_result = vaddq_f32(vec_a, vec_b);

        // Store the result from the NEON register back to memory
        vst1q_f32(result + i, vec_result);
    }
}

int main() {
    std::vector<float> a = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> b = {5.0, 6.0, 7.0, 8.0};
    std::vector<float> result(4);

    vector_add_neon(a.data(), b.data(), result.data(), 4);

    std::cout << "NEON Result: ";
    for (int i = 0; i < 4; ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
