#include <iostream>
#include <vector>
#include <arm_sve.h>

void vector_add_sve(float* a, float* b, float* result, int n) {
    svbool_t pg = svwhilelt_b32(0, n);
    int i = 0;
    do {
        // Load vectors from memory
        svfloat32_t vec_a = svld1_f32(pg, a + i);
        svfloat32_t vec_b = svld1_f32(pg, b + i);

        // Perform vector addition
        svfloat32_t vec_result = svadd_f32_x(pg, vec_a, vec_b);

        // Store the result back to memory
        svst1_f32(pg, result + i, vec_result);

        i += svcntw();
        pg = svwhilelt_b32(i, n);
    } while (svptest_first(svptrue_b32(), pg));
}

int main() {
    int n = svcntw();
    std::vector<float> a(n);
    std::vector<float> b(n);
    std::vector<float> result(n);

    for(int i=0; i<n; ++i) {
        a[i] = i;
        b[i] = n - i;
    }

    vector_add_sve(a.data(), b.data(), result.data(), n);

    std::cout << "SVE Result: ";
    for (int i = 0; i < n; ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
