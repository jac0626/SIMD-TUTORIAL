

#include <iostream>
#include <vector>
#include <numeric>
#include <arm_sve.h>

// Helper function to print a vector of int32_t
void print_s32(const std::vector<int32_t>& vec, const std::string& label) {
    std::cout << label << ": ";
    for (int32_t val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

// Helper function to print a vector of uint32_t
void print_u32(const std::vector<uint32_t>& vec, const std::string& label) {
    std::cout << label << ": ";
    for (uint32_t val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}


void nt_gather_load_offset_s32() {
    std::cout << "\n--- Non-Temporal Gather Load with Offsets (s32) ---" << std::endl;
    uint64_t count = svcntw();
    std::vector<int32_t> data_source(count * 2);
    std::iota(data_source.begin(), data_source.end(), 100); // 100, 101, ...

    std::vector<uint32_t> offset_vals(count);
    for(uint64_t i = 0; i < count; ++i) {
        offset_vals[i] = i * 2; // Offsets: 0, 2, 4, ...
    }

    print_s32(data_source, "Data Source  ");
    print_u32(offset_vals, "Offsets      ");

    svbool_t pg = svptrue_b32();
    svuint32_t offsets_vec = svld1_u32(pg, offset_vals.data());

    // Non-temporal gather load using base + vector of offsets
    svint32_t gathered_vec = svldnt1_gather_u32offset_s32(pg, data_source.data(), offsets_vec);

    std::vector<int32_t> result(count);
    svst1_s32(pg, result.data(), gathered_vec);
    print_s32(result, "Gathered Data");
}

void nt_gather_load_offset_u32() {
    std::cout << "\n--- Non-Temporal Gather Load with Offsets (u32) ---" << std::endl;
    uint64_t count = svcntw();
    std::vector<uint32_t> data_source(count * 2);
    std::iota(data_source.begin(), data_source.end(), 200); // 200, 201, ...

    std::vector<uint32_t> offset_vals(count);
    for(uint64_t i = 0; i < count; ++i) {
        offset_vals[i] = i * 2 + 1; // Offsets: 1, 3, 5, ...
    }

    print_u32(data_source, "Data Source  ");
    print_u32(offset_vals, "Offsets      ");

    svbool_t pg = svptrue_b32();
    svuint32_t offsets_vec = svld1_u32(pg, offset_vals.data());

    // Non-temporal gather load using base + vector of offsets
    svuint32_t gathered_vec = svldnt1_gather_u32offset_u32(pg, data_source.data(), offsets_vec);

    std::vector<uint32_t> result(count);
    svst1_u32(pg, result.data(), gathered_vec);
    print_u32(result, "Gathered Data");
}

void nt_gather_load_scaling_s16() {
    std::cout << "\n--- Non-Temporal Gather Load with Sign-Extension (s16) ---" << std::endl;
    uint64_t count = svcntw(); // Number of 32-bit lanes
    std::vector<int16_t> data_source(count * 2);
    std::iota(data_source.begin(), data_source.end(), -50); // -50, -49, ...

    std::vector<uint32_t> offset_vals(count);
    for(uint64_t i = 0; i < count; ++i) {
        offset_vals[i] = i; // Offsets: 0, 1, 2, ...
    }
    
    // Convert to a printable vector<int32_t>
    std::vector<int32_t> printable_source(data_source.begin(), data_source.end());
    print_s32(printable_source, "Data Source (s16)");
    print_u32(offset_vals, "Offsets (u32)    ");

    svbool_t pg = svptrue_b32();
    svuint32_t offsets_vec = svld1_u32(pg, offset_vals.data());

    // Gather load 16-bit signed values and sign-extend to 32-bit
    svint32_t gathered_vec = svldnt1sh_gather_u32offset_s32(pg, data_source.data(), offsets_vec);

    std::vector<int32_t> result(count);
    svst1_s32(pg, result.data(), gathered_vec);
    print_s32(result, "Gathered Data (s32)");
}


int main() {
    std::cout << "SVE2 vector width for int32_t is " << svcntw() << " elements." << std::endl;

    nt_gather_load_offset_s32();
    nt_gather_load_offset_u32();
    nt_gather_load_scaling_s16();

    return 0;
}
