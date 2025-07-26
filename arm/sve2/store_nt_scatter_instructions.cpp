
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

// Helper function to print a vector of int16_t
void print_s16(const std::vector<int16_t>& vec, const std::string& label) {
    std::cout << label << ": ";
    for (int16_t val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}


void nt_scatter_store_offset_s32() {
    std::cout << "\n--- Non-Temporal Scatter Store with Offsets (s32) ---" << std::endl;
    uint64_t count = svcntw();
    std::vector<int32_t> data_to_store(count);
    std::iota(data_to_store.begin(), data_to_store.end(), 100); // 100, 101, ...

    std::vector<uint32_t> offset_vals(count);
    for(uint64_t i = 0; i < count; ++i) {
        offset_vals[i] = i * 2; // Offsets: 0, 2, 4, ...
    }

    print_s32(data_to_store, "Data to Store");
    print_u32(offset_vals,   "Offsets      ");

    svbool_t pg = svptrue_b32();
    svint32_t data_vec = svld1_s32(pg, data_to_store.data());
    svuint32_t offsets_vec = svld1_u32(pg, offset_vals.data());

    std::vector<int32_t> destination(count * 2, 0);
    svstnt1_scatter_u32offset_s32(pg, destination.data(), offsets_vec, data_vec);
    print_s32(destination, "Scattered Data");
}

void nt_scatter_store_offset_u32() {
    std::cout << "\n--- Non-Temporal Scatter Store with Offsets (u32) ---" << std::endl;
    uint64_t count = svcntw();
    std::vector<uint32_t> data_to_store(count);
    std::iota(data_to_store.begin(), data_to_store.end(), 200); // 200, 201, ...

    std::vector<uint32_t> offset_vals(count);
    for(uint64_t i = 0; i < count; ++i) {
        offset_vals[i] = i * 2 + 1; // Offsets: 1, 3, 5, ...
    }

    print_u32(data_to_store, "Data to Store");
    print_u32(offset_vals,   "Offsets      ");

    svbool_t pg = svptrue_b32();
    svuint32_t data_vec = svld1_u32(pg, data_to_store.data());
    svuint32_t offsets_vec = svld1_u32(pg, offset_vals.data());

    std::vector<uint32_t> destination(count * 2, 0);
    svstnt1_scatter_u32offset_u32(pg, destination.data(), offsets_vec, data_vec);
    print_u32(destination, "Scattered Data");
}

void nt_scatter_store_truncating_s16() {
    std::cout << "\n--- Non-Temporal Scatter Store with Truncation (s32 -> s16) ---" << std::endl;
    uint64_t count = svcntw(); // Number of 32-bit lanes
    std::vector<int32_t> data_to_store(count);
    std::iota(data_to_store.begin(), data_to_store.end(), 300); // 300, 301, ...

    std::vector<uint32_t> offset_vals(count);
    for(uint64_t i = 0; i < count; ++i) {
        offset_vals[i] = i; // Offsets: 0, 1, 2, ...
    }

    print_s32(data_to_store, "Data to Store (s32)");
    print_u32(offset_vals,   "Offsets (u32)    ");

    svbool_t pg = svptrue_b32();
    svint32_t data_vec = svld1_s32(pg, data_to_store.data());
    svuint32_t offsets_vec = svld1_u32(pg, offset_vals.data());

    std::vector<int16_t> destination(count * 2, 0);
    // Scatter store, truncating 32-bit vector elements to 16-bit memory locations
    svstnt1h_scatter_u32offset_s32(pg, destination.data(), offsets_vec, data_vec);
    print_s16(destination, "Scattered Data (s16)");
}


int main() {
    std::cout << "SVE2 vector width for int32_t is " << svcntw() << " elements." << std::endl;

    nt_scatter_store_offset_s32();
    nt_scatter_store_offset_u32();
    nt_scatter_store_truncating_s16();

    return 0;
}
