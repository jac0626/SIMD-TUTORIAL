
#include <iostream>
#include <vector>
#include <numeric>
#include <arm_sve.h>

// Helper function to print a vector of uint32_t
void print_u32(const std::vector<uint32_t>& vec, const std::string& label) {
    std::cout << label << ": ";
    for (uint32_t val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

// --- Demonstration Functions ---

void consecutive_store_u32() {
    std::cout << "\n--- Consecutive Store (svst1_u32) ---" << std::endl;
    uint64_t count = svcntw();
    std::vector<uint32_t> data_to_store(count);
    std::iota(data_to_store.begin(), data_to_store.end(), 10); // Fill with 10, 11, 12, ...
    print_u32(data_to_store, "Vector to Store");

    svbool_t pg = svptrue_b32();
    svuint32_t vec = svld1_u32(pg, data_to_store.data());

    std::vector<uint32_t> destination(count, 0);
    svst1_u32(pg, destination.data(), vec);
    print_u32(destination, "Stored Data    ");
}

void scatter_store_offset_u32() {
    std::cout << "\n--- Scatter Store with Offsets (svst1_scatter_u32offset_u32) ---" << std::endl;
    uint64_t count = svcntw();
    std::vector<uint32_t> data_to_store(count);
    std::iota(data_to_store.begin(), data_to_store.end(), 100);
    
    // Offsets: 0, 2, 4, 6, ...
    std::vector<uint32_t> offset_vals(count);
    for(uint64_t i = 0; i < count; ++i) {
        offset_vals[i] = i * 2;
    }

    print_u32(data_to_store, "Vector to Store");
    print_u32(offset_vals,   "Offsets        ");

    svbool_t pg = svptrue_b32();
    svuint32_t data_vec = svld1_u32(pg, data_to_store.data());
    svuint32_t offsets_vec = svld1_u32(pg, offset_vals.data());

    std::vector<uint32_t> destination(count * 2, 0); // Make destination large enough
    svst1_scatter_u32offset_u32(pg, destination.data(), offsets_vec, data_vec);
    print_u32(destination, "Scattered Data ");
}

void scatter_store_index_u32() {
    std::cout << "\n--- Scatter Store with Indices (svst1_scatter_u32index_u32) ---" << std::endl;
    uint64_t count = svcntw();
    std::vector<uint32_t> data_to_store(count);
    std::iota(data_to_store.begin(), data_to_store.end(), 200);

    // Indices: 1, 3, 5, 7, ...
    std::vector<uint32_t> index_vals(count);
    for(uint64_t i = 0; i < count; ++i) {
        index_vals[i] = i * 2 + 1;
    }

    print_u32(data_to_store, "Vector to Store");
    print_u32(index_vals,   "Indices        ");

    svbool_t pg = svptrue_b32();
    svuint32_t data_vec = svld1_u32(pg, data_to_store.data());
    svuint32_t indices_vec = svld1_u32(pg, index_vals.data());

    std::vector<uint32_t> destination(count * 2, 0); // Make destination large enough
    svst1_scatter_u32index_u32(pg, destination.data(), indices_vec, data_vec);
    print_u32(destination, "Scattered Data ");
}

void multi_vector_store_u32() {
    std::cout << "\n--- Multi-vector Store for Interleaving (svst2_u32) ---" << std::endl;
    uint64_t count = svcntw();
    std::vector<uint32_t> data0(count), data1(count);
    std::iota(data0.begin(), data0.end(), 300);
    std::iota(data1.begin(), data1.end(), 400);

    print_u32(data0, "Vector 0 to Store");
    print_u32(data1, "Vector 1 to Store");

    svbool_t pg = svptrue_b32();
    svuint32_t vec0 = svld1_u32(pg, data0.data());
    svuint32_t vec1 = svld1_u32(pg, data1.data());
    svuint32x2_t two_vecs = svcreate2_u32(vec0, vec1);

    std::vector<uint32_t> destination(count * 2, 0);
    svst2_u32(pg, destination.data(), two_vecs);
    print_u32(destination, "Interleaved Data ");
}

void non_temporal_store_u32() {
    std::cout << "\n--- Non-Temporal Store (svstnt1_u32) ---" << std::endl;
    uint64_t count = svcntw();
    std::vector<uint32_t> data_to_store(count);
    std::iota(data_to_store.begin(), data_to_store.end(), 500);
    print_u32(data_to_store, "Vector to Store");

    svbool_t pg = svptrue_b32();
    svuint32_t vec = svld1_u32(pg, data_to_store.data());

    std::vector<uint32_t> destination(count, 0);
    // Hints to the CPU that this data won't be reused soon
    svstnt1_u32(pg, destination.data(), vec);
    print_u32(destination, "Stored Data    ");
}

int main() {
    std::cout << "SVE vector width for uint32_t is " << svcntw() << " elements." << std::endl;

    consecutive_store_u32();
    scatter_store_offset_u32();
    scatter_store_index_u32();
    multi_vector_store_u32();
    non_temporal_store_u32();

    return 0;
}
