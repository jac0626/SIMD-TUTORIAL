

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

// --- Demonstration Functions ---

void consecutive_load_s32() {
    std::cout << "\n--- Consecutive Load (svld1_s32) ---" << std::endl;
    uint64_t count = svcntw();
    std::vector<int32_t> data(count);
    std::iota(data.begin(), data.end(), 10); // Fill with 10, 11, 12, ...
    print_s32(data, "Original Data ");

    svbool_t pg = svptrue_b32();
    svint32_t vec = svld1_s32(pg, data.data());

    std::vector<int32_t> result(count);
    svst1_s32(pg, result.data(), vec);
    print_s32(result, "Loaded Vector ");
}

void gather_load_offset_s32() {
    std::cout << "\n--- Gather Load with Offsets (svld1_gather_s32offset_s32) ---" << std::endl;
    uint64_t count = svcntw();
    // Data source is larger, as offsets can point anywhere
    std::vector<int32_t> data_source(count * 2);
    std::iota(data_source.begin(), data_source.end(), 100); // 100, 101, ...

    // Offsets: 0, 2, 4, 6, ...
    std::vector<int32_t> offset_vals(count);
    for(uint64_t i = 0; i < count; ++i) {
        offset_vals[i] = i * 2;
    }
    
    print_s32(data_source, "Data Source   ");
    print_s32(offset_vals, "Offsets       ");

    svbool_t pg = svptrue_b32();
    svint32_t offsets_vec = svld1_s32(pg, offset_vals.data());
    
    // Gather load using base + vector of offsets
    svint32_t gathered_vec = svld1_gather_s32offset_s32(pg, data_source.data(), offsets_vec);

    std::vector<int32_t> result(count);
    svst1_s32(pg, result.data(), gathered_vec);
    print_s32(result, "Gathered Data ");
}

void gather_load_index_s32() {
    std::cout << "\n--- Gather Load with Indices (svld1_gather_s32index_s32) ---" << std::endl;
    uint64_t count = svcntw();
    std::vector<int32_t> data_source(count * 2);
    std::iota(data_source.begin(), data_source.end(), 200); // 200, 201, ...

    // Indices: 1, 3, 5, 7, ...
    std::vector<int32_t> index_vals(count);
    for(uint64_t i = 0; i < count; ++i) {
        index_vals[i] = i * 2 + 1;
    }

    print_s32(data_source, "Data Source  ");
    print_s32(index_vals, "Indices      ");

    svbool_t pg = svptrue_b32();
    svint32_t indices_vec = svld1_s32(pg, index_vals.data());

    // Gather load using base + vector of indices (indices are scaled by element size)
    svint32_t gathered_vec = svld1_gather_s32index_s32(pg, data_source.data(), indices_vec);

    std::vector<int32_t> result(count);
    svst1_s32(pg, result.data(), gathered_vec);
    print_s32(result, "Gathered Data");
}

void multi_vector_load_s32() {
    std::cout << "\n--- Multi-vector Load for De-interleaving (svld2_s32) ---" << std::endl;
    uint64_t count = svcntw();
    // Interleaved data: {v0_0, v1_0, v0_1, v1_1, ...}
    std::vector<int32_t> interleaved_data(count * 2);
    for(uint64_t i = 0; i < count; ++i) {
        interleaved_data[i * 2] = 300 + i;      // Belongs to vector 0
        interleaved_data[i * 2 + 1] = 400 + i;  // Belongs to vector 1
    }
    print_s32(interleaved_data, "Interleaved Data");

    svbool_t pg = svptrue_b32();
    svint32x2_t two_vecs = svld2_s32(pg, interleaved_data.data());

    svint32_t vec0 = svget2_s32(two_vecs, 0);
    svint32_t vec1 = svget2_s32(two_vecs, 1);

    std::vector<int32_t> result0(count), result1(count);
    svst1_s32(pg, result0.data(), vec0);
    svst1_s32(pg, result1.data(), vec1);

    print_s32(result0, "De-interleaved Vec 0");
    print_s32(result1, "De-interleaved Vec 1");
}



void special_loads_s32() {
    std::cout << "\n--- Specialized Loads (First-Fault, Non-Fault, Non-Temporal) ---" << std::endl;
    uint64_t count = svcntw();
    std::vector<int32_t> data(count);
    std::iota(data.begin(), data.end(), 600);
    print_s32(data, "Original Data");

    svbool_t pg = svptrue_b32();
    std::vector<int32_t> result(count);

    // First-fault load: stops on a memory fault, updating the predicate.
    // We can't trigger a real fault here, but this is the syntax.
    svint32_t ff_vec = svldff1_s32(pg, data.data());
    svst1_s32(pg, result.data(), ff_vec);
    print_s32(result, "First-Fault Load Result");

    // Non-fault load: if a fault occurs, zeroes the result and clears the predicate.
    svint32_t nf_vec = svldnf1_s32(pg, data.data());
    svst1_s32(pg, result.data(), nf_vec);
    print_s32(result, "Non-Fault Load Result  ");

    // Non-temporal load: hints to the CPU that this data won't be reused soon.
    svint32_t nt_vec = svldnt1_s32(pg, data.data());
    svst1_s32(pg, result.data(), nt_vec);
    print_s32(result, "Non-Temporal Load Result");
}


int main() {
    std::cout << "SVE vector width for int32_t is " << svcntw() << " elements." << std::endl;

    consecutive_load_s32();
    gather_load_offset_s32();
    gather_load_index_s32();
    multi_vector_load_s32();
    special_loads_s32();

    return 0;
}

