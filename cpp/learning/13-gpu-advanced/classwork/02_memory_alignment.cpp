/*
 * MODULE 13: GPU Advanced Topics
 * File: 02_memory_alignment.cpp
 *
 * TOPIC: Memory Alignment for GPU Performance
 *
 * COMPILE: g++ -std=c++17 -o memory_alignment 02_memory_alignment.cpp
 */

#include <iostream>
#include <cstddef>
#include <cstdint>

// Unaligned struct (bad for GPU)
struct UnalignedData {
    float x;      // 4 bytes
    char flag;    // 1 byte  -> padding added here
    float y;      // 4 bytes
    short id;     // 2 bytes -> padding added here
};

// Aligned struct (good for GPU)
struct alignas(16) AlignedData {
    float x, y, z, w;  // 16 bytes, naturally aligned
};

// GPU-friendly struct (matches float4 in CUDA)
struct __attribute__((aligned(16))) GPUVector {
    float x, y, z, w;
};

void demonstrateSizes() {
    std::cout << "\n=== Structure Sizes and Alignment ===\n\n";

    std::cout << "UnalignedData:\n";
    std::cout << "  Size: " << sizeof(UnalignedData) << " bytes\n";
    std::cout << "  Alignment: " << alignof(UnalignedData) << " bytes\n";
    std::cout << "  Layout: float(4) + char(1) + PAD(3) + float(4) + short(2) + PAD(2)\n\n";

    std::cout << "AlignedData:\n";
    std::cout << "  Size: " << sizeof(AlignedData) << " bytes\n";
    std::cout << "  Alignment: " << alignof(AlignedData) << " bytes\n";
    std::cout << "  Layout: 4 floats, perfectly aligned to 16 bytes\n\n";

    std::cout << "GPUVector:\n";
    std::cout << "  Size: " << sizeof(GPUVector) << " bytes\n";
    std::cout << "  Alignment: " << alignof(GPUVector) << " bytes\n";
    std::cout << "  Matches CUDA float4 exactly!\n\n";
}

void demonstrateAlignment() {
    std::cout << "=== Memory Addresses and Alignment ===\n\n";

    AlignedData data[4];

    for (int i = 0; i < 4; ++i) {
        uintptr_t addr = reinterpret_cast<uintptr_t>(&data[i]);
        std::cout << "data[" << i << "] address: 0x" << std::hex << addr;
        std::cout << " (aligned: " << std::dec << (addr % 16 == 0 ? "YES" : "NO") << ")\n";
    }

    std::cout << "\nAll addresses are multiples of 16 (aligned)\n";
    std::cout << "GPU can load in single 128-bit transaction!\n\n";
}

void demonstrateGPUConcepts() {
    std::cout << "=== GPU Memory Alignment Requirements ===\n\n";

    std::cout << "CUDA Memory Transaction Sizes:\n";
    std::cout << "  - 32 bytes (L1 cache line segment)\n";
    std::cout << "  - 128 bytes (L2 cache line)\n\n";

    std::cout << "Best Performance:\n";
    std::cout << "  - Data aligned to transaction size\n";
    std::cout << "  - Consecutive threads access consecutive aligned addresses\n";
    std::cout << "  - Example: Thread 0 at addr 0x0, Thread 1 at addr 0x10, etc.\n\n";

    std::cout << "CUDA Types and Alignment:\n";
    std::cout << "  float2:  8 bytes, 8-byte aligned\n";
    std::cout << "  float3: 12 bytes, 4-byte aligned (padded to 16 in arrays!)\n";
    std::cout << "  float4: 16 bytes, 16-byte aligned\n\n";

    std::cout << "Why float4 is common:\n";
    std::cout << "  - Perfect 16-byte alignment\n";
    std::cout << "  - Matches vectorized load/store instructions\n";
    std::cout << "  - Efficient memory transactions\n\n";
}

void demonstratePadding() {
    std::cout << "=== Structure Padding ===\n\n";

    struct BadLayout {
        char a;    // 1 byte
        int b;     // 4 bytes (needs 4-byte alignment)
        char c;    // 1 byte
    };

    struct GoodLayout {
        int b;     // 4 bytes
        char a;    // 1 byte
        char c;    // 1 byte
    };

    std::cout << "BadLayout size: " << sizeof(BadLayout) << " bytes\n";
    std::cout << "  Layout: char(1) + PAD(3) + int(4) + char(1) + PAD(3) = 12 bytes\n\n";

    std::cout << "GoodLayout size: " << sizeof(GoodLayout) << " bytes\n";
    std::cout << "  Layout: int(4) + char(1) + char(1) + PAD(2) = 8 bytes\n\n";

    std::cout << "Lesson: Order struct members largest to smallest!\n\n";
}

int main() {
    std::cout << "=== MODULE 13: Memory Alignment for GPU ===\n";

    demonstrateSizes();
    demonstrateAlignment();
    demonstratePadding();
    demonstrateGPUConcepts();

    std::cout << "=== Key Takeaways ===\n";
    std::cout << "1. Alignment matters GREATLY on GPU\n";
    std::cout << "2. Use alignas() to control alignment\n";
    std::cout << "3. Prefer float4 over float3\n";
    std::cout << "4. Order struct members by size\n";
    std::cout << "5. Check sizeof() and alignof()\n";
    std::cout << "6. Coalesced access requires alignment\n";

    std::cout << "\n=== Program Complete ===\n";
    return 0;
}

/*
 * TRY THIS:
 * 1. Create aligned allocator for std::vector
 * 2. Measure performance of aligned vs unaligned
 * 3. Implement padding calculator
 * 4. Test with different alignment values
 * 5. Use __attribute__((packed)) and measure impact
 */
