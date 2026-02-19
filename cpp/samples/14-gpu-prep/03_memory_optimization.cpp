/*
 * MODULE 14: GPU Preparation
 * File: 03_memory_optimization.cpp
 *
 * TOPIC: GPU Memory Optimization Patterns
 *
 * COMPILE: g++ -std=c++17 -o memory_optimization 03_memory_optimization.cpp
 */

#include <iostream>
#include <vector>

int main() {
    std::cout << "=== GPU Memory Optimization ===\n\n";

    // Pattern 1: Coalesced Access
    std::cout << "1. COALESCED MEMORY ACCESS\n";
    std::cout << "   BAD:  Thread i accesses data[i * stride] (non-adjacent)\n";
    std::cout << "   GOOD: Thread i accesses data[i] (adjacent)\n";
    std::cout << "   Impact: 10-100x performance difference!\n\n";

    // Pattern 2: Shared Memory
    std::cout << "2. SHARED MEMORY (per-block cache)\n";
    std::cout << "   - 100x faster than global memory\n";
    std::cout << "   - Use for data reuse within block\n";
    std::cout << "   - Example: Matrix multiply tile\n";
    std::cout << "   __shared__ float tile[16][16];\n\n";

    // Pattern 3: Memory Padding
    std::cout << "3. AVOID BANK CONFLICTS\n";
    std::cout << "   Shared memory divided into 32 banks\n";
    std::cout << "   BAD:  tile[tid]     // All threads -> same bank\n";
    std::cout << "   GOOD: tile[tid + 1] // Threads spread across banks\n\n";

    // Pattern 4: Texture Memory
    std::cout << "4. TEXTURE MEMORY\n";
    std::cout << "   - Cached\n";
    std::cout << "   - 2D/3D locality\n";
    std::cout << "   - Hardware interpolation\n";
    std::cout << "   - Great for image processing\n\n";

    // Pattern 5: Constant Memory
    std::cout << "5. CONSTANT MEMORY\n";
    std::cout << "   - Cached\n";
    std::cout << "   - Fast when all threads read same address\n";
    std::cout << "   - Limited to 64KB\n";
    std::cout << "   - Example: Convolution kernels\n\n";

    std::cout << "=== Key Principles ===\n";
    std::cout << "1. Maximize coalesced access\n";
    std::cout << "2. Use shared memory for reuse\n";
    std::cout << "3. Minimize global memory transactions\n";
    std::cout << "4. Use appropriate memory type\n";
    std::cout << "5. Pad arrays to avoid bank conflicts\n";

    return 0;
}

/*
 * MEMORY HIERARCHY (fastest to slowest):
 * 1. Registers: 1 cycle, per-thread
 * 2. Shared Memory: ~5 cycles, per-block, 48-96 KB
 * 3. L1 Cache: ~25 cycles, automatic
 * 4. L2 Cache: ~200 cycles, automatic
 * 5. Global Memory: 400-800 cycles, GB sized
 *
 * TRY THIS:
 * 1. Profile with NVIDIA Nsight
 * 2. Check memory bandwidth utilization
 * 3. Optimize matrix multiply with shared memory
 * 4. Measure impact of coalesced access
 * 5. Test shared memory bank conflicts
 */
