/*
 * MODULE 14: GPU Preparation
 * File: 01_cuda_concepts.cpp
 *
 * TOPIC: CUDA Concepts Simulated in C++
 *
 * CONCEPTS:
 * - Grid, Block, Thread hierarchy
 * - Thread indexing
 * - Kernel execution model
 * - Parallel execution patterns
 *
 * NOTE: This is C++ simulation of CUDA concepts
 *       Real CUDA code would use .cu files and nvcc compiler
 *
 * COMPILE: g++ -std=c++17 -pthread -o cuda_concepts 01_cuda_concepts.cpp
 */

#include <iostream>
#include <vector>
#include <thread>
#include <cmath>

// ========================================
// CUDA CONCEPTS IN C++
// ========================================

// Simulate CUDA dim3 type
struct dim3 {
    int x, y, z;
    dim3(int x_ = 1, int y_ = 1, int z_ = 1) : x(x_), y(y_), z(z_) {}
};

// Simulate CUDA thread indices (normally provided by CUDA runtime)
struct ThreadIdx {
    int x, y, z;
};

struct BlockIdx {
    int x, y, z;
};

struct BlockDim {
    int x, y, z;
};

struct GridDim {
    int x, y, z;
};

// These would be built-in CUDA variables
thread_local ThreadIdx threadIdx;
thread_local BlockIdx blockIdx;
thread_local BlockDim blockDim;
thread_local GridDim gridDim;

void demonstrateCUDAHierarchy() {
    std::cout << "\n=== CUDA Execution Hierarchy ===\n\n";

    std::cout << "GPU Organization:\n";
    std::cout << "  Grid (entire kernel launch)\n";
    std::cout << "    └─ Blocks (groups of threads)\n";
    std::cout << "         └─ Threads (individual workers)\n\n";

    std::cout << "Example Configuration:\n";
    std::cout << "  Grid:   2x2 blocks\n";
    std::cout << "  Block:  4x4 threads\n";
    std::cout << "  Total:  4 blocks × 16 threads = 64 threads\n\n";

    std::cout << "Typical CUDA Launch:\n";
    std::cout << "  dim3 blocks(2, 2);        // Grid dimensions\n";
    std::cout << "  dim3 threads(4, 4);       // Block dimensions\n";
    std::cout << "  kernel<<<blocks, threads>>>(args);\n\n";
}

void demonstrateThreadIndexing() {
    std::cout << "=== Thread Indexing ===\n\n";

    std::cout << "CUDA Built-in Variables:\n";
    std::cout << "  threadIdx.x, threadIdx.y, threadIdx.z  - Thread index within block\n";
    std::cout << "  blockIdx.x, blockIdx.y, blockIdx.z     - Block index within grid\n";
    std::cout << "  blockDim.x, blockDim.y, blockDim.z     - Block dimensions\n";
    std::cout << "  gridDim.x, gridDim.y, gridDim.z        - Grid dimensions\n\n";

    std::cout << "Computing Global Thread Index (1D):\n";
    std::cout << "  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n\n";

    std::cout << "Example:\n";
    std::cout << "  Block 0, Thread 2: idx = 0 * 256 + 2 = 2\n";
    std::cout << "  Block 1, Thread 2: idx = 1 * 256 + 2 = 258\n";
    std::cout << "  Block 2, Thread 2: idx = 2 * 256 + 2 = 514\n\n";
}

// ========================================
// SIMULATED CUDA KERNELS
// ========================================

// Kernel 1: Vector addition (simulated)
void vectorAddKernel_simulation(float* a, float* b, float* c, int n, int threadId) {
    int i = threadId;  // In real CUDA: blockIdx.x * blockDim.x + threadIdx.x
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

/*
 * REAL CUDA VERSION:
 *
 * __global__ void vectorAddKernel(float* a, float* b, float* c, int n) {
 *     int i = blockIdx.x * blockDim.x + threadIdx.x;
 *     if (i < n) {
 *         c[i] = a[i] + b[i];
 *     }
 * }
 *
 * // Launch:
 * int threadsPerBlock = 256;
 * int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
 * vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
 */

void demonstrateVectorAdd() {
    std::cout << "=== Vector Addition Kernel ===\n\n";

    const int n = 1000;
    std::vector<float> a(n, 1.0f);
    std::vector<float> b(n, 2.0f);
    std::vector<float> c(n, 0.0f);

    std::cout << "Problem: Add two vectors of size " << n << "\n\n";

    std::cout << "CUDA Configuration:\n";
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "  Threads per block: " << threadsPerBlock << "\n";
    std::cout << "  Blocks per grid: " << blocksPerGrid << "\n";
    std::cout << "  Total threads: " << (blocksPerGrid * threadsPerBlock) << "\n";
    std::cout << "  (Note: Some threads idle since " << (blocksPerGrid * threadsPerBlock)
              << " > " << n << ")\n\n";

    std::cout << "Simulating parallel execution with C++ threads:\n";
    std::vector<std::thread> threads;

    for (int i = 0; i < n; ++i) {
        threads.emplace_back(vectorAddKernel_simulation,
                             a.data(), b.data(), c.data(), n, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    // Verify
    bool correct = true;
    for (int i = 0; i < n; ++i) {
        if (std::abs(c[i] - 3.0f) > 1e-5f) {
            correct = false;
            break;
        }
    }

    std::cout << "Result: " << (correct ? "CORRECT" : "INCORRECT") << "\n";
    std::cout << "c[0] = " << c[0] << ", c[999] = " << c[999] << "\n\n";
}

// Kernel 2: Square kernel
void squareKernel_simulation(float* data, int n, int threadId) {
    int i = threadId;
    if (i < n) {
        data[i] = data[i] * data[i];
    }
}

void demonstrateSquare() {
    std::cout << "=== Square Kernel ===\n\n";

    const int n = 10;
    std::vector<float> data(n);
    for (int i = 0; i < n; ++i) {
        data[i] = i + 1;
    }

    std::cout << "Before: ";
    for (float x : data) std::cout << x << " ";
    std::cout << "\n";

    // Simulate parallel execution
    std::vector<std::thread> threads;
    for (int i = 0; i < n; ++i) {
        threads.emplace_back(squareKernel_simulation, data.data(), n, i);
    }
    for (auto& t : threads) {
        t.join();
    }

    std::cout << "After:  ";
    for (float x : data) std::cout << x << " ";
    std::cout << "\n\n";
}

void demonstrateMemoryModel() {
    std::cout << "=== CUDA Memory Hierarchy ===\n\n";

    std::cout << "Memory Types:\n\n";

    std::cout << "1. Global Memory:\n";
    std::cout << "   - Accessible by all threads\n";
    std::cout << "   - Large (GB)\n";
    std::cout << "   - Slow (400-800 cycles latency)\n";
    std::cout << "   - Example: float* d_data\n\n";

    std::cout << "2. Shared Memory:\n";
    std::cout << "   - Per-block memory\n";
    std::cout << "   - Fast (few cycles)\n";
    std::cout << "   - Limited (48-96 KB per block)\n";
    std::cout << "   - Declared: __shared__ float cache[256];\n\n";

    std::cout << "3. Registers:\n";
    std::cout << "   - Per-thread memory\n";
    std::cout << "   - Fastest (1 cycle)\n";
    std::cout << "   - Very limited (32K-64K per SM)\n";
    std::cout << "   - Automatic variables\n\n";

    std::cout << "4. Constant Memory:\n";
    std::cout << "   - Read-only\n";
    std::cout << "   - Cached\n";
    std::cout << "   - Fast for broadcasts\n";
    std::cout << "   - Example: __constant__ float params[10];\n\n";
}

void demonstrateGridBlockThread() {
    std::cout << "=== Grid/Block/Thread Example ===\n\n";

    std::cout << "Launching kernel with:\n";
    std::cout << "  Grid:  (2, 1, 1) - 2 blocks\n";
    std::cout << "  Block: (4, 1, 1) - 4 threads per block\n";
    std::cout << "  Total: 8 threads\n\n";

    std::cout << "Thread Mapping:\n";

    for (int blockId = 0; blockId < 2; ++blockId) {
        for (int threadId = 0; threadId < 4; ++threadId) {
            int globalIdx = blockId * 4 + threadId;
            std::cout << "  Block " << blockId << ", Thread " << threadId
                      << " -> Global Index " << globalIdx << "\n";
        }
    }
    std::cout << "\n";
}

void demonstrateCUDAWorkflow() {
    std::cout << "=== Typical CUDA Workflow ===\n\n";

    std::cout << "1. Allocate host memory:\n";
    std::cout << "   float* h_data = new float[n];\n\n";

    std::cout << "2. Allocate device (GPU) memory:\n";
    std::cout << "   float* d_data;\n";
    std::cout << "   cudaMalloc(&d_data, n * sizeof(float));\n\n";

    std::cout << "3. Copy data to GPU:\n";
    std::cout << "   cudaMemcpy(d_data, h_data, n * sizeof(float),\n";
    std::cout << "              cudaMemcpyHostToDevice);\n\n";

    std::cout << "4. Launch kernel:\n";
    std::cout << "   dim3 blocks(gridSize);\n";
    std::cout << "   dim3 threads(blockSize);\n";
    std::cout << "   myKernel<<<blocks, threads>>>(d_data, n);\n\n";

    std::cout << "5. Copy results back:\n";
    std::cout << "   cudaMemcpy(h_data, d_data, n * sizeof(float),\n";
    std::cout << "              cudaMemcpyDeviceToHost);\n\n";

    std::cout << "6. Free memory:\n";
    std::cout << "   cudaFree(d_data);\n";
    std::cout << "   delete[] h_data;\n\n";
}

int main() {
    std::cout << "=== MODULE 14: CUDA Concepts (Simulated in C++) ===\n";

    demonstrateCUDAHierarchy();
    demonstrateThreadIndexing();
    demonstrateGridBlockThread();
    demonstrateVectorAdd();
    demonstrateSquare();
    demonstrateMemoryModel();
    demonstrateCUDAWorkflow();

    std::cout << "=== Key Takeaways ===\n";
    std::cout << "1. GPU has thousands of threads (not dozens like CPU)\n";
    std::cout << "2. Threads organized in blocks, blocks in grid\n";
    std::cout << "3. Thread indexing is key to assigning work\n";
    std::cout << "4. Memory hierarchy is critical for performance\n";
    std::cout << "5. Kernels are parallel functions run by all threads\n";
    std::cout << "6. Same code runs on all threads (SIMT model)\n\n";

    std::cout << "=== Next Steps ===\n";
    std::cout << "To use real CUDA:\n";
    std::cout << "1. Install NVIDIA CUDA Toolkit\n";
    std::cout << "2. Get GPU with CUDA support\n";
    std::cout << "3. Write .cu files (CUDA C++)\n";
    std::cout << "4. Compile with nvcc\n";
    std::cout << "5. Run on GPU!\n";

    std::cout << "\n=== Program Complete ===\n";
    return 0;
}

/*
 * TRY THIS:
 * 1. Implement matrix multiplication kernel simulation
 * 2. Simulate shared memory usage
 * 3. Add 2D thread indexing
 * 4. Implement reduction pattern
 * 5. Simulate warp divergence effects
 *
 * CUDA RESOURCES:
 * - CUDA Programming Guide: docs.nvidia.com/cuda
 * - CUDA Samples: github.com/NVIDIA/cuda-samples
 * - Thrust Library: github.com/NVIDIA/thrust
 */
