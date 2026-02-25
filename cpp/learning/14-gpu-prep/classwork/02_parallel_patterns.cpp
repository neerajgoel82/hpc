/*
 * MODULE 14: GPU Preparation
 * File: 02_parallel_patterns.cpp
 *
 * TOPIC: Common GPU Parallel Patterns
 *
 * COMPILE: g++ -std=c++17 -pthread -o parallel_patterns 02_parallel_patterns.cpp
 */

#include <iostream>
#include <vector>
#include <thread>
#include <numeric>
#include <algorithm>
#include <cmath>

// ========================================
// PATTERN 1: MAP
// Apply operation to each element independently
// ========================================
void mapPattern() {
    std::cout << "\n=== MAP Pattern ===\n";
    std::cout << "Apply operation to each element independently\n\n";

    std::vector<float> input{1, 2, 3, 4, 5};
    std::vector<float> output(input.size());

    std::cout << "Operation: x = x * 2 + 1\n";
    std::cout << "Input:  ";
    for (float x : input) std::cout << x << " ";
    std::cout << "\n";

    // Sequential
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] * 2.0f + 1.0f;
    }

    std::cout << "Output: ";
    for (float x : output) std::cout << x << " ";
    std::cout << "\n\n";

    std::cout << "CUDA Kernel:\n";
    std::cout << R"(
    __global__ void mapKernel(float* in, float* out, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            out[i] = in[i] * 2.0f + 1.0f;  // Each thread independent
        }
    }
)" << "\n";

    std::cout << "Key: No dependencies between threads!\n\n";
}

// ========================================
// PATTERN 2: REDUCE
// Combine all elements into single value
// ========================================
void reducePattern() {
    std::cout << "=== REDUCE Pattern ===\n";
    std::cout << "Combine all elements into single value (sum, max, min, etc.)\n\n";

    std::vector<int> data{1, 2, 3, 4, 5, 6, 7, 8};

    std::cout << "Operation: Sum all elements\n";
    std::cout << "Data: ";
    for (int x : data) std::cout << x << " ";
    std::cout << "\n";

    int sum = std::accumulate(data.begin(), data.end(), 0);
    std::cout << "Sum: " << sum << "\n\n";

    std::cout << "Tree-based parallel reduction:\n";
    std::cout << "Step 1: [1 2 3 4 5 6 7 8]\n";
    std::cout << "Step 2: [3   7   11  15  ]  // Pairs added\n";
    std::cout << "Step 3: [10      26      ]  // Pairs added\n";
    std::cout << "Step 4: [36             ]  // Final sum\n\n";

    std::cout << "CUDA Kernel (simplified):\n";
    std::cout << R"(
    __global__ void reduceSum(int* data, int* result, int n) {
        __shared__ int cache[256];
        int tid = threadIdx.x;
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        // Load to shared memory
        cache[tid] = (i < n) ? data[i] : 0;
        __syncthreads();

        // Tree reduction in shared memory
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                cache[tid] += cache[tid + stride];
            }
            __syncthreads();
        }

        // Thread 0 writes block result
        if (tid == 0) {
            atomicAdd(result, cache[0]);
        }
    }
)" << "\n";

    std::cout << "Key: Uses shared memory + synchronization!\n\n";
}

// ========================================
// PATTERN 3: SCAN (Prefix Sum)
// Each output is sum of all previous inputs
// ========================================
void scanPattern() {
    std::cout << "=== SCAN Pattern (Prefix Sum) ===\n";
    std::cout << "Each element is sum of all previous elements\n\n";

    std::vector<int> input{1, 2, 3, 4, 5};
    std::vector<int> output(input.size());

    std::cout << "Input:  ";
    for (int x : input) std::cout << x << " ";
    std::cout << "\n";

    // Exclusive scan
    output[0] = 0;
    for (size_t i = 1; i < input.size(); ++i) {
        output[i] = output[i-1] + input[i-1];
    }

    std::cout << "Output: ";
    for (int x : output) std::cout << x << " ";
    std::cout << " (exclusive scan)\n\n";

    std::cout << "Use cases:\n";
    std::cout << "  - Memory allocation (find offsets)\n";
    std::cout << "  - Stream compaction\n";
    std::cout << "  - Sorting algorithms\n\n";

    std::cout << "Note: Efficient parallel scan is complex!\n";
    std::cout << "      (Blelloch scan, work-efficient, O(n) work)\n\n";
}

// ========================================
// PATTERN 4: STENCIL
// Each output depends on neighborhood of inputs
// ========================================
void stencilPattern() {
    std::cout << "=== STENCIL Pattern ===\n";
    std::cout << "Each element computed from neighbors (e.g., blur, Laplacian)\n\n";

    std::vector<float> input{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> output(input.size(), 0);

    std::cout << "Operation: 3-point average (left + center + right) / 3\n";
    std::cout << "Input:  ";
    for (float x : input) std::cout << x << " ";
    std::cout << "\n";

    // 3-point stencil
    for (size_t i = 1; i < input.size() - 1; ++i) {
        output[i] = (input[i-1] + input[i] + input[i+1]) / 3.0f;
    }

    std::cout << "Output: ";
    for (float x : output) std::cout << x << " ";
    std::cout << "\n\n";

    std::cout << "CUDA Kernel:\n";
    std::cout << R"(
    __global__ void stencil3pt(float* in, float* out, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i > 0 && i < n - 1) {
            out[i] = (in[i-1] + in[i] + in[i+1]) / 3.0f;
        }
    }
)" << "\n";

    std::cout << "Use cases:\n";
    std::cout << "  - Image blur/sharpen\n";
    std::cout << "  - Finite difference methods\n";
    std::cout << "  - Convolution\n\n";
}

// ========================================
// PATTERN 5: SCATTER / GATHER
// ========================================
void scatterGatherPattern() {
    std::cout << "=== SCATTER / GATHER Pattern ===\n\n";

    std::cout << "GATHER: Read from multiple locations\n";
    std::cout << "  Example: texture sampling with interpolation\n";
    std::cout << "  out[i] = in[indices[i]]\n\n";

    std::cout << "SCATTER: Write to multiple locations\n";
    std::cout << "  Example: particle rasterization\n";
    std::cout << "  out[indices[i]] = value[i]\n";
    std::cout << "  Problem: Race conditions! (need atomics)\n\n";

    std::vector<int> data{10, 20, 30, 40, 50};
    std::vector<int> indices{4, 2, 0, 3, 1};
    std::vector<int> gathered(5);

    // Gather
    for (size_t i = 0; i < indices.size(); ++i) {
        gathered[i] = data[indices[i]];
    }

    std::cout << "Data:     ";
    for (int x : data) std::cout << x << " ";
    std::cout << "\n";

    std::cout << "Indices:  ";
    for (int x : indices) std::cout << x << " ";
    std::cout << "\n";

    std::cout << "Gathered: ";
    for (int x : gathered) std::cout << x << " ";
    std::cout << "\n\n";
}

// ========================================
// PATTERN 6: HISTOGRAM
// Count occurrences in bins
// ========================================
void histogramPattern() {
    std::cout << "=== HISTOGRAM Pattern ===\n";
    std::cout << "Count occurrences in bins (requires atomics on GPU)\n\n";

    std::vector<int> data{0, 1, 2, 0, 1, 0, 3, 2, 1, 0};
    std::vector<int> histogram(4, 0);

    std::cout << "Data: ";
    for (int x : data) std::cout << x << " ";
    std::cout << "\n";

    // Count
    for (int x : data) {
        histogram[x]++;
    }

    std::cout << "Histogram: ";
    for (int x : histogram) std::cout << x << " ";
    std::cout << "\n\n";

    std::cout << "CUDA Kernel:\n";
    std::cout << R"(
    __global__ void histogram(int* data, int* hist, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            atomicAdd(&hist[data[i]], 1);  // Atomic increment!
        }
    }
)" << "\n";

    std::cout << "Key: Use atomicAdd to avoid race conditions!\n\n";
}

int main() {
    std::cout << "=== MODULE 14: GPU Parallel Patterns ===\n";

    mapPattern();
    reducePattern();
    scanPattern();
    stencilPattern();
    scatterGatherPattern();
    histogramPattern();

    std::cout << "=== Pattern Summary ===\n\n";
    std::cout << "1. MAP: Independent operations (easiest)\n";
    std::cout << "2. REDUCE: Combine to single value (needs synchronization)\n";
    std::cout << "3. SCAN: Prefix sum (complex but useful)\n";
    std::cout << "4. STENCIL: Neighborhood operations (shared memory helps)\n";
    std::cout << "5. SCATTER/GATHER: Indirect access (gather safe, scatter needs atomics)\n";
    std::cout << "6. HISTOGRAM: Counting (needs atomics)\n\n";

    std::cout << "=== Performance Tips ===\n";
    std::cout << "- Map: Perfect parallelism, no dependencies\n";
    std::cout << "- Reduce: Use shared memory for first stage\n";
    std::cout << "- Scan: Use library implementation (Thrust, CUB)\n";
    std::cout << "- Stencil: Use shared memory to cache neighbors\n";
    std::cout << "- Scatter: Minimize atomics (expensive)\n";
    std::cout << "- Histogram: Use local histograms + merge\n";

    std::cout << "\n=== Program Complete ===\n";
    return 0;
}

/*
 * TRY THIS:
 * 1. Implement parallel matrix transpose
 * 2. Code reduction with different operations (max, min)
 * 3. Implement 2D stencil (image blur)
 * 4. Build parallel sort using patterns
 * 5. Create stream compaction with scan
 *
 * RESOURCES:
 * - "Programming Massively Parallel Processors" (Hwu, Kirk, Hajj)
 * - NVIDIA CUDA Best Practices Guide
 * - Thrust library (thrust::reduce, thrust::scan, etc.)
 */
