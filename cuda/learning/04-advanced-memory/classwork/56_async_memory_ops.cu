/*
 * Async Memory Operations - Stream-ordered memory allocation
 *
 * CUDA 11.2+ introduced stream-ordered memory operations that eliminate
 * synchronization overhead of traditional cudaMalloc/cudaFree.
 *
 * Benefits:
 * - No implicit synchronization
 * - Memory pooling for reduced overhead
 * - Better integration with CUDA graphs
 * - Improved performance for dynamic allocation patterns
 *
 * Requires: Compute capability 7.0+, CUDA 11.2+
 * Compile: nvcc -arch=sm_70 -O2 56_async_memory_ops.cu -o async_memory
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Simple kernel to process data
__global__ void processData(float *data, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * scale + 1.0f;
    }
}

// Traditional synchronous allocation
void traditionalAllocation(int n, int iterations) {
    printf("\n=== Traditional Synchronous Allocation ===\n");

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; i++) {
        float *d_data;
        CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(float)));

        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        processData<<<blocksPerGrid, threadsPerBlock>>>(d_data, n, 2.0f);

        CUDA_CHECK(cudaFree(d_data));
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Time: %.3f ms (%.3f ms per iteration)\n", ms, ms / iterations);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// Stream-ordered async allocation
void asyncAllocation(int n, int iterations) {
    printf("\n=== Stream-Ordered Async Allocation ===\n");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));

    for (int i = 0; i < iterations; i++) {
        float *d_data;
        // Async allocation - no synchronization
        CUDA_CHECK(cudaMallocAsync(&d_data, n * sizeof(float), stream));

        // Launch kernel in same stream
        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        processData<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, n, 2.0f);

        // Async free - no synchronization
        CUDA_CHECK(cudaFreeAsync(d_data, stream));
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Time: %.3f ms (%.3f ms per iteration)\n", ms, ms / iterations);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// Memory pool demonstration
void memoryPoolDemo(int n) {
    printf("\n=== Memory Pool Management ===\n");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Get default memory pool
    cudaMemPool_t mempool;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&mempool, device));

    // Query pool properties
    size_t threshold;
    CUDA_CHECK(cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold));
    printf("Pool release threshold: %zu bytes (%.2f MB)\n",
           threshold, threshold / (1024.0 * 1024.0));

    // Allocate from pool
    float *d_data1, *d_data2;
    CUDA_CHECK(cudaMallocAsync(&d_data1, n * sizeof(float), stream));
    printf("Allocated array 1: %zu bytes (%.2f MB)\n",
           n * sizeof(float), n * sizeof(float) / (1024.0 * 1024.0));

    CUDA_CHECK(cudaMallocAsync(&d_data2, n * sizeof(float), stream));
    printf("Allocated array 2: %zu bytes (%.2f MB)\n",
           n * sizeof(float), n * sizeof(float) / (1024.0 * 1024.0));

    // Free back to pool (doesn't return to OS immediately)
    CUDA_CHECK(cudaFreeAsync(d_data1, stream));
    CUDA_CHECK(cudaFreeAsync(d_data2, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Trim pool to release memory to OS
    CUDA_CHECK(cudaMemPoolTrimTo(mempool, 0));
    printf("Memory pool trimmed\n");

    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main() {
    printf("=== Async Memory Operations Demo ===\n");

    // Check device capability
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("\nDevice: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    if (prop.major < 7) {
        printf("Error: Async memory operations require compute capability 7.0+\n");
        return 1;
    }

    const int n = 1024 * 1024;  // 1M elements
    const int iterations = 100;

    printf("\nArray size: %d elements (%.2f MB)\n", n, n * sizeof(float) / (1024.0 * 1024.0));
    printf("Iterations: %d\n", iterations);

    // Benchmark traditional vs async
    traditionalAllocation(n, iterations);
    asyncAllocation(n, iterations);

    // Demonstrate memory pool
    memoryPoolDemo(n * 4);

    printf("\n=== Key Takeaways ===\n");
    printf("1. cudaMallocAsync/cudaFreeAsync eliminate synchronization overhead\n");
    printf("2. Memory pools reduce allocation latency through caching\n");
    printf("3. Stream-ordered operations integrate seamlessly with kernels\n");
    printf("4. Especially beneficial for dynamic allocation patterns\n");
    printf("5. Async ops are required for optimal CUDA graph performance\n");

    return 0;
}
