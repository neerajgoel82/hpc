
/*
 * Dynamic Parallelism - Parent kernel launches child kernels
 *
 * This demonstrates CUDA Dynamic Parallelism where GPU kernels can launch
 * other kernels directly without CPU involvement. This is useful for
 * recursive algorithms, adaptive mesh refinement, and dynamic workloads.
 *
 * Compile with: nvcc -arch=sm_35 -rdc=true 50_dynamic_parallelism.cu -o dynamic_parallelism
 * Note: Requires compute capability 3.5 or higher
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Child kernel - processes a sub-range of data
__global__ void childKernel(int *data, int offset, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Simple computation: square the index and add offset
        data[offset + idx] = (idx * idx) + offset;
    }
}

// Parent kernel - launches multiple child kernels
__global__ void parentKernel(int *data, int n, int childSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numChildren = (n + childSize - 1) / childSize;

    if (idx < numChildren) {
        int offset = idx * childSize;
        int size = min(childSize, n - offset);

        // Calculate grid and block dimensions for child kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

        // Launch child kernel from GPU
        childKernel<<<blocksPerGrid, threadsPerBlock>>>(data, offset, size);

        // Child kernel launches are asynchronous, but we can sync if needed
        // cudaDeviceSynchronize() would wait for child to complete
    }
}

// Verification kernel - simple operation for comparison
__global__ void simpleKernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Same computation as child kernel
        data[idx] = (idx * idx);
    }
}

int main() {
    printf("=== CUDA Dynamic Parallelism Demo ===\n\n");

    // Check if device supports dynamic parallelism
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    if (prop.major < 3 || (prop.major == 3 && prop.minor < 5)) {
        printf("Error: Dynamic Parallelism requires compute capability 3.5 or higher\n");
        return 1;
    }

    // Problem size
    const int N = 1024 * 1024;  // 1M elements
    const int childSize = 256;   // Each child processes 256 elements
    const size_t bytes = N * sizeof(int);

    printf("Array size: %d elements\n", N);
    printf("Child kernel size: %d elements\n\n", childSize);

    // Allocate host memory
    int *h_data = (int*)malloc(bytes);
    int *h_verify = (int*)malloc(bytes);

    // Allocate device memory
    int *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = 0;
    }

    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- Test 1: Dynamic Parallelism ---
    printf("Test 1: Dynamic Parallelism\n");
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    int numParents = (N + childSize - 1) / childSize;
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParents + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching %d parent threads in %d blocks\n", numParents, blocksPerGrid);

    CUDA_CHECK(cudaEventRecord(start));
    parentKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N, childSize);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Time: %.3f ms\n", ms);

    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));

    // --- Test 2: Simple kernel for comparison ---
    printf("\nTest 2: Simple Kernel (no dynamic parallelism)\n");
    CUDA_CHECK(cudaMemset(d_data, 0, bytes));

    threadsPerBlock = 256;
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching %d threads in %d blocks\n", N, blocksPerGrid);

    CUDA_CHECK(cudaEventRecord(start));
    simpleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Time: %.3f ms\n", ms);

    CUDA_CHECK(cudaMemcpy(h_verify, d_data, bytes, cudaMemcpyDeviceToHost));

    // Verify results
    printf("\nVerifying results...\n");
    int errors = 0;
    for (int i = 0; i < N && errors < 10; i++) {
        int expected = i * i;
        if (h_data[i] != expected) {
            printf("Mismatch at %d: got %d, expected %d\n", i, h_data[i], expected);
            errors++;
        }
    }

    if (errors == 0) {
        printf("SUCCESS: All results match!\n");
    } else {
        printf("ERRORS: Found %d mismatches\n", errors);
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);
    free(h_verify);

    printf("\nNote: Dynamic parallelism adds overhead. It's beneficial when:\n");
    printf("  - Workload is highly irregular or adaptive\n");
    printf("  - Cost of CPU-GPU synchronization is high\n");
    printf("  - Algorithm is naturally recursive\n");

    return 0;
}
