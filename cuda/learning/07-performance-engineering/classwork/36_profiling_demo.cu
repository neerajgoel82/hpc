#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Fast kernel
__global__ void fastKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

// Memory-bound kernel
__global__ void memoryBoundKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] + input[(idx + 1) % n] + input[(idx + 2) % n];
    }
}

// Compute-bound kernel
__global__ void computeBoundKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < 100; i++) {
            val = sqrtf(val + 1.0f);
        }
        data[idx] = val;
    }
}

int main() {
    printf("=== Profiling Demo ===\n\n");
    printf("Compile with: nvcc -arch=sm_70 -lineinfo 36_profiling_demo.cu\n");
    printf("Profile with: nvprof ./a.out\n");
    printf("Or use: nsys profile ./a.out\n\n");

    int n = 1 << 20;
    size_t size = n * sizeof(float);

    float *h_data = (float*)malloc(size);
    for (int i = 0; i < n; i++) h_data[i] = i + 1.0f;

    float *d_data, *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_data, size));
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));

    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, h_data, size, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Test 1: Fast kernel
    cudaEventRecord(start);
    fastKernel<<<blocks, threads>>>(d_data, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float fastTime;
    cudaEventElapsedTime(&fastTime, start, stop);

    // Test 2: Memory-bound
    cudaEventRecord(start);
    memoryBoundKernel<<<blocks, threads>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float memTime;
    cudaEventElapsedTime(&memTime, start, stop);

    // Test 3: Compute-bound
    cudaEventRecord(start);
    computeBoundKernel<<<blocks, threads>>>(d_data, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float compTime;
    cudaEventElapsedTime(&compTime, start, stop);

    printf("Results:\n");
    printf("  Fast kernel:         %.2f ms\n", fastTime);
    printf("  Memory-bound kernel: %.2f ms\n", memTime);
    printf("  Compute-bound kernel: %.2f ms\n", compTime);
    printf("\nUse profiler to see detailed metrics!\n");

    free(h_data);
    cudaFree(d_data);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
