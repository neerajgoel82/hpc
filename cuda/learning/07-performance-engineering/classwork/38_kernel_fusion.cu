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

// Separate kernels (not fused)
__global__ void kernel1(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

__global__ void kernel2(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] + 10.0f;
    }
}

__global__ void kernel3(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = sqrtf(data[idx]);
    }
}

// Fused kernel (all operations in one)
__global__ void fusedKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        val = val * 2.0f;      // Operation 1
        val = val + 10.0f;     // Operation 2
        val = sqrtf(val);      // Operation 3
        data[idx] = val;
    }
}

int main() {
    printf("=== Kernel Fusion ===\n\n");

    int n = 1 << 24;
    size_t size = n * sizeof(float);

    float *h_data = (float*)malloc(size);
    for (int i = 0; i < n; i++) h_data[i] = i + 1.0f;

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Test 1: Separate kernels
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    cudaEventRecord(start);
    kernel1<<<blocks, threads>>>(d_data, n);
    kernel2<<<blocks, threads>>>(d_data, n);
    kernel3<<<blocks, threads>>>(d_data, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float separateTime;
    cudaEventElapsedTime(&separateTime, start, stop);

    // Test 2: Fused kernel
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    cudaEventRecord(start);
    fusedKernel<<<blocks, threads>>>(d_data, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float fusedTime;
    cudaEventElapsedTime(&fusedTime, start, stop);

    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // Verify
    bool correct = true;
    for (int i = 0; i < 1000; i++) {
        float expected = sqrtf((i + 1.0f) * 2.0f + 10.0f);
        if (abs(h_data[i] - expected) > 1e-3) {
            correct = false;
            break;
        }
    }

    printf("Result: %s\n", correct ? "CORRECT" : "INCORRECT");
    printf("\nSeparate kernels: %.2f ms\n", separateTime);
    printf("Fused kernel:     %.2f ms\n", fusedTime);
    printf("Speedup:          %.2fx\n", separateTime / fusedTime);
    printf("\nFusion reduces:\n");
    printf("  - Global memory accesses\n");
    printf("  - Kernel launch overhead\n");
    printf("  - Device synchronization\n");

    free(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
