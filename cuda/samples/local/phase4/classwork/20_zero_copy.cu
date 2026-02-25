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

__global__ void processKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = sqrtf(data[idx]) * 2.0f + 1.0f;
    }
}

int main() {
    printf("=== Zero-Copy Memory ===\n\n");

    int n = 1 << 20;
    size_t size = n * sizeof(float);

    // Check if device supports mapped memory
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    if (!prop.canMapHostMemory) {
        printf("Device doesn't support mapped memory\n");
        return 1;
    }

    // Allocate zero-copy (mapped) memory
    float *h_data;
    CUDA_CHECK(cudaHostAlloc(&h_data, size, cudaHostAllocMapped));

    for (int i = 0; i < n; i++) h_data[i] = i + 1.0f;

    // Get device pointer to mapped memory
    float *d_data;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_data, h_data, 0));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    processKernel<<<blocks, threads>>>(d_data, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Verify - host can directly access results
    bool correct = true;
    for (int i = 0; i < 1000; i++) {
        float expected = sqrtf(i + 1.0f) * 2.0f + 1.0f;
        if (abs(h_data[i] - expected) > 1e-3) {
            correct = false;
            break;
        }
    }

    printf("Result: %s\n", correct ? "CORRECT" : "INCORRECT");
    printf("Time: %.2f ms\n", ms);
    printf("Bandwidth: %.2f GB/s\n", (size * 2 / 1e9) / (ms / 1000.0));
    printf("\nNote: Zero-copy avoids explicit cudaMemcpy!\n");
    printf("      Host and device share same physical memory\n");

    CUDA_CHECK(cudaFreeHost(h_data));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
