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

__global__ void copyKernel(float *out, float *in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

void benchmark_transfer(size_t size, bool pinned) {
    float *h_data;

    if (pinned) {
        CUDA_CHECK(cudaHostAlloc(&h_data, size, cudaHostAllocDefault));
    } else {
        h_data = (float*)malloc(size);
    }

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("  %s memory: %.2f ms (%.2f GB/s)\n",
           pinned ? "Pinned  " : "Pageable",
           ms, (size / 1e9) / (ms / 1000.0));

    if (pinned) {
        cudaFreeHost(h_data);
    } else {
        free(h_data);
    }
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("=== Memory Bandwidth Benchmarking ===\n\n");

    size_t size = 256 * 1024 * 1024;  // 256 MB

    printf("Testing %.0f MB transfer:\n", size / 1024.0 / 1024.0);
    benchmark_transfer(size, false);  // Pageable
    benchmark_transfer(size, true);   // Pinned

    return 0;
}
