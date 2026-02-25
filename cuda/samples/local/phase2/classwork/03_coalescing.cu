// Phase 2: Memory Coalescing
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


__global__ void coalescedAccess(float *data, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = data[idx] * 2.0f;
    }
}

__global__ void stridedAccess(float *data, float *result, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (idx < n) {
        result[idx] = data[idx] * 2.0f;
    }
}

int main() {
    printf("=== Memory Coalescing Impact ===\n\n");

    int n = 10000000;
    size_t size = n * sizeof(float);

    float *d_data, *d_result;
    CUDA_CHECK(cudaMalloc(&d_data, size));
    CUDA_CHECK(cudaMalloc(&d_result, size));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Coalesced access
    cudaEventRecord(start);
    coalescedAccess<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_result, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_coalesced;
    cudaEventElapsedTime(&time_coalesced, start, stop);

    // Strided access (stride = 32)
    blocksPerGrid = (n/32 + threadsPerBlock - 1) / threadsPerBlock;
    cudaEventRecord(start);
    stridedAccess<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_result, n, 32);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_strided;
    cudaEventElapsedTime(&time_strided, start, stop);

    float bandwidth_coalesced = (size * 2 / 1e9) / (time_coalesced / 1000.0);
    float bandwidth_strided = (size * 2 / 1e9) / (time_strided / 1000.0);

    printf("Coalesced access: %.3f ms (%.2f GB/s)\n", time_coalesced, bandwidth_coalesced);
    printf("Strided access:   %.3f ms (%.2f GB/s)\n", time_strided, bandwidth_strided);
    printf("Performance degradation: %.2fx slower\n\n", time_strided / time_coalesced);
    printf("KEY: Adjacent threads should access adjacent memory!\n");

    cudaFree(d_data); cudaFree(d_result);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
