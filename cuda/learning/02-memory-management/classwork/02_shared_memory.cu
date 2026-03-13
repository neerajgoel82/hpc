// Phase 2: Shared Memory Demonstration
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


#define TILE_SIZE 256

__global__ void reverseArrayShared(float *input, float *output, int n) {
    __shared__ float tile[TILE_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory
    if (idx < n) {
        tile[threadIdx.x] = input[idx];
    }
    __syncthreads();

    // Write in reverse order within block
    if (idx < n) {
        output[idx] = tile[blockDim.x - 1 - threadIdx.x];
    }
}

__global__ void reverseArrayGlobal(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int reverseIdx = (blockIdx.x + 1) * blockDim.x - 1 - threadIdx.x;
        if (reverseIdx < n) {
            output[idx] = input[reverseIdx];
        }
    }
}

int main() {
    printf("=== Shared Memory Performance ===\n\n");

    int n = 10000000;
    size_t size = n * sizeof(float);

    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    for (int i = 0; i < n; i++) h_input[i] = i;

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = TILE_SIZE;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Test shared memory
    cudaEventRecord(start);
    reverseArrayShared<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_shared;
    cudaEventElapsedTime(&time_shared, start, stop);

    // Test global memory
    cudaEventRecord(start);
    reverseArrayGlobal<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_global;
    cudaEventElapsedTime(&time_global, start, stop);

    printf("Array size: %d elements\n", n);
    printf("Shared memory: %.3f ms\n", time_shared);
    printf("Global memory: %.3f ms\n", time_global);
    printf("Speedup: %.2fx\n\n", time_global / time_shared);
    printf("Shared memory is faster due to on-chip access!\n");

    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
