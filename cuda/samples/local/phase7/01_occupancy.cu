// Occupancy Tuning
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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


__global__ void kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        for (int i = 0; i < 100; i++) {
            x = sqrt(x + 1.0f);
        }
        data[idx] = x;
    }
}

int main() {
    printf("=== Occupancy Tuning ===\n\n");

    int n = 1000000;
    size_t size = n * sizeof(float);

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));

    int blockSizes[] = {32, 64, 128, 256, 512, 1024};

    printf("%-15s %-15s %-15s\n", "Block Size", "Time (ms)", "Bandwidth (GB/s)");
    printf("-----------------------------------------------\n");

    for (int i = 0; i < 6; i++) {
        int blockSize = blockSizes[i];
        int gridSize = (n + blockSize - 1) / blockSize;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        kernel<<<gridSize, blockSize>>>(d_data, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        float bandwidth = (size * 2 / 1e9) / (ms / 1000.0);

        printf("%-15d %-15.2f %-15.2f\n", blockSize, ms, bandwidth);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaFree(d_data);
    return 0;
}
