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

int main() {
    printf("=== Memory Basics and Data Transfer ===\n\n");

    size_t sizes[] = {1 << 20, 1 << 22, 1 << 24};  // 1MB, 4MB, 16MB

    for (int i = 0; i < 3; i++) {
        size_t size = sizes[i];

        // Allocate host memory
        float *h_data = (float*)malloc(size);
        for (size_t j = 0; j < size / sizeof(float); j++) {
            h_data[j] = j;
        }

        // Allocate device memory
        float *d_data;
        CUDA_CHECK(cudaMalloc(&d_data, size));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Host to Device
        cudaEventRecord(start);
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float h2d_ms;
        cudaEventElapsedTime(&h2d_ms, start, stop);

        // Device to Host
        cudaEventRecord(start);
        CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float d2h_ms;
        cudaEventElapsedTime(&d2h_ms, start, stop);

        printf("Size: %.2f MB\n", size / 1024.0 / 1024.0);
        printf("  H2D: %.2f ms (%.2f GB/s)\n", h2d_ms, (size / 1e9) / (h2d_ms / 1000.0));
        printf("  D2H: %.2f ms (%.2f GB/s)\n", d2h_ms, (size / 1e9) / (d2h_ms / 1000.0));
        printf("\n");

        free(h_data);
        cudaFree(d_data);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}
