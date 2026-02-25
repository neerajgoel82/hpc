// Phase 1: 2D Matrix Addition
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


__global__ void matrixAdd(float *a, float *b, float *c, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int idx = row * width + col;
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    printf("=== 2D Matrix Addition ===\n\n");

    int width = 4096;
    int height = 4096;
    size_t size = width * height * sizeof(float);

    printf("Matrix size: %dx%d (%.2f MB)\n\n", width, height, size / 1024.0 / 1024.0);

    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize matrices
    for (int i = 0; i < width * height; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // 2D grid and block configuration
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    printf("Grid: %dx%d blocks\n", gridDim.x, gridDim.y);
    printf("Block: %dx%d threads\n\n", blockDim.x, blockDim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixAdd<<<gridDim, blockDim>>>(d_a, d_b, d_c, width, height);
    cudaEventRecord(stop);

    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Verify
    bool correct = true;
    for (int i = 0; i < width * height; i++) {
        if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5) {
            correct = false;
            break;
        }
    }

    printf("Time: %.2f ms\n", milliseconds);
    printf("Bandwidth: %.2f GB/s\n", (3 * size / 1e9) / (milliseconds / 1000.0));
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");

    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
