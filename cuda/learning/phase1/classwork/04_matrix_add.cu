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
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

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

    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

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

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixAdd<<<blocks, threads>>>(d_a, d_b, d_c, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify
    bool correct = true;
    for (int i = 0; i < 1000; i++) {
        int idx = rand() % (width * height);
        if (abs(h_c[idx] - (h_a[idx] + h_b[idx])) > 1e-5) {
            correct = false;
            break;
        }
    }

    printf("Matrix: %dx%d\n", width, height);
    printf("Block: %dx%d, Grid: %dx%d\n", threads.x, threads.y, blocks.x, blocks.y);
    printf("Time: %.2f ms\n", ms);
    printf("Bandwidth: %.2f GB/s\n", (size * 3 / 1e9) / (ms / 1000.0));
    printf("Result: %s\n", correct ? "CORRECT" : "INCORRECT");

    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
