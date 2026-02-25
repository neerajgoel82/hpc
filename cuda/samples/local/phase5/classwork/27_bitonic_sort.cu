#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


__global__ void bitonicSortKernel(float* data, int j, int k) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ixj = i ^ j;

    if (ixj > i) {
        if ((i & k) == 0) {
            if (data[i] > data[ixj]) {
                float temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        } else {
            if (data[i] < data[ixj]) {
                float temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}

int main() {
    printf("=== Bitonic Sort ===\n\n");
    const int N = 1024;  // Must be power of 2
    size_t bytes = N * sizeof(float);

    float *h_data = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_data[i] = (float)(rand() % 1000);

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int k = 2; k <= N; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            bitonicSortKernel<<<blocks, threads>>>(d_data, j, k);
        }
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));

    printf("Sorted %d elements in %.3f ms\n", N, ms);
    printf("First 10: ");
    for (int i = 0; i < 10; i++) printf("%.0f ", h_data[i]);
    printf("\nLast 10:  ");
    for (int i = N - 10; i < N; i++) printf("%.0f ", h_data[i]);
    printf("\n");

    free(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
