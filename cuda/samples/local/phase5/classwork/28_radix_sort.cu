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


__global__ void radixSortKernel(unsigned int* data, unsigned int* temp,
                                 int n, int bit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned int val = data[idx];
        unsigned int b = (val >> bit) & 1;
        temp[idx] = b;
    }
}

int main() {
    printf("=== Radix Sort (Simplified) ===\n\n");
    const int N = 1024;
    size_t bytes = N * sizeof(unsigned int);

    unsigned int *h_data = (unsigned int*)malloc(bytes);
    for (int i = 0; i < N; i++) h_data[i] = rand() % 1000;

    unsigned int *d_data, *d_temp;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_temp, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    printf("Radix sort implementation (simplified demonstration)\n");
    printf("Full radix sort requires scan/prefix sum for efficient partitioning\n");
    printf("N = %d elements\n", N);

    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    printf("Sample values: %u %u %u %u %u\n",
           h_data[0], h_data[1], h_data[2], h_data[3], h_data[4]);

    free(h_data);
    cudaFree(d_data); cudaFree(d_temp);
    return 0;
}
