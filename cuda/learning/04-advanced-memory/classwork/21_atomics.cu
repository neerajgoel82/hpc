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


__global__ void atomicAddKernel(int* counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) atomicAdd(counter, 1);
}

__global__ void atomicMinMaxKernel(const int* data, int* min_val, int* max_val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicMin(min_val, data[idx]);
        atomicMax(max_val, data[idx]);
    }
}

int main() {
    printf("=== Atomic Operations ===\n\n");
    const int N = 1 << 20;

    printf("Test 1: atomicAdd\n");
    int *d_counter;
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    atomicAddKernel<<<(N + 255) / 256, 256>>>(d_counter, N);

    int h_counter;
    CUDA_CHECK(cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    printf("  Counter: %d (expected %d) - %s\n\n", h_counter, N,
           (h_counter == N) ? "PASS" : "FAIL");

    printf("Test 2: atomicMin/Max\n");
    int *h_data = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) h_data[i] = rand() % 1000;

    int *d_data, *d_min, *d_max;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_min, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_max, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));

    int init_min = 999999, init_max = -1;
    CUDA_CHECK(cudaMemcpy(d_min, &init_min, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_max, &init_max, sizeof(int), cudaMemcpyHostToDevice));

    atomicMinMaxKernel<<<(N + 255) / 256, 256>>>(d_data, d_min, d_max, N);

    int h_min, h_max;
    CUDA_CHECK(cudaMemcpy(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost));

    int ref_min = h_data[0], ref_max = h_data[0];
    for (int i = 1; i < N; i++) {
        if (h_data[i] < ref_min) ref_min = h_data[i];
        if (h_data[i] > ref_max) ref_max = h_data[i];
    }

    printf("  Min: %d (expected %d) - %s\n", h_min, ref_min,
           (h_min == ref_min) ? "PASS" : "FAIL");
    printf("  Max: %d (expected %d) - %s\n", h_max, ref_max,
           (h_max == ref_max) ? "PASS" : "FAIL");

    free(h_data);
    cudaFree(d_counter); cudaFree(d_data); cudaFree(d_min); cudaFree(d_max);
    return 0;
}
