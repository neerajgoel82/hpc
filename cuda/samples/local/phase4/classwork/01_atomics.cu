// Atomic Operations
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


__global__ void atomicHistogram(int *input, int *histogram, int n, int bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int bin = input[idx] % bins;
        atomicAdd(&histogram[bin], 1);
    }
}

int main() {
    printf("=== Atomic Histogram ===\n\n");
    int n = 1000000;
    int bins = 256;

    int *h_input = (int*)malloc(n * sizeof(int));
    int *h_histogram = (int*)calloc(bins, sizeof(int));

    for (int i = 0; i < n; i++) h_input[i] = rand();

    int *d_input, *d_histogram;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_histogram, bins * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_histogram, 0, bins * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    atomicHistogram<<<(n+255)/256, 256>>>(d_input, d_histogram, n, bins);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_histogram, d_histogram, bins * sizeof(int), cudaMemcpyDeviceToHost));

    int total = 0;
    for (int i = 0; i < bins; i++) total += h_histogram[i];

    printf("Processed %d elements\n", n);
    printf("Time: %.2f ms\n", ms);
    printf("Verification: %s\n", (total == n) ? "PASSED" : "FAILED");

    free(h_input); free(h_histogram);
    cudaFree(d_input); cudaFree(d_histogram);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
