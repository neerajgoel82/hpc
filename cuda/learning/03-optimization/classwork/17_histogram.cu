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


__global__ void histogramKernel(const unsigned char* data, int* hist, int n) {
    __shared__ int smem[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < 256) smem[tid] = 0;
    __syncthreads();

    if (idx < n) atomicAdd(&smem[data[idx]], 1);
    __syncthreads();

    if (tid < 256) atomicAdd(&hist[tid], smem[tid]);
}

int main() {
    printf("=== Histogram ===\n\n");
    const int N = 1 << 20;

    unsigned char *h_data = (unsigned char*)malloc(N);
    int *h_hist = (int*)calloc(256, sizeof(int));

    srand(42);
    for (int i = 0; i < N; i++) h_data[i] = rand() % 256;

    unsigned char *d_data;
    int *d_hist;
    CUDA_CHECK(cudaMalloc(&d_data, N));
    CUDA_CHECK(cudaMalloc(&d_hist, 256 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_hist, 0, 256 * sizeof(int)));

    histogramKernel<<<(N + 255) / 256, 256>>>(d_data, d_hist, N);

    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost));

    printf("Sample bins [0-4]: %d %d %d %d %d\n",
           h_hist[0], h_hist[1], h_hist[2], h_hist[3], h_hist[4]);

    int total = 0;
    for (int i = 0; i < 256; i++) total += h_hist[i];
    printf("Total count: %d (expected %d) - %s\n", total, N,
           (total == N) ? "PASS" : "FAIL");

    free(h_data); free(h_hist);
    cudaFree(d_data); cudaFree(d_hist);
    return 0;
}
