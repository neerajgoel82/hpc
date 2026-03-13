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


__global__ void scanKernel(float* data, int n) {
    __shared__ float temp[512];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    temp[tid] = (idx < n) ? data[idx] : 0.0f;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        float val = 0.0f;
        if (tid >= stride) val = temp[tid - stride];
        __syncthreads();
        if (tid >= stride) temp[tid] += val;
        __syncthreads();
    }

    if (idx < n) data[idx] = temp[tid];
}

int main() {
    printf("=== Prefix Sum (Inclusive Scan) ===\n\n");
    const int N = 512;
    size_t bytes = N * sizeof(float);

    float *h_data = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    scanKernel<<<1, N>>>(d_data, N);

    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));

    printf("First 10 values: ");
    for (int i = 0; i < 10; i++) printf("%.0f ", h_data[i]);
    printf("\nExpected:        1 2 3 4 5 6 7 8 9 10\n");

    free(h_data);
    cudaFree(d_data);
    return 0;
}
