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


__global__ void workKernel(float* data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < iterations; i++)
            val = sqrtf(val + 1.0f);
        data[idx] = val;
    }
}

int main() {
    printf("=== Events and Timing ===\n\n");
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *h_data = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_data[i] = (float)i;

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop, kernel1, kernel2;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventCreate(&kernel1));
    CUDA_CHECK(cudaEventCreate(&kernel2));

    CUDA_CHECK(cudaEventRecord(start));

    workKernel<<<(N + 255) / 256, 256>>>(d_data, N, 50);
    CUDA_CHECK(cudaEventRecord(kernel1));

    workKernel<<<(N + 255) / 256, 256>>>(d_data, N, 100);
    CUDA_CHECK(cudaEventRecord(kernel2));

    workKernel<<<(N + 255) / 256, 256>>>(d_data, N, 150);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_total, ms_k1, ms_k2, ms_k3;
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, start, stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_k1, start, kernel1));
    CUDA_CHECK(cudaEventElapsedTime(&ms_k2, kernel1, kernel2));
    CUDA_CHECK(cudaEventElapsedTime(&ms_k3, kernel2, stop));

    printf("Kernel 1 (50 iters):  %.3f ms\n", ms_k1);
    printf("Kernel 2 (100 iters): %.3f ms\n", ms_k2);
    printf("Kernel 3 (150 iters): %.3f ms\n", ms_k3);
    printf("Total time:           %.3f ms\n", ms_total);

    free(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaEventDestroy(kernel1); cudaEventDestroy(kernel2);
    return 0;
}
