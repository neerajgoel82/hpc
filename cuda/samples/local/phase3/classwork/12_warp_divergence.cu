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


__global__ void divergentKernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (idx % 2 == 0) {
            float val = in[idx];
            for (int i = 0; i < 100; i++) val = sqrtf(val + 1.0f);
            out[idx] = val;
        } else {
            out[idx] = in[idx] * 2.0f;
        }
    }
}

__global__ void nonDivergentKernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        for (int i = 0; i < 100; i++) val = sqrtf(val + 1.0f);
        out[idx] = val;
    }
}

int main() {
    printf("=== Warp Divergence ===\n\n");
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *h_in = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = (float)i;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int threads = 256, blocks = (N + threads - 1) / threads;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    divergentKernel<<<blocks, threads>>>(d_out, d_in, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms1;
    CUDA_CHECK(cudaEventElapsedTime(&ms1, start, stop));

    CUDA_CHECK(cudaEventRecord(start));
    nonDivergentKernel<<<blocks, threads>>>(d_out, d_in, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms2;
    CUDA_CHECK(cudaEventElapsedTime(&ms2, start, stop));

    printf("Divergent kernel:     %.3f ms\n", ms1);
    printf("Non-divergent kernel: %.3f ms\n", ms2);
    printf("Speedup:              %.2fx\n", ms1/ms2);

    free(h_in);
    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
