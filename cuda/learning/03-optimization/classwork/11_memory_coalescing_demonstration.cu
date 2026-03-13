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


__global__ void coalescedCopy(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx];
}

__global__ void stridedCopy(float* out, const float* in, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stridedIdx = idx * stride;
    if (stridedIdx < n) out[idx] = in[stridedIdx];
}

int main() {
    printf("=== Memory Coalescing Demonstration ===\n\n");
    const int N = 1 << 24;
    size_t bytes = N * sizeof(float);

    float *h_in = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = (float)i;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    coalescedCopy<<<blocks, threads>>>(d_out, d_in, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms1;
    CUDA_CHECK(cudaEventElapsedTime(&ms1, start, stop));
    float bw1 = (2.0f * bytes) / (ms1 * 1e6);
    printf("Coalesced: %.3f ms, BW: %.2f GB/s\n", ms1, bw1);

    int stride = 32;
    int strided_n = N / stride;
    blocks = (strided_n + threads - 1) / threads;

    CUDA_CHECK(cudaEventRecord(start));
    stridedCopy<<<blocks, threads>>>(d_out, d_in, N, stride);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms2;
    CUDA_CHECK(cudaEventElapsedTime(&ms2, start, stop));
    printf("Strided (stride=%d): %.3f ms, Slowdown: %.2fx\n",
           stride, ms2, ms2/ms1*(N/strided_n));

    free(h_in);
    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
