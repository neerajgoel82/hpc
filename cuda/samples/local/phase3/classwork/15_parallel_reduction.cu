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


__global__ void reduceKernel(const float* in, float* out, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < n)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = smem[0];
}

int main() {
    printf("=== Parallel Reduction ===\n\n");
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *h_in = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    float *d_in, *d_out;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, blocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    reduceKernel<<<blocks, threads>>>(d_in, d_out, N);

    float *h_out = (float*)malloc(blocks * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, blocks * sizeof(float), cudaMemcpyDeviceToHost));

    float sum = 0;
    for (int i = 0; i < blocks; i++) sum += h_out[i];

    printf("Sum: %.0f (expected %.0f) - %s\n", sum, (float)N,
           (fabsf(sum - N) < 0.1f) ? "PASS" : "FAIL");

    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
