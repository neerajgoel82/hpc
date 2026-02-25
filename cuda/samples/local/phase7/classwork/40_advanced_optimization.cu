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


__global__ void optimizedKernel(const float* __restrict__ in,
                                float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int index = idx * 4 + i;
        if (index < n) {
            float val = in[index];
            val = val * 2.0f + 1.0f;
            out[index] = val;
        }
    }
}

int main() {
    printf("=== Advanced Optimization ===\n\n");
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *h_in = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = (float)i;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    optimizedKernel<<<(N/4 + 255) / 256, 256>>>(d_in, d_out, N);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Optimized kernel time: %.3f ms\n", ms);
    printf("\nOptimizations used:\n");
    printf("  - __restrict__ pointers\n");
    printf("  - Loop unrolling (#pragma unroll)\n");
    printf("  - Vector loads (processing 4 elements/thread)\n");

    free(h_in);
    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
