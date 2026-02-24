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


__global__ void kernel1(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < 50; i++)
            data[idx] = sqrtf(data[idx] + 1.0f);
    }
}

__global__ void kernel2(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = data[idx] * 2.0f + 1.0f;
}

int main() {
    printf("=== Profiling Demo ===\n\n");
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *h_data = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_data[i] = (float)i;

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    printf("Running kernels for profiling...\n");
    kernel1<<<(N + 255) / 256, 256>>>(d_data, N);
    kernel2<<<(N + 255) / 256, 256>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\nTo profile this application:\n");
    printf("  nvprof ./36_profiling_demo\n");
    printf("  ncu --set full ./36_profiling_demo\n");
    printf("  nsys profile ./36_profiling_demo\n");

    free(h_data);
    cudaFree(d_data);
    return 0;
}
