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


__global__ void standardMath(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        data[idx] = sqrtf(x) + sinf(x) + cosf(x) + expf(x);
    }
}

__global__ void fastMath(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        data[idx] = __fsqrt_rn(x) + __sinf(x) + __cosf(x) + __expf(x);
    }
}

int main() {
    printf("=== Fast Math ===\n\n");
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *h_data = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_data[i] = (float)(i % 100) + 1.0f;

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(start));
    standardMath<<<(N + 255) / 256, 256>>>(d_data, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms1;
    CUDA_CHECK(cudaEventElapsedTime(&ms1, start, stop));

    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(start));
    fastMath<<<(N + 255) / 256, 256>>>(d_data, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms2;
    CUDA_CHECK(cudaEventElapsedTime(&ms2, start, stop));

    printf("Standard math: %.3f ms\n", ms1);
    printf("Fast math:     %.3f ms\n", ms2);
    printf("Speedup:       %.2fx\n", ms1/ms2);
    printf("\nFast math trades accuracy for speed\n");
    printf("Compile with --use_fast_math for automatic fast math\n");

    free(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
