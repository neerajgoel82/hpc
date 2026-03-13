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


#define FILTER_SIZE 9

__constant__ float d_filter[FILTER_SIZE];

__global__ void convolutionKernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= FILTER_SIZE/2 && idx < n - FILTER_SIZE/2) {
        float sum = 0.0f;
        for (int i = 0; i < FILTER_SIZE; i++)
            sum += input[idx - FILTER_SIZE/2 + i] * d_filter[i];
        output[idx] = sum;
    }
}

int main() {
    printf("=== Constant Memory ===\n\n");
    const int N = 1 << 24;
    size_t bytes = N * sizeof(float);

    float *h_input = (float*)malloc(bytes);
    float *h_filter = (float*)malloc(FILTER_SIZE * sizeof(float));

    for (int i = 0; i < N; i++) h_input[i] = (float)(rand() % 100);

    for (int i = 0; i < FILTER_SIZE; i++)
        h_filter[i] = expf(-(i - FILTER_SIZE/2) * (i - FILTER_SIZE/2) / 2.0f);

    float sum = 0;
    for (int i = 0; i < FILTER_SIZE; i++) sum += h_filter[i];
    for (int i = 0; i < FILTER_SIZE; i++) h_filter[i] /= sum;

    CUDA_CHECK(cudaMemcpyToSymbol(d_filter, h_filter, FILTER_SIZE * sizeof(float)));

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    convolutionKernel<<<(N + 255) / 256, 256>>>(d_input, d_output, N);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Convolution with constant memory: %.3f ms\n", ms);
    printf("Constant memory: 64KB, cached, broadcast to all threads\n");

    free(h_input); free(h_filter);
    cudaFree(d_input); cudaFree(d_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
