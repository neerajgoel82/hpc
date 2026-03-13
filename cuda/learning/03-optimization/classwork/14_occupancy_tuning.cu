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


template<int BLOCK_SIZE>
__global__ void occupancyTestKernel(float* out, const float* in, int n) {
    __shared__ float smem[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < n) smem[tid] = in[idx];
    __syncthreads();

    float val = smem[tid];
    for (int i = 0; i < 50; i++) val = sqrtf(val + 1.0f);

    if (idx < n) out[idx] = val;
}

void testOccupancy(int blockSize, int N) {
    size_t bytes = N * sizeof(float);
    float *h_in = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = (float)i;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int blocks = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    if (blockSize == 128) occupancyTestKernel<128><<<blocks, blockSize>>>(d_out, d_in, N);
    else if (blockSize == 256) occupancyTestKernel<256><<<blocks, blockSize>>>(d_out, d_in, N);
    else if (blockSize == 512) occupancyTestKernel<512><<<blocks, blockSize>>>(d_out, d_in, N);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Block size %4d: %.3f ms\n", blockSize, ms);

    free(h_in);
    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    printf("=== Occupancy Tuning ===\n\n");
    const int N = 1 << 24;
    int blockSizes[] = {128, 256, 512};

    for (int i = 0; i < 3; i++)
        testOccupancy(blockSizes[i], N);

    printf("\nOptimal occupancy depends on register and shared memory usage\n");
    return 0;
}
