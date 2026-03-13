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


__global__ void warpReduceKernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int warpId = idx / 32;

    float val = (idx < n) ? in[idx] : 0.0f;

    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);

    if (lane == 0 && warpId < (n + 31) / 32)
        out[warpId] = val;
}

__global__ void warpScanKernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;

    float val = (idx < n) ? in[idx] : 0.0f;

    for (int offset = 1; offset < 32; offset *= 2) {
        float temp = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) val += temp;
    }

    if (idx < n) out[idx] = val;
}

int main() {
    printf("=== Warp Shuffle Operations ===\n\n");
    const int N = 1024;
    size_t bytes = N * sizeof(float);

    float *h_in = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    float *d_in, *d_out;
    int numWarps = (N + 31) / 32;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, numWarps * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    printf("Warp Reduction:\n");
    warpReduceKernel<<<(N + 255) / 256, 256>>>(d_in, d_out, N);

    float *h_out = (float*)malloc(numWarps * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, numWarps * sizeof(float), cudaMemcpyDeviceToHost));

    printf("  First 5 warp sums: ");
    for (int i = 0; i < 5 && i < numWarps; i++) printf("%.0f ", h_out[i]);
    printf("\n  Expected: 32 for full warps\n\n");

    printf("Warp Scan (prefix sum):\n");
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    warpScanKernel<<<(N + 255) / 256, 256>>>(d_in, d_out, N);
    CUDA_CHECK(cudaMemcpy(h_in, d_out, bytes, cudaMemcpyDeviceToHost));

    printf("  First 10 values: ");
    for (int i = 0; i < 10; i++) printf("%.0f ", h_in[i]);
    printf("\n  Expected: 1 2 3 4 5 6 7 8 9 10\n");

    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
