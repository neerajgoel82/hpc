#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void withoutShared(float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = -2; i <= 2; i++) {
            int pos = idx + i;
            if (pos >= 0 && pos < n) {
                sum += in[pos];  // Multiple global memory reads
            }
        }
        out[idx] = sum / 5.0f;
    }
}

__global__ void withShared(float *in, float *out, int n) {
    extern __shared__ float smem[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load into shared memory with halo
    if (idx < n) {
        smem[tid + 2] = in[idx];
        if (tid < 2 && idx >= 2) {
            smem[tid] = in[idx - 2];
        }
        if (tid >= blockDim.x - 2 && idx + 2 < n) {
            smem[tid + 4] = in[idx + 2];
        }
    }
    __syncthreads();

    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < 5; i++) {
            sum += smem[tid + i];  // Fast shared memory reads
        }
        out[idx] = sum / 5.0f;
    }
}

int main() {
    printf("=== Shared Memory Performance ===\n\n");

    int n = 1 << 20;
    size_t size = n * sizeof(float);

    float *h_in = (float*)malloc(size);
    for (int i = 0; i < n; i++) h_in[i] = i;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, size));
    CUDA_CHECK(cudaMalloc(&d_out, size));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Without shared memory
    cudaEventRecord(start);
    withoutShared<<<blocks, threads>>>(d_in, d_out, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float noSharedTime;
    cudaEventElapsedTime(&noSharedTime, start, stop);

    // With shared memory
    size_t smemSize = (threads + 4) * sizeof(float);
    cudaEventRecord(start);
    withShared<<<blocks, threads, smemSize>>>(d_in, d_out, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float sharedTime;
    cudaEventElapsedTime(&sharedTime, start, stop);

    printf("Without shared memory: %.2f ms\n", noSharedTime);
    printf("With shared memory: %.2f ms\n", sharedTime);
    printf("Speedup: %.2fx\n", noSharedTime / sharedTime);

    free(h_in);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
