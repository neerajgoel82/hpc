// Phase 3: Parallel Reduction
#include <stdio.h>
#include <stdlib.h>
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


__global__ void reductionOptimized(float *input, float *output, int n) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Load and add during load
    sdata[tid] = 0;
    if (idx < n) sdata[tid] = input[idx];
    if (idx + blockDim.x < n) sdata[tid] += input[idx + blockDim.x];
    __syncthreads();

    // Sequential addressing (no divergence)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

float reductionCPU(float *data, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

int main() {
    printf("=== Optimized Parallel Reduction ===\n\n");

    int n = 10000000;
    size_t size = n * sizeof(float);

    float *h_input = (float*)malloc(size);
    for (int i = 0; i < n; i++) h_input[i] = 1.0f;

    // CPU reduction
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    float cpu_sum = reductionCPU(h_input, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float cpu_time;
    cudaEventElapsedTime(&cpu_time, start, stop);

    // GPU reduction
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
    CUDA_CHECK(cudaMalloc(&d_output, blocksPerGrid * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    cudaEventRecord(start);
    reductionOptimized<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    cudaEventRecord(stop);

    float *h_output = (float*)malloc(blocksPerGrid * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventSynchronize(stop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // Final reduction on CPU
    float gpu_sum = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        gpu_sum += h_output[i];
    }

    printf("Array size: %d elements\n", n);
    printf("CPU sum: %.0f (%.2f ms)\n", cpu_sum, cpu_time);
    printf("GPU sum: %.0f (%.2f ms)\n", gpu_sum, gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("Verification: %s\n", (cpu_sum == gpu_sum) ? "PASSED" : "FAILED");

    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
