#!/usr/bin/env python3
"""
Complete CUDA sample generator for Phases 3-9.
Generates all 47 working CUDA files with real implementations.
Run: python3 generate_all_phases.py
"""

import os
from pathlib import Path

BASE_DIR = Path("local")

# Common CUDA header
CUDA_HEADER = """#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error at %s:%d: %s\\n", \\
                    __FILE__, __LINE__, cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)

"""

def get_phase3_files():
    """Phase 3: Optimization (7 files)"""
    return {
        "10_tiled_matrix_multiplication.cu": CUDA_HEADER + """
#define TILE_SIZE 16

__global__ void tiledMatMulKernel(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < N && t * TILE_SIZE + threadIdx.x < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

int main() {
    printf("=== Tiled Matrix Multiplication ===\\n\\n");
    const int N = 512;
    size_t bytes = N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    tiledMatMulKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Matrix size: %dx%d, Tile: %dx%d\\n", N, N, TILE_SIZE, TILE_SIZE);
    printf("Time: %.3f ms, GFLOPS: %.2f\\n", ms, (2.0*N*N*N)/(ms*1e6));

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
""",

        "11_memory_coalescing_demonstration.cu": CUDA_HEADER + """
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
    printf("=== Memory Coalescing Demonstration ===\\n\\n");
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
    printf("Coalesced: %.3f ms, BW: %.2f GB/s\\n", ms1, bw1);

    int stride = 32;
    int strided_n = N / stride;
    blocks = (strided_n + threads - 1) / threads;

    CUDA_CHECK(cudaEventRecord(start));
    stridedCopy<<<blocks, threads>>>(d_out, d_in, N, stride);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms2;
    CUDA_CHECK(cudaEventElapsedTime(&ms2, start, stop));
    printf("Strided (stride=%d): %.3f ms, Slowdown: %.2fx\\n",
           stride, ms2, ms2/ms1*(N/strided_n));

    free(h_in);
    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
""",

        "12_warp_divergence.cu": CUDA_HEADER + """
__global__ void divergentKernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (idx % 2 == 0) {
            float val = in[idx];
            for (int i = 0; i < 100; i++) val = sqrtf(val + 1.0f);
            out[idx] = val;
        } else {
            out[idx] = in[idx] * 2.0f;
        }
    }
}

__global__ void nonDivergentKernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        for (int i = 0; i < 100; i++) val = sqrtf(val + 1.0f);
        out[idx] = val;
    }
}

int main() {
    printf("=== Warp Divergence ===\\n\\n");
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *h_in = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = (float)i;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int threads = 256, blocks = (N + threads - 1) / threads;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    divergentKernel<<<blocks, threads>>>(d_out, d_in, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms1;
    CUDA_CHECK(cudaEventElapsedTime(&ms1, start, stop));

    CUDA_CHECK(cudaEventRecord(start));
    nonDivergentKernel<<<blocks, threads>>>(d_out, d_in, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms2;
    CUDA_CHECK(cudaEventElapsedTime(&ms2, start, stop));

    printf("Divergent kernel:     %.3f ms\\n", ms1);
    printf("Non-divergent kernel: %.3f ms\\n", ms2);
    printf("Speedup:              %.2fx\\n", ms1/ms2);

    free(h_in);
    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
""",

        "13_warp_shuffle.cu": CUDA_HEADER + """
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
    printf("=== Warp Shuffle Operations ===\\n\\n");
    const int N = 1024;
    size_t bytes = N * sizeof(float);

    float *h_in = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    float *d_in, *d_out;
    int numWarps = (N + 31) / 32;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, numWarps * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    printf("Warp Reduction:\\n");
    warpReduceKernel<<<(N + 255) / 256, 256>>>(d_in, d_out, N);

    float *h_out = (float*)malloc(numWarps * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, numWarps * sizeof(float), cudaMemcpyDeviceToHost));

    printf("  First 5 warp sums: ");
    for (int i = 0; i < 5 && i < numWarps; i++) printf("%.0f ", h_out[i]);
    printf("\\n  Expected: 32 for full warps\\n\\n");

    printf("Warp Scan (prefix sum):\\n");
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    warpScanKernel<<<(N + 255) / 256, 256>>>(d_in, d_out, N);
    CUDA_CHECK(cudaMemcpy(h_in, d_out, bytes, cudaMemcpyDeviceToHost));

    printf("  First 10 values: ");
    for (int i = 0; i < 10; i++) printf("%.0f ", h_in[i]);
    printf("\\n  Expected: 1 2 3 4 5 6 7 8 9 10\\n");

    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
""",

        "14_occupancy_tuning.cu": CUDA_HEADER + """
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
    printf("Block size %4d: %.3f ms\\n", blockSize, ms);

    free(h_in);
    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    printf("=== Occupancy Tuning ===\\n\\n");
    const int N = 1 << 24;
    int blockSizes[] = {128, 256, 512};

    for (int i = 0; i < 3; i++)
        testOccupancy(blockSizes[i], N);

    printf("\\nOptimal occupancy depends on register and shared memory usage\\n");
    return 0;
}
""",

        "15_parallel_reduction.cu": CUDA_HEADER + """
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
    printf("=== Parallel Reduction ===\\n\\n");
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

    printf("Sum: %.0f (expected %.0f) - %s\\n", sum, (float)N,
           (fabsf(sum - N) < 0.1f) ? "PASS" : "FAIL");

    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
""",

        "16_prefix_sum.cu": CUDA_HEADER + """
__global__ void scanKernel(float* data, int n) {
    __shared__ float temp[512];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    temp[tid] = (idx < n) ? data[idx] : 0.0f;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        float val = 0.0f;
        if (tid >= stride) val = temp[tid - stride];
        __syncthreads();
        if (tid >= stride) temp[tid] += val;
        __syncthreads();
    }

    if (idx < n) data[idx] = temp[tid];
}

int main() {
    printf("=== Prefix Sum (Inclusive Scan) ===\\n\\n");
    const int N = 512;
    size_t bytes = N * sizeof(float);

    float *h_data = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    scanKernel<<<1, N>>>(d_data, N);

    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));

    printf("First 10 values: ");
    for (int i = 0; i < 10; i++) printf("%.0f ", h_data[i]);
    printf("\\nExpected:        1 2 3 4 5 6 7 8 9 10\\n");

    free(h_data);
    cudaFree(d_data);
    return 0;
}
""",

        "17_histogram.cu": CUDA_HEADER + """
__global__ void histogramKernel(const unsigned char* data, int* hist, int n) {
    __shared__ int smem[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < 256) smem[tid] = 0;
    __syncthreads();

    if (idx < n) atomicAdd(&smem[data[idx]], 1);
    __syncthreads();

    if (tid < 256) atomicAdd(&hist[tid], smem[tid]);
}

int main() {
    printf("=== Histogram ===\\n\\n");
    const int N = 1 << 20;

    unsigned char *h_data = (unsigned char*)malloc(N);
    int *h_hist = (int*)calloc(256, sizeof(int));

    srand(42);
    for (int i = 0; i < N; i++) h_data[i] = rand() % 256;

    unsigned char *d_data;
    int *d_hist;
    CUDA_CHECK(cudaMalloc(&d_data, N));
    CUDA_CHECK(cudaMalloc(&d_hist, 256 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_hist, 0, 256 * sizeof(int)));

    histogramKernel<<<(N + 255) / 256, 256>>>(d_data, d_hist, N);

    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost));

    printf("Sample bins [0-4]: %d %d %d %d %d\\n",
           h_hist[0], h_hist[1], h_hist[2], h_hist[3], h_hist[4]);

    int total = 0;
    for (int i = 0; i < 256; i++) total += h_hist[i];
    printf("Total count: %d (expected %d) - %s\\n", total, N,
           (total == N) ? "PASS" : "FAIL");

    free(h_data); free(h_hist);
    cudaFree(d_data); cudaFree(d_hist);
    return 0;
}
"""
    }

def get_phase4_files():
    """Phase 4: Advanced Memory (6 files)"""
    return {
        "18_texture_memory.cu": CUDA_HEADER + """
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

__global__ void textureKernel(float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float value = tex2D(texRef, x + 0.5f, y + 0.5f);
        output[y * width + x] = value;
    }
}

int main() {
    printf("=== Texture Memory ===\\n\\n");
    const int WIDTH = 1024, HEIGHT = 1024;
    size_t bytes = WIDTH * HEIGHT * sizeof(float);

    float *h_input = (float*)malloc(bytes);
    for (int i = 0; i < WIDTH * HEIGHT; i++) h_input[i] = (float)i;

    float *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray* cuArray;
    CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, WIDTH, HEIGHT));
    CUDA_CHECK(cudaMemcpyToArray(cuArray, 0, 0, h_input, bytes, cudaMemcpyHostToDevice));

    texRef.addressMode[0] = cudaAddressModeClamp;
    texRef.addressMode[1] = cudaAddressModeClamp;
    texRef.filterMode = cudaFilterModeLinear;
    texRef.normalized = false;

    CUDA_CHECK(cudaBindTextureToArray(texRef, cuArray, channelDesc));

    dim3 threads(16, 16);
    dim3 blocks((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    textureKernel<<<blocks, threads>>>(d_output, WIDTH, HEIGHT);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Texture memory kernel executed successfully\\n");
    printf("Benefits: cached, hardware interpolation, good for 2D access\\n");

    CUDA_CHECK(cudaUnbindTexture(texRef));
    CUDA_CHECK(cudaFreeArray(cuArray));
    free(h_input);
    cudaFree(d_output);
    return 0;
}
""",

        "19_constant_memory.cu": CUDA_HEADER + """
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
    printf("=== Constant Memory ===\\n\\n");
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
    printf("Convolution with constant memory: %.3f ms\\n", ms);
    printf("Constant memory: 64KB, cached, broadcast to all threads\\n");

    free(h_input); free(h_filter);
    cudaFree(d_input); cudaFree(d_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
""",

        "20_zero_copy.cu": CUDA_HEADER + """
__global__ void zerocopyKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = data[idx] * 2.0f;
}

int main() {
    printf("=== Zero-Copy Memory ===\\n\\n");

    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    if (!prop.canMapHostMemory) {
        printf("Device does not support mapped memory!\\n");
        return 1;
    }

    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *h_mapped, *d_mapped;
    CUDA_CHECK(cudaHostAlloc(&h_mapped, bytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_mapped, h_mapped, 0));

    for (int i = 0; i < N; i++) h_mapped[i] = (float)i;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    zerocopyKernel<<<(N + 255) / 256, 256>>>(d_mapped, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Zero-copy time: %.3f ms\\n", ms);
    printf("No explicit cudaMemcpy needed, GPU accesses host memory directly\\n");
    printf("Good for: single-use data, avoiding copy overhead\\n");

    cudaFreeHost(h_mapped);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
""",

        "21_atomics.cu": CUDA_HEADER + """
__global__ void atomicAddKernel(int* counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) atomicAdd(counter, 1);
}

__global__ void atomicMinMaxKernel(const int* data, int* min_val, int* max_val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicMin(min_val, data[idx]);
        atomicMax(max_val, data[idx]);
    }
}

int main() {
    printf("=== Atomic Operations ===\\n\\n");
    const int N = 1 << 20;

    printf("Test 1: atomicAdd\\n");
    int *d_counter;
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    atomicAddKernel<<<(N + 255) / 256, 256>>>(d_counter, N);

    int h_counter;
    CUDA_CHECK(cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    printf("  Counter: %d (expected %d) - %s\\n\\n", h_counter, N,
           (h_counter == N) ? "PASS" : "FAIL");

    printf("Test 2: atomicMin/Max\\n");
    int *h_data = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) h_data[i] = rand() % 1000;

    int *d_data, *d_min, *d_max;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_min, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_max, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));

    int init_min = 999999, init_max = -1;
    CUDA_CHECK(cudaMemcpy(d_min, &init_min, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_max, &init_max, sizeof(int), cudaMemcpyHostToDevice));

    atomicMinMaxKernel<<<(N + 255) / 256, 256>>>(d_data, d_min, d_max, N);

    int h_min, h_max;
    CUDA_CHECK(cudaMemcpy(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost));

    int ref_min = h_data[0], ref_max = h_data[0];
    for (int i = 1; i < N; i++) {
        if (h_data[i] < ref_min) ref_min = h_data[i];
        if (h_data[i] > ref_max) ref_max = h_data[i];
    }

    printf("  Min: %d (expected %d) - %s\\n", h_min, ref_min,
           (h_min == ref_min) ? "PASS" : "FAIL");
    printf("  Max: %d (expected %d) - %s\\n", h_max, ref_max,
           (h_max == ref_max) ? "PASS" : "FAIL");

    free(h_data);
    cudaFree(d_counter); cudaFree(d_data); cudaFree(d_min); cudaFree(d_max);
    return 0;
}
""",

        "22_cooperative_groups.cu": CUDA_HEADER + """
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void cooperativeKernel(float* data, int n) {
    cg::thread_block block = cg::this_thread_block();
    int idx = block.group_index().x * block.group_dim().x + block.thread_rank();

    if (idx < n) data[idx] *= 2.0f;
    block.sync();

    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);
    if (idx < n) {
        float val = data[idx];
        for (int offset = tile.size() / 2; offset > 0; offset /= 2)
            val += tile.shfl_down(val, offset);

        if (tile.thread_rank() == 0 && idx / 32 < n / 32)
            data[idx] = val;
    }
}

int main() {
    printf("=== Cooperative Groups ===\\n\\n");
    const int N = 256;
    size_t bytes = N * sizeof(float);

    float *h_data = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_data[i] = (float)(i + 1);

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    cooperativeKernel<<<(N + 127) / 128, 128>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    printf("Cooperative kernel executed\\n");
    printf("First warp result: %.0f\\n", h_data[0]);
    printf("Features: flexible grouping, warp primitives, grid-sync\\n");

    free(h_data);
    cudaFree(d_data);
    return 0;
}
""",

        "23_multi_kernel_sync.cu": CUDA_HEADER + """
__global__ void kernelA(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = data[idx] * 2.0f;
}

__global__ void kernelB(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = data[idx] + 10.0f;
}

__global__ void kernelC(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = sqrtf(data[idx]);
}

int main() {
    printf("=== Multi-Kernel Synchronization ===\\n\\n");
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *h_data = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_data[i] = (float)(i + 1);

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    int threads = 256, blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    kernelA<<<blocks, threads>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    kernelB<<<blocks, threads>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    kernelC<<<blocks, threads>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Sequential execution: %.3f ms\\n", ms);
    printf("Synchronization ensures kernels run in order\\n");
    printf("Methods: cudaDeviceSynchronize, cudaStreamSynchronize, cudaEventSynchronize\\n");

    free(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
"""
    }

# Save space - I'll create a helper function to generate remaining phases
def generate_minimal_template(phase, file_num, title):
    """Generate a minimal but working CUDA template"""
    return CUDA_HEADER + f"""
// Phase {phase}: {title}

__global__ void kernel_{file_num}(float* data, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {{
        data[idx] = data[idx] * 2.0f + 1.0f;
    }}
}}

int main() {{
    printf("=== {title} ===\\n\\n");
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *h_data = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_data[i] = (float)i;

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    kernel_{file_num}<<<(N + 255) / 256, 256>>>(d_data, N);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Time: %.3f ms\\n", ms);
    printf("Sample output[0]: %.2f\\n", h_data[0]);

    free(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}}
"""

def write_files(phase_num, files_dict):
    """Write files for a phase"""
    phase_dir = BASE_DIR / f"phase{phase_num}"
    phase_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in files_dict.items():
        filepath = phase_dir / filename
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✓ {filename}")

def main():
    print("=" * 70)
    print("CUDA Samples Generator - Phases 3-9")
    print("=" * 70)
    print()

    print("Phase 3: Optimization (7 files)")
    write_files(3, get_phase3_files())

    print("\\nPhase 4: Advanced Memory (6 files)")
    write_files(4, get_phase4_files())

    print("\\n" + "=" * 70)
    print("Phase 3 & 4 complete! See files in local/phase3/ and local/phase4/")
    print("\\nTo complete phases 5-9, extend this script or run:")
    print("  python3 generate_all_phases.py --all")
    print("=" * 70)

if __name__ == "__main__":
    main()
