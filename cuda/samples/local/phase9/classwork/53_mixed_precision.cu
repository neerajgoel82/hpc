
/*
 * Mixed Precision Computing - FP16 and FP32 operations
 *
 * Mixed precision uses both FP16 (half precision) and FP32 (single precision)
 * to optimize both performance and accuracy. FP16 offers:
 * - 2x memory bandwidth
 * - 2x throughput on Tensor Cores
 * - Lower power consumption
 *
 * But requires careful handling to maintain accuracy.
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Vector addition in FP32 (baseline)
__global__ void vecAddFP32(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Vector addition in FP16
__global__ void vecAddFP16(__half *a, __half *b, __half *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

// Mixed precision: FP16 input, FP32 accumulation, FP16 output
__global__ void vecAddMixed(__half *a, __half *b, __half *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Convert to FP32 for addition
        float a_fp32 = __half2float(a[idx]);
        float b_fp32 = __half2float(b[idx]);
        float sum = a_fp32 + b_fp32;
        // Convert back to FP16
        c[idx] = __float2half(sum);
    }
}

// Vector dot product in FP32
__global__ void dotProductFP32(float *a, float *b, float *result, int n) {
    __shared__ float shared[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load and multiply
    float temp = 0.0f;
    if (idx < n) {
        temp = a[idx] * b[idx];
    }
    shared[tid] = temp;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        atomicAdd(result, shared[0]);
    }
}

// Vector dot product - Mixed precision (FP16 input, FP32 accumulation)
__global__ void dotProductMixed(__half *a, __half *b, float *result, int n) {
    __shared__ float shared[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load, convert to FP32, and multiply
    float temp = 0.0f;
    if (idx < n) {
        float a_fp32 = __half2float(a[idx]);
        float b_fp32 = __half2float(b[idx]);
        temp = a_fp32 * b_fp32;
    }
    shared[tid] = temp;
    __syncthreads();

    // Reduction in FP32
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        atomicAdd(result, shared[0]);
    }
}

// Matrix multiplication in FP32
__global__ void matMulFP32(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Matrix multiplication - Mixed precision
__global__ void matMulMixed(__half *A, __half *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            float a_val = __half2float(A[row * n + k]);
            float b_val = __half2float(B[k * n + col]);
            sum += a_val * b_val;
        }
        C[row * n + col] = sum;
    }
}

int main() {
    printf("=== Mixed Precision Computing Demo ===\n\n");

    // Check device capabilities
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    // Check FP16 support
    bool hasFP16 = (prop.major >= 6);
    if (!hasFP16) {
        printf("Warning: FP16 operations may be slow on this device (requires Pascal or newer)\n");
    }
    printf("\n");

    // Problem sizes
    const int N = 1024 * 1024;  // 1M elements for vectors
    const int M = 1024;          // 1K x 1K matrix

    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- Test 1: Vector Addition ---
    printf("=== Test 1: Vector Addition ===\n");
    printf("Vector size: %d elements\n\n", N);

    size_t vecBytesFP32 = N * sizeof(float);
    size_t vecBytesFP16 = N * sizeof(__half);

    // Allocate FP32 arrays
    float *d_a_fp32, *d_b_fp32, *d_c_fp32;
    CUDA_CHECK(cudaMalloc(&d_a_fp32, vecBytesFP32));
    CUDA_CHECK(cudaMalloc(&d_b_fp32, vecBytesFP32));
    CUDA_CHECK(cudaMalloc(&d_c_fp32, vecBytesFP32));

    // Allocate FP16 arrays
    __half *d_a_fp16, *d_b_fp16, *d_c_fp16;
    CUDA_CHECK(cudaMalloc(&d_a_fp16, vecBytesFP16));
    CUDA_CHECK(cudaMalloc(&d_b_fp16, vecBytesFP16));
    CUDA_CHECK(cudaMalloc(&d_c_fp16, vecBytesFP16));

    // Initialize with some values
    CUDA_CHECK(cudaMemset(d_a_fp32, 0, vecBytesFP32));
    CUDA_CHECK(cudaMemset(d_b_fp32, 0, vecBytesFP32));
    CUDA_CHECK(cudaMemset(d_a_fp16, 0, vecBytesFP16));
    CUDA_CHECK(cudaMemset(d_b_fp16, 0, vecBytesFP16));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // FP32 addition
    printf("FP32 Vector Addition:\n");
    CUDA_CHECK(cudaEventRecord(start));
    vecAddFP32<<<blocksPerGrid, threadsPerBlock>>>(d_a_fp32, d_b_fp32, d_c_fp32, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float fp32Time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&fp32Time, start, stop));
    printf("  Time: %.3f ms\n", fp32Time);
    printf("  Bandwidth: %.2f GB/s\n", 3.0 * vecBytesFP32 / fp32Time / 1e6);

    // FP16 addition
    printf("\nFP16 Vector Addition:\n");
    CUDA_CHECK(cudaEventRecord(start));
    vecAddFP16<<<blocksPerGrid, threadsPerBlock>>>(d_a_fp16, d_b_fp16, d_c_fp16, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float fp16Time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&fp16Time, start, stop));
    printf("  Time: %.3f ms\n", fp16Time);
    printf("  Bandwidth: %.2f GB/s\n", 3.0 * vecBytesFP16 / fp16Time / 1e6);
    printf("  Speedup: %.2fx\n", fp32Time / fp16Time);
    printf("  Memory saved: %.2f MB (%.1f%%)\n",
           (vecBytesFP32 - vecBytesFP16) / 1e6 * 3, 50.0);

    // Mixed precision addition
    printf("\nMixed Precision Vector Addition:\n");
    CUDA_CHECK(cudaEventRecord(start));
    vecAddMixed<<<blocksPerGrid, threadsPerBlock>>>(d_a_fp16, d_b_fp16, d_c_fp16, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float mixedTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&mixedTime, start, stop));
    printf("  Time: %.3f ms\n", mixedTime);
    printf("  Speedup: %.2fx\n\n", fp32Time / mixedTime);

    // --- Test 2: Dot Product ---
    printf("=== Test 2: Dot Product ===\n");

    float *d_result_fp32;
    CUDA_CHECK(cudaMalloc(&d_result_fp32, sizeof(float)));

    // FP32 dot product
    printf("FP32 Dot Product:\n");
    CUDA_CHECK(cudaMemset(d_result_fp32, 0, sizeof(float)));
    CUDA_CHECK(cudaEventRecord(start));
    dotProductFP32<<<blocksPerGrid, threadsPerBlock>>>(d_a_fp32, d_b_fp32, d_result_fp32, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventElapsedTime(&fp32Time, start, stop));
    printf("  Time: %.3f ms\n", fp32Time);

    // Mixed precision dot product
    printf("\nMixed Precision Dot Product:\n");
    CUDA_CHECK(cudaMemset(d_result_fp32, 0, sizeof(float)));
    CUDA_CHECK(cudaEventRecord(start));
    dotProductMixed<<<blocksPerGrid, threadsPerBlock>>>(d_a_fp16, d_b_fp16, d_result_fp32, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventElapsedTime(&mixedTime, start, stop));
    printf("  Time: %.3f ms\n", mixedTime);
    printf("  Speedup: %.2fx\n\n", fp32Time / mixedTime);

    // --- Test 3: Matrix Multiplication ---
    printf("=== Test 3: Matrix Multiplication ===\n");
    printf("Matrix size: %d x %d\n\n", M, M);

    size_t matBytesFP32 = M * M * sizeof(float);
    size_t matBytesFP16 = M * M * sizeof(__half);

    // Allocate matrices
    float *d_A_fp32, *d_B_fp32, *d_C_fp32;
    CUDA_CHECK(cudaMalloc(&d_A_fp32, matBytesFP32));
    CUDA_CHECK(cudaMalloc(&d_B_fp32, matBytesFP32));
    CUDA_CHECK(cudaMalloc(&d_C_fp32, matBytesFP32));

    __half *d_A_fp16, *d_B_fp16;
    CUDA_CHECK(cudaMalloc(&d_A_fp16, matBytesFP16));
    CUDA_CHECK(cudaMalloc(&d_B_fp16, matBytesFP16));

    // Initialize
    CUDA_CHECK(cudaMemset(d_A_fp32, 0, matBytesFP32));
    CUDA_CHECK(cudaMemset(d_B_fp32, 0, matBytesFP32));
    CUDA_CHECK(cudaMemset(d_A_fp16, 0, matBytesFP16));
    CUDA_CHECK(cudaMemset(d_B_fp16, 0, matBytesFP16));

    dim3 blockDim(16, 16);
    dim3 gridDim((M + 15) / 16, (M + 15) / 16);

    // FP32 matrix multiply
    printf("FP32 Matrix Multiply:\n");
    CUDA_CHECK(cudaEventRecord(start));
    matMulFP32<<<gridDim, blockDim>>>(d_A_fp32, d_B_fp32, d_C_fp32, M);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventElapsedTime(&fp32Time, start, stop));
    float fp32Gflops = 2.0 * M * M * M / fp32Time / 1e6;
    printf("  Time: %.3f ms\n", fp32Time);
    printf("  Performance: %.2f GFLOPS\n", fp32Gflops);

    // Mixed precision matrix multiply
    printf("\nMixed Precision Matrix Multiply:\n");
    CUDA_CHECK(cudaEventRecord(start));
    matMulMixed<<<gridDim, blockDim>>>(d_A_fp16, d_B_fp16, d_C_fp32, M);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventElapsedTime(&mixedTime, start, stop));
    float mixedGflops = 2.0 * M * M * M / mixedTime / 1e6;
    printf("  Time: %.3f ms\n", mixedTime);
    printf("  Performance: %.2f GFLOPS\n", mixedGflops);
    printf("  Speedup: %.2fx\n", fp32Time / mixedTime);
    printf("  Input memory saved: %.2f MB (%.1f%%)\n\n",
           (matBytesFP32 - matBytesFP16) / 1e6 * 2, 50.0);

    // Summary
    printf("=== Summary ===\n");
    printf("Mixed precision benefits:\n");
    printf("  - 2x memory bandwidth (stores half the data)\n");
    printf("  - Faster computation (especially on Tensor Cores)\n");
    printf("  - Lower power consumption\n\n");

    printf("Best practices:\n");
    printf("  - Use FP16 for storage and bandwidth-bound operations\n");
    printf("  - Use FP32 for accumulation to maintain accuracy\n");
    printf("  - Use loss scaling in deep learning to prevent underflow\n");
    printf("  - Profile to find the right balance for your application\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_a_fp32));
    CUDA_CHECK(cudaFree(d_b_fp32));
    CUDA_CHECK(cudaFree(d_c_fp32));
    CUDA_CHECK(cudaFree(d_a_fp16));
    CUDA_CHECK(cudaFree(d_b_fp16));
    CUDA_CHECK(cudaFree(d_c_fp16));
    CUDA_CHECK(cudaFree(d_result_fp32));
    CUDA_CHECK(cudaFree(d_A_fp32));
    CUDA_CHECK(cudaFree(d_B_fp32));
    CUDA_CHECK(cudaFree(d_C_fp32));
    CUDA_CHECK(cudaFree(d_A_fp16));
    CUDA_CHECK(cudaFree(d_B_fp16));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
