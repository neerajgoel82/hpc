
/*
 * Tensor Cores Demo - Hardware-accelerated matrix operations
 *
 * Tensor Cores are specialized hardware units for matrix multiply-accumulate
 * operations. They provide massive speedup for deep learning workloads.
 *
 * Available on:
 * - Volta (V100): FP16 input, FP32 accumulation
 * - Turing (RTX 20xx): FP16, INT8, INT4
 * - Ampere (A100, RTX 30xx): FP16, BF16, TF32, INT8, INT4, INT1
 * - Hopper (H100): FP8, FP16, BF16, TF32, INT8
 *
 * This sample demonstrates the WMMA (Warp Matrix Multiply-Accumulate) API.
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Check if WMMA is available (requires sm_70 or higher)
#if __CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__)
#include <mma.h>
#define WMMA_AVAILABLE 1
#else
#define WMMA_AVAILABLE 0
#endif

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Simple matrix multiply without Tensor Cores (FP32)
__global__ void matmulFP32(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Matrix multiply with FP16 (no Tensor Cores)
__global__ void matmulFP16(__half *A, __half *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            float a = __half2float(A[row * K + i]);
            float b = __half2float(B[i * N + col]);
            sum += a * b;
        }
        C[row * N + col] = sum;
    }
}

#if WMMA_AVAILABLE
using namespace nvcuda::wmma;

// Matrix multiply using WMMA (Tensor Cores)
// WMMA works on 16x16x16 matrix fragments
__global__ void matmulWMMA(__half *A, __half *B, float *C, int M, int N, int K) {
    // Warp and lane identification
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x);

    // Declare the fragments
    fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> acc_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    // Initialize the output to zero
    fill_fragment(acc_frag, 0.0f);

    // Loop over K dimension in chunks of 16
    for (int k = 0; k < K; k += 16) {
        int aRow = warpM * 16;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * 16;

        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load the inputs
            load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            load_matrix_sync(b_frag, B + bRow * N + bCol, N);

            // Perform the matrix multiply-accumulate
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Store the output
    int cRow = warpM * 16;
    int cCol = warpN * 16;

    if (cRow < M && cCol < N) {
        store_matrix_sync(C + cRow * N + cCol, acc_frag, N, mem_row_major);
    }
}
#endif

void printTensorCoreInfo() {
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("=== Tensor Core Information ===\n\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

    if (prop.major >= 7) {
        printf("Tensor Cores: AVAILABLE\n\n");

        printf("Supported operations by architecture:\n");
        if (prop.major == 7 && prop.minor == 0) {
            printf("  Volta (sm_70):\n");
            printf("    - FP16 input, FP32 accumulation\n");
            printf("    - D = A * B + C (16x16x16 tiles)\n");
        } else if (prop.major == 7 && prop.minor == 5) {
            printf("  Turing (sm_75):\n");
            printf("    - FP16, INT8, INT4, INT1\n");
            printf("    - 16x16x16 and 8x8x32 tiles\n");
        } else if (prop.major == 8 && prop.minor == 0) {
            printf("  Ampere (sm_80):\n");
            printf("    - FP64, TF32, BF16, FP16, INT8, INT4, INT1\n");
            printf("    - Multiple tile sizes\n");
            printf("    - Sparsity support\n");
        } else if (prop.major == 8 && prop.minor == 6) {
            printf("  Ampere (sm_86) - Gaming:\n");
            printf("    - TF32, BF16, FP16, INT8, INT4, INT1\n");
        } else if (prop.major == 9 && prop.minor == 0) {
            printf("  Hopper (sm_90):\n");
            printf("    - FP8, FP64, TF32, BF16, FP16, INT8\n");
            printf("    - Thread block clusters\n");
            printf("    - Tensor memory accelerator\n");
        }

        printf("\nPerformance characteristics:\n");
        printf("  - Up to 8x faster than CUDA cores for FP16\n");
        printf("  - Up to 16x faster for INT8\n");
        printf("  - Optimized for matrix sizes that are multiples of 16\n");
        printf("  - Best with mixed precision (FP16 input, FP32 accumulation)\n");
    } else {
        printf("Tensor Cores: NOT AVAILABLE\n");
        printf("  Requires compute capability 7.0 or higher (Volta+)\n");
    }
    printf("\n");
}

int main() {
    printf("=== Tensor Cores Demo ===\n\n");

    printTensorCoreInfo();

    // Check device capabilities
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

#if !WMMA_AVAILABLE
    printf("Warning: This code was not compiled with Tensor Core support\n");
    printf("Recompile with: nvcc -arch=sm_70 or higher\n\n");
#endif

    if (prop.major < 7) {
        printf("This GPU does not support Tensor Cores.\n");
        printf("Running comparison with standard CUDA cores only.\n\n");
    }

    // Matrix dimensions (must be multiples of 16 for WMMA)
    const int M = 1024;  // Rows of A and C
    const int K = 1024;  // Cols of A, Rows of B
    const int N = 1024;  // Cols of B and C

    printf("Matrix dimensions: M=%d, K=%d, N=%d\n", M, K, N);
    printf("Total operations: %.2f GFLOP\n\n", 2.0 * M * N * K / 1e9);

    // Allocate memory
    size_t bytesA_fp32 = M * K * sizeof(float);
    size_t bytesB_fp32 = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);
    size_t bytesA_fp16 = M * K * sizeof(__half);
    size_t bytesB_fp16 = K * N * sizeof(__half);

    float *d_A_fp32, *d_B_fp32, *d_C_fp32;
    __half *d_A_fp16, *d_B_fp16;
    float *d_C_wmma;

    CUDA_CHECK(cudaMalloc(&d_A_fp32, bytesA_fp32));
    CUDA_CHECK(cudaMalloc(&d_B_fp32, bytesB_fp32));
    CUDA_CHECK(cudaMalloc(&d_C_fp32, bytesC));
    CUDA_CHECK(cudaMalloc(&d_A_fp16, bytesA_fp16));
    CUDA_CHECK(cudaMalloc(&d_B_fp16, bytesB_fp16));
    CUDA_CHECK(cudaMalloc(&d_C_wmma, bytesC));

    // Initialize matrices
    CUDA_CHECK(cudaMemset(d_A_fp32, 0, bytesA_fp32));
    CUDA_CHECK(cudaMemset(d_B_fp32, 0, bytesB_fp32));
    CUDA_CHECK(cudaMemset(d_A_fp16, 0, bytesA_fp16));
    CUDA_CHECK(cudaMemset(d_B_fp16, 0, bytesB_fp16));

    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- Test 1: FP32 Standard CUDA Cores ---
    printf("Test 1: FP32 (Standard CUDA Cores)\n");

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);

    CUDA_CHECK(cudaEventRecord(start));
    matmulFP32<<<gridDim, blockDim>>>(d_A_fp32, d_B_fp32, d_C_fp32, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float fp32Time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&fp32Time, start, stop));
    float fp32Gflops = 2.0 * M * N * K / fp32Time / 1e6;

    printf("  Time: %.3f ms\n", fp32Time);
    printf("  Performance: %.2f GFLOPS\n\n", fp32Gflops);

    // --- Test 2: FP16 without Tensor Cores ---
    printf("Test 2: FP16 (CUDA Cores, no Tensor Cores)\n");

    CUDA_CHECK(cudaEventRecord(start));
    matmulFP16<<<gridDim, blockDim>>>(d_A_fp16, d_B_fp16, d_C_wmma, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float fp16Time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&fp16Time, start, stop));
    float fp16Gflops = 2.0 * M * N * K / fp16Time / 1e6;

    printf("  Time: %.3f ms\n", fp16Time);
    printf("  Performance: %.2f GFLOPS\n", fp16Gflops);
    printf("  Speedup vs FP32: %.2fx\n\n", fp32Time / fp16Time);

#if WMMA_AVAILABLE
    // --- Test 3: Tensor Cores with WMMA ---
    if (prop.major >= 7) {
        printf("Test 3: FP16 with Tensor Cores (WMMA)\n");

        // For WMMA, we need different grid dimensions
        // Each warp processes a 16x16 output tile
        dim3 wmmaBlockDim(32, 4);  // 128 threads per block
        dim3 wmmaGridDim((N + 15) / 16, (M + 15) / 16);

        CUDA_CHECK(cudaEventRecord(start));
        matmulWMMA<<<wmmaGridDim, wmmaBlockDim>>>(d_A_fp16, d_B_fp16, d_C_wmma, M, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaDeviceSynchronize());

        float wmmaTime = 0;
        CUDA_CHECK(cudaEventElapsedTime(&wmmaTime, start, stop));
        float wmmaGflops = 2.0 * M * N * K / wmmaTime / 1e6;

        printf("  Time: %.3f ms\n", wmmaTime);
        printf("  Performance: %.2f GFLOPS\n", wmmaGflops);
        printf("  Speedup vs FP32: %.2fx\n", fp32Time / wmmaTime);
        printf("  Speedup vs FP16: %.2fx\n\n", fp16Time / wmmaTime);

        // Summary
        printf("=== Performance Summary ===\n");
        printf("FP32 (CUDA Cores):    %6.2f GFLOPS (%.2fx)\n", fp32Gflops, 1.0f);
        printf("FP16 (CUDA Cores):    %6.2f GFLOPS (%.2fx)\n", fp16Gflops, fp32Gflops / fp16Gflops);
        printf("FP16 (Tensor Cores):  %6.2f GFLOPS (%.2fx)\n\n", wmmaGflops, fp32Gflops / wmmaGflops);
    }
#else
    printf("Test 3: Tensor Cores NOT AVAILABLE\n");
    printf("  Compile with -arch=sm_70 or higher to enable\n\n");
#endif

    printf("=== Notes ===\n");
    printf("Tensor Cores are best for:\n");
    printf("  - Deep learning training and inference\n");
    printf("  - Large matrix multiplications\n");
    printf("  - Mixed precision workloads\n");
    printf("  - Batch processing\n\n");

    printf("Optimization tips:\n");
    printf("  - Use dimensions that are multiples of 16\n");
    printf("  - Keep matrices in FP16 for bandwidth\n");
    printf("  - Accumulate in FP32 for accuracy\n");
    printf("  - Use cuBLAS or cuDNN when possible (highly optimized)\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_A_fp32));
    CUDA_CHECK(cudaFree(d_B_fp32));
    CUDA_CHECK(cudaFree(d_C_fp32));
    CUDA_CHECK(cudaFree(d_A_fp16));
    CUDA_CHECK(cudaFree(d_B_fp16));
    CUDA_CHECK(cudaFree(d_C_wmma));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
