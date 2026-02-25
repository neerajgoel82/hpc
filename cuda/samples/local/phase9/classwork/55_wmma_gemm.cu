
/*
 * WMMA GEMM - Warp Matrix Multiply-Accumulate for Tensor Cores
 *
 * This implements a complete GEMM (General Matrix Multiply) using the
 * WMMA API to leverage Tensor Cores. This is the building block for
 * deep learning operations.
 *
 * Operation: C = alpha * A * B + beta * C
 *
 * WMMA operates on small matrix fragments (typically 16x16x16):
 * - matrix_a: 16x16 input matrix A
 * - matrix_b: 16x16 input matrix B
 * - accumulator: 16x16 accumulator/output matrix C
 *
 * Compile: nvcc -arch=sm_70 55_wmma_gemm.cu -o wmma_gemm
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

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

// WMMA tile sizes
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

#if WMMA_AVAILABLE
using namespace nvcuda::wmma;

/*
 * WMMA-based GEMM kernel
 *
 * Each warp computes a WMMA_M x WMMA_N tile of the output matrix.
 * The warp loads tiles from A and B, performs matrix multiply-accumulate,
 * and stores the result to C.
 *
 * Grid and block dimensions:
 * - blockDim: (WARP_SIZE, 4) = 128 threads, 4 warps per block
 * - gridDim: enough blocks to cover the output matrix
 */
__global__ void wmmaGemm(
    const __half *A,    // M x K matrix
    const __half *B,    // K x N matrix
    float *C,           // M x N matrix
    int M, int N, int K,
    float alpha, float beta)
{
    // Warp and lane identification
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator to zero
    fill_fragment(acc_frag, 0.0f);

    // Calculate base positions for this warp's output tile
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    // Bounds checking
    if (cRow >= M || cCol >= N) return;

    // Loop over K dimension in WMMA_K chunks
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = cRow;
        int aCol = k;
        int bRow = k;
        int bCol = cCol;

        // Check bounds for A and B
        if (aCol + WMMA_K <= K && bRow + WMMA_K <= K) {
            // Load matrix fragments from A and B
            load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            load_matrix_sync(b_frag, B + bRow * N + bCol, N);

            // Perform matrix multiply-accumulate: acc = a * b + acc
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Load existing C values if beta != 0
    if (beta != 0.0f) {
        load_matrix_sync(c_frag, C + cRow * N + cCol, N, mem_row_major);

        // Scale C by beta and add to accumulator
        for (int i = 0; i < c_frag.num_elements; i++) {
            acc_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }
    } else {
        // Just scale by alpha
        for (int i = 0; i < acc_frag.num_elements; i++) {
            acc_frag.x[i] = alpha * acc_frag.x[i];
        }
    }

    // Store the result
    store_matrix_sync(C + cRow * N + cCol, acc_frag, N, mem_row_major);
}

/*
 * Optimized WMMA GEMM with tiling for better performance
 *
 * Each warp processes multiple WMMA tiles to improve data reuse
 * and reduce memory traffic.
 */
__global__ void wmmaGemmTiled(
    const __half *A,
    const __half *B,
    float *C,
    int M, int N, int K,
    float alpha, float beta)
{
    // Each block processes a larger tile
    const int BLOCK_ROW_TILES = 4;
    const int BLOCK_COL_TILES = 4;

    // Warp identification within block
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    int warpRow = warpId / BLOCK_COL_TILES;
    int warpCol = warpId % BLOCK_COL_TILES;

    // Block position in output matrix
    int blockRowOffset = blockIdx.x * WMMA_M * BLOCK_ROW_TILES;
    int blockColOffset = blockIdx.y * WMMA_N * BLOCK_COL_TILES;

    // This warp's position
    int warpRowOffset = blockRowOffset + warpRow * WMMA_M;
    int warpColOffset = blockColOffset + warpCol * WMMA_N;

    // Bounds checking
    if (warpRowOffset >= M || warpColOffset >= N) return;

    // Declare fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // Initialize accumulator
    fill_fragment(acc_frag, 0.0f);

    // Main loop over K dimension
    for (int k = 0; k < K; k += WMMA_K) {
        if (k + WMMA_K <= K) {
            // Load tiles
            load_matrix_sync(a_frag, A + warpRowOffset * K + k, K);
            load_matrix_sync(b_frag, B + k * N + warpColOffset, N);

            // Multiply-accumulate
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Apply alpha and beta
    if (beta != 0.0f) {
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        load_matrix_sync(c_frag, C + warpRowOffset * N + warpColOffset, N, mem_row_major);

        for (int i = 0; i < acc_frag.num_elements; i++) {
            acc_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }
    } else {
        for (int i = 0; i < acc_frag.num_elements; i++) {
            acc_frag.x[i] = alpha * acc_frag.x[i];
        }
    }

    // Store result
    store_matrix_sync(C + warpRowOffset * N + warpColOffset, acc_frag, N, mem_row_major);
}
#endif

// Reference CPU implementation for verification
void cpuGemm(const float *A, const float *B, float *C,
             int M, int N, int K, float alpha, float beta) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

int main() {
    printf("=== WMMA GEMM Demo ===\n\n");

    // Check device support
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

    if (prop.major < 7) {
        printf("Error: Tensor Cores require compute capability 7.0 or higher\n");
        return 1;
    }

#if !WMMA_AVAILABLE
    printf("Error: Code not compiled with WMMA support\n");
    printf("Recompile with: nvcc -arch=sm_70 or higher\n");
    return 1;
#else

    // Matrix dimensions (must be multiples of 16)
    const int M = 512;
    const int N = 512;
    const int K = 512;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Operation: C = %.1f * A * B + %.1f * C\n", alpha, beta);
    printf("WMMA tile size: %dx%dx%d\n", WMMA_M, WMMA_N, WMMA_K);
    printf("Total FLOPs: %.2f GFLOP\n\n", 2.0 * M * N * K / 1e9);

    // Allocate host memory
    size_t bytesA_fp32 = M * K * sizeof(float);
    size_t bytesB_fp32 = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);
    size_t bytesA_fp16 = M * K * sizeof(__half);
    size_t bytesB_fp16 = K * N * sizeof(__half);

    float *h_A = (float*)malloc(bytesA_fp32);
    float *h_B = (float*)malloc(bytesB_fp32);
    float *h_C = (float*)malloc(bytesC);
    float *h_C_ref = (float*)malloc(bytesC);

    // Initialize with random values
    for (int i = 0; i < M * K; i++) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = (float)(rand() % 100) / 100.0f;
    }
    for (int i = 0; i < M * N; i++) {
        h_C[i] = 0.0f;
        h_C_ref[i] = 0.0f;
    }

    // Convert to FP16
    __half *h_A_fp16 = (__half*)malloc(bytesA_fp16);
    __half *h_B_fp16 = (__half*)malloc(bytesB_fp16);

    for (int i = 0; i < M * K; i++) {
        h_A_fp16[i] = __float2half(h_A[i]);
    }
    for (int i = 0; i < K * N; i++) {
        h_B_fp16[i] = __float2half(h_B[i]);
    }

    // Allocate device memory
    __half *d_A, *d_B;
    float *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, bytesA_fp16));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB_fp16));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A_fp16, bytesA_fp16, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_fp16, bytesB_fp16, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, bytesC, cudaMemcpyHostToDevice));

    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- Test 1: Basic WMMA GEMM ---
    printf("Test 1: Basic WMMA GEMM\n");

    dim3 blockDim(32, 4);  // 128 threads per block (4 warps)
    dim3 gridDim((M + WMMA_M - 1) / WMMA_M, (N + WMMA_N - 1) / WMMA_N);

    printf("Grid: (%d, %d), Block: (%d, %d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // Warm-up
    wmmaGemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    CUDA_CHECK(cudaEventRecord(start));
    wmmaGemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float basicTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&basicTime, start, stop));
    float basicGflops = 2.0 * M * N * K / basicTime / 1e6;

    printf("Time: %.3f ms\n", basicTime);
    printf("Performance: %.2f GFLOPS\n\n", basicGflops);

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));

    // --- Test 2: Tiled WMMA GEMM ---
    printf("Test 2: Tiled WMMA GEMM\n");

    CUDA_CHECK(cudaMemset(d_C, 0, bytesC));

    const int BLOCK_TILES = 4;
    dim3 tiledBlockDim(128);
    dim3 tiledGridDim((M + WMMA_M * BLOCK_TILES - 1) / (WMMA_M * BLOCK_TILES),
                      (N + WMMA_N * BLOCK_TILES - 1) / (WMMA_N * BLOCK_TILES));

    printf("Grid: (%d, %d), Block: %d\n", tiledGridDim.x, tiledGridDim.y, tiledBlockDim.x);

    // Warm-up
    wmmaGemmTiled<<<tiledGridDim, tiledBlockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    CUDA_CHECK(cudaEventRecord(start));
    wmmaGemmTiled<<<tiledGridDim, tiledBlockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float tiledTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&tiledTime, start, stop));
    float tiledGflops = 2.0 * M * N * K / tiledTime / 1e6;

    printf("Time: %.3f ms\n", tiledTime);
    printf("Performance: %.2f GFLOPS\n", tiledGflops);
    printf("Speedup vs basic: %.2fx\n\n", basicTime / tiledTime);

    // Verify results
    printf("Verifying results...\n");
    cpuGemm(h_A, h_B, h_C_ref, M, N, K, alpha, beta);

    float maxError = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float error = fabsf(h_C[i] - h_C_ref[i]);
        maxError = fmaxf(maxError, error);
    }

    printf("Max error: %f\n", maxError);
    if (maxError < 1e-2) {
        printf("Verification: PASSED\n\n");
    } else {
        printf("Verification: FAILED\n\n");
    }

    // Summary
    printf("=== Performance Summary ===\n");
    printf("Basic WMMA: %.2f GFLOPS\n", basicGflops);
    printf("Tiled WMMA: %.2f GFLOPS\n\n", tiledGflops);

    printf("=== Key Takeaways ===\n");
    printf("1. WMMA API provides direct access to Tensor Cores\n");
    printf("2. Operations are performed on 16x16x16 matrix fragments\n");
    printf("3. Mixed precision (FP16 input, FP32 accumulation) is standard\n");
    printf("4. Tiling improves performance through better data reuse\n");
    printf("5. For production, use cuBLAS which is highly optimized\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    free(h_A_fp16);
    free(h_B_fp16);

    return 0;
#endif
}
