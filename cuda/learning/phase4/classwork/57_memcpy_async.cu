/*
 * Asynchronous Memory Copy with Pipeline - Hide transfer latency
 *
 * Modern GPUs support asynchronous memory copy operations that can overlap
 * data transfer with computation. The cuda::pipeline API provides structured
 * async copy primitives for efficient data staging.
 *
 * Benefits:
 * - Hide memory transfer latency
 * - Overlap copy with computation
 * - Efficient use of copy engines
 * - Better performance for streaming workloads
 *
 * Requires: Compute capability 8.0+ for optimal performance
 * Compile: nvcc -arch=sm_80 -O2 57_memcpy_async.cu -o memcpy_async
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda/pipeline>
#define ASYNC_COPY_AVAILABLE 1
#else
#define ASYNC_COPY_AVAILABLE 0
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

// Traditional synchronous kernel - copy then compute
__global__ void traditionalKernel(const float *input, float *output, int n) {
    __shared__ float shared[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Synchronous copy to shared memory
    if (idx < n) {
        shared[tid] = input[idx];
    }
    __syncthreads();

    // Compute on data in shared memory
    if (idx < n) {
        float val = shared[tid];
        val = val * val + 2.0f * val + 1.0f;
        output[idx] = val;
    }
}

#if ASYNC_COPY_AVAILABLE
// Modern async copy kernel - overlap copy and compute
__global__ void asyncCopyKernel(const float *input, float *output, int n) {
    __shared__ float shared[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Create pipeline for async operations
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        2  // 2 stage pipeline
    > pipe_state;

    auto block = cooperative_groups::this_thread_block();
    auto pipe = cuda::make_pipeline(block, &pipe_state);

    // Async copy from global to shared memory
    if (idx < n) {
        cuda::memcpy_async(block, &shared[tid], &input[idx], sizeof(float), pipe);
    }

    // Commit the async copy
    pipe.producer_commit();

    // Can do other work here while copy is in flight

    // Wait for async copy to complete
    pipe.consumer_wait();

    // Now compute on data (copy is done)
    if (idx < n) {
        float val = shared[tid];
        val = val * val + 2.0f * val + 1.0f;
        output[idx] = val;
    }

    pipe.consumer_release();
}

// Multi-stage pipeline kernel - process data in chunks
__global__ void pipelineKernel(const float *input, float *output, int n) {
    constexpr int CHUNK_SIZE = 256;
    constexpr int STAGES = 2;

    __shared__ float shared[STAGES][CHUNK_SIZE];

    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        STAGES
    > pipe_state;

    auto block = cooperative_groups::this_thread_block();
    auto pipe = cuda::make_pipeline(block, &pipe_state);

    int base_idx = blockIdx.x * CHUNK_SIZE;
    int tid = threadIdx.x;

    // Prime the pipeline - load first chunk
    for (int stage = 0; stage < STAGES && (base_idx + stage * CHUNK_SIZE) < n; stage++) {
        int chunk_idx = base_idx + stage * CHUNK_SIZE + tid;
        if (chunk_idx < n && tid < CHUNK_SIZE) {
            cuda::memcpy_async(block, &shared[stage][tid],
                             &input[chunk_idx], sizeof(float), pipe);
        }
        pipe.producer_commit();
    }

    // Main processing loop - overlap copy of next chunk with compute of current
    for (int stage = 0; stage < STAGES; stage++) {
        pipe.consumer_wait();
        block.sync();

        // Process current chunk
        int chunk_idx = base_idx + stage * CHUNK_SIZE + tid;
        if (chunk_idx < n && tid < CHUNK_SIZE) {
            float val = shared[stage % STAGES][tid];
            val = val * val + 2.0f * val + 1.0f;
            output[chunk_idx] = val;
        }

        pipe.consumer_release();
    }
}
#endif // ASYNC_COPY_AVAILABLE

int main() {
    printf("=== Asynchronous Memory Copy Demo ===\n");

    // Check device capability
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("\nDevice: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    if (prop.major < 8) {
        printf("Warning: Optimal async copy requires compute capability 8.0+ (Ampere)\n");
        printf("This device is %d.%d - will use basic async operations\n",
               prop.major, prop.minor);
    }

    const int n = 1024 * 1024;  // 1M elements
    const size_t bytes = n * sizeof(float);

    // Allocate host memory
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);

    // Initialize input
    for (int i = 0; i < n; i++) {
        h_input[i] = (float)i / 1000.0f;
    }

    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Test traditional synchronous kernel
    printf("\n=== Traditional Synchronous Copy ===\n");
    CUDA_CHECK(cudaEventRecord(start));
    traditionalKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Time: %.3f ms\n", ms);

#if ASYNC_COPY_AVAILABLE
    // Test async copy kernel (if available)
    if (prop.major >= 8) {
        printf("\n=== Async Copy with Pipeline ===\n");
        CUDA_CHECK(cudaEventRecord(start));
        asyncCopyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("Time: %.3f ms\n", ms);

        // Test pipelined kernel
        printf("\n=== Multi-Stage Pipeline ===\n");
        CUDA_CHECK(cudaEventRecord(start));
        pipelineKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("Time: %.3f ms\n", ms);
    } else {
        printf("\nAsync copy kernel requires sm_80+ (skipped)\n");
    }
#else
    printf("\nAsync copy API not available at compile time\n");
    printf("Compile with -arch=sm_80 or higher to enable\n");
#endif

    // Verify results
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int i = 0; i < 100 && correct; i++) {
        float expected = h_input[i];
        expected = expected * expected + 2.0f * expected + 1.0f;
        if (fabsf(h_output[i] - expected) > 1e-5) {
            printf("Error at index %d: expected %.3f, got %.3f\n",
                   i, expected, h_output[i]);
            correct = false;
        }
    }

    if (correct) {
        printf("\n✓ Results verified successfully\n");
    }

    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n=== Key Concepts ===\n");
    printf("1. cuda::memcpy_async enables asynchronous global->shared transfers\n");
    printf("2. cuda::pipeline provides structured async copy management\n");
    printf("3. Multi-stage pipelines overlap copy and compute\n");
    printf("4. producer_commit/consumer_wait manage pipeline stages\n");
    printf("5. Ampere+ GPUs have hardware acceleration for async copy\n");

    return 0;
}
