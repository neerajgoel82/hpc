/*
 * Async Barriers - Efficient synchronization with asynchronous operations
 *
 * cuda::barrier provides fine-grained synchronization for async operations,
 * particularly useful with memcpy_async for overlapping data movement and
 * computation. This is essential for maximizing Ampere+ GPU performance.
 *
 * Key Features:
 * - Asynchronous arrive/wait semantics
 * - Integration with cp.async hardware instructions
 * - Producer-consumer patterns for pipelining
 * - Lower overhead than __syncthreads for async operations
 *
 * Requires: Compute capability 8.0+ (Ampere or newer)
 * Compile: nvcc -arch=sm_80 -O2 59_async_barriers.cu -o async_barriers
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda/barrier>
#define BARRIER_AVAILABLE 1
#else
#define BARRIER_AVAILABLE 0
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

#if BARRIER_AVAILABLE
// Producer-consumer pattern with async barriers
// Producers load data, consumers process it
__global__ void producerConsumerKernel(const int *input, int *output, int n) {
    constexpr int STAGES = 2;
    constexpr int CHUNK_SIZE = 256;

    __shared__ int buffer[STAGES][CHUNK_SIZE];

    // Barrier for each pipeline stage
    __shared__ cuda::barrier<cuda::thread_scope_block> barriers[STAGES];

    auto block = cooperative_groups::this_thread_block();
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Initialize barriers
    if (tid < STAGES) {
        init(&barriers[tid], block.size());
    }
    block.sync();

    // Producer threads (first half) load data asynchronously
    // Consumer threads (second half) process data
    bool is_producer = (tid < block.size() / 2);

    if (is_producer) {
        // Producer: Load data using async copy
        for (int stage = 0; stage < STAGES; stage++) {
            int global_idx = bid * CHUNK_SIZE + tid;
            if (global_idx < n && tid < CHUNK_SIZE) {
                // Async copy from global to shared
                cuda::memcpy_async(block, &buffer[stage][tid],
                                 &input[global_idx], sizeof(int), barriers[stage]);
            }

            // Arrive at barrier (signal data ready)
            barriers[stage].arrive();
        }
    } else {
        // Consumer: Wait for data and process
        int consumer_tid = tid - block.size() / 2;

        for (int stage = 0; stage < STAGES; stage++) {
            // Wait for producers to finish loading this stage
            barriers[stage].wait();

            // Process data
            int global_idx = bid * CHUNK_SIZE + consumer_tid;
            if (global_idx < n && consumer_tid < CHUNK_SIZE) {
                int val = buffer[stage][consumer_tid];
                val = val * 2 + 1;
                output[global_idx] = val;
            }
        }
    }
}

// Double-buffering with async barriers for continuous pipeline
__global__ void doubleBufferKernel(const float *input, float *output, int n) {
    constexpr int CHUNK_SIZE = 256;

    __shared__ float buffer[2][CHUNK_SIZE];
    __shared__ cuda::barrier<cuda::thread_scope_block> barriers[2];

    auto block = cooperative_groups::this_thread_block();
    int tid = threadIdx.x;
    int chunks_per_block = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Initialize barriers
    if (tid < 2) {
        init(&barriers[tid], block.size());
    }
    block.sync();

    // Process multiple chunks with double buffering
    for (int chunk = 0; chunk < chunks_per_block; chunk++) {
        int buffer_idx = chunk % 2;
        int global_idx = chunk * CHUNK_SIZE + tid;

        // Load current chunk asynchronously
        if (global_idx < n && tid < CHUNK_SIZE) {
            cuda::memcpy_async(block, &buffer[buffer_idx][tid],
                             &input[global_idx], sizeof(float),
                             barriers[buffer_idx]);
        }
        barriers[buffer_idx].arrive();

        // If not first chunk, process previous chunk while current loads
        if (chunk > 0) {
            int prev_buffer_idx = (chunk - 1) % 2;
            int prev_global_idx = (chunk - 1) * CHUNK_SIZE + tid;

            // Wait for previous chunk to be ready
            barriers[prev_buffer_idx].wait();

            // Process previous chunk
            if (prev_global_idx < n && tid < CHUNK_SIZE) {
                float val = buffer[prev_buffer_idx][tid];
                val = sqrtf(val * val + 1.0f);
                output[prev_global_idx] = val;
            }
        }
    }

    // Process final chunk
    int final_chunk = chunks_per_block - 1;
    int final_buffer_idx = final_chunk % 2;
    int final_global_idx = final_chunk * CHUNK_SIZE + tid;

    barriers[final_buffer_idx].wait();
    if (final_global_idx < n && tid < CHUNK_SIZE) {
        float val = buffer[final_buffer_idx][tid];
        val = sqrtf(val * val + 1.0f);
        output[final_global_idx] = val;
    }
}

// Warp-level async operations with barriers
__global__ void warpAsyncKernel(const int *input, int *output, int n) {
    constexpr int WARP_SIZE = 32;

    __shared__ int buffer[256];
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;

    auto block = cooperative_groups::this_thread_block();
    auto warp = cooperative_groups::tiled_partition<WARP_SIZE>(block);

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Initialize barrier
    if (tid == 0) {
        init(&barrier, block.size());
    }
    block.sync();

    // Each warp loads its portion asynchronously
    int global_idx = blockIdx.x * blockDim.x + tid;

    if (global_idx < n) {
        // Warp-level async copy
        cuda::memcpy_async(warp, &buffer[tid], &input[global_idx],
                         sizeof(int), barrier);
    }

    // All threads arrive at barrier
    barrier.arrive();

    // Wait for all data to be ready
    barrier.wait();

    // Process data
    if (global_idx < n) {
        int val = buffer[tid];
        // Simple computation
        val = (val << 1) | (val & 1);
        output[global_idx] = val;
    }
}
#endif // BARRIER_AVAILABLE

// Fallback kernel for older architectures
__global__ void fallbackKernel(const int *input, int *output, int n) {
    __shared__ int buffer[256];

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;

    // Traditional sync copy
    if (global_idx < n) {
        buffer[tid] = input[global_idx];
    }
    __syncthreads();

    // Process
    if (global_idx < n) {
        output[global_idx] = buffer[tid] * 2 + 1;
    }
}

int main() {
    printf("=== Async Barriers Demo ===\n");

    // Check device capability
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("\nDevice: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    if (prop.major < 8) {
        printf("\nError: Async barriers require compute capability 8.0+ (Ampere)\n");
        printf("This device is %d.%d\n", prop.major, prop.minor);
        printf("Running fallback kernel instead...\n\n");
    }

    const int n = 1024 * 256;  // 256K elements
    const size_t bytes = n * sizeof(int);

    // Allocate memory
    int *h_input = (int*)malloc(bytes);
    int *h_output = (int*)malloc(bytes);

    for (int i = 0; i < n; i++) {
        h_input[i] = i;
    }

    int *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

#if BARRIER_AVAILABLE
    if (prop.major >= 8) {
        // Test producer-consumer pattern
        printf("=== Producer-Consumer with Barriers ===\n");
        CUDA_CHECK(cudaEventRecord(start));
        producerConsumerKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("Time: %.3f ms\n", ms);

        // Test warp-level async operations
        printf("\n=== Warp-Level Async Operations ===\n");
        CUDA_CHECK(cudaEventRecord(start));
        warpAsyncKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("Time: %.3f ms\n", ms);

        // Verify results
        CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

        bool correct = true;
        for (int i = 0; i < 100 && correct; i++) {
            int expected = (h_input[i] << 1) | (h_input[i] & 1);
            if (h_output[i] != expected) {
                printf("Error at %d: expected %d, got %d\n", i, expected, h_output[i]);
                correct = false;
            }
        }

        if (correct) {
            printf("\n✓ Results verified successfully\n");
        }
    } else
#endif
    {
        // Run fallback for older GPUs
        printf("=== Fallback Kernel (Traditional __syncthreads) ===\n");
        CUDA_CHECK(cudaEventRecord(start));
        fallbackKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("Time: %.3f ms\n", ms);
    }

    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n=== Key Concepts ===\n");
    printf("1. cuda::barrier provides async synchronization primitives\n");
    printf("2. arrive/wait semantics enable producer-consumer patterns\n");
    printf("3. Integration with memcpy_async for zero-overhead data staging\n");
    printf("4. Double-buffering hides memory latency with computation\n");
    printf("5. Hardware acceleration on Ampere+ (cp.async instructions)\n");
    printf("6. Lower overhead than __syncthreads for async workloads\n");

    return 0;
}
