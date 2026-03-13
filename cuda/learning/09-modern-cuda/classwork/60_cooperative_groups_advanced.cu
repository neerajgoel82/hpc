/*
 * Advanced Cooperative Groups - Grid-wide and cluster synchronization
 *
 * This demonstrates advanced cooperative groups features including:
 * - Grid-wide synchronization across all blocks
 * - Multi-device grid groups
 * - Thread block clusters (Hopper sm_90+)
 * - Advanced group partitioning and collectives
 *
 * Cooperative Groups provide flexible, explicit synchronization primitives
 * that are more powerful than __syncthreads().
 *
 * Requires: Compute capability 7.0+ (6.0+ for basic features)
 * Compile: nvcc -arch=sm_70 -O2 -rdc=true 60_cooperative_groups_advanced.cu -o cg_advanced -lcudadevrt
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Grid-wide synchronization kernel
// All blocks must cooperate - requires cooperative launch
__global__ void gridSyncKernel(int *data, int n, int *counter) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Phase 1: Each thread increments its element
    if (idx < n) {
        data[idx] += 1;
    }

    // Grid-wide synchronization - all blocks wait here
    grid.sync();

    // Phase 2: Now we can safely use data written by other blocks
    // Each block atomically increments global counter
    if (threadIdx.x == 0) {
        atomicAdd(counter, 1);
    }

    grid.sync();

    // Phase 3: All blocks can now read final counter value
    if (idx == 0) {
        printf("Total blocks that executed: %d\n", *counter);
    }
}

// Warp-level reductions with tiled partitions
__global__ void warpReductionKernel(const float *input, float *output, int n) {
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (idx < n) ? input[idx] : 0.0f;

    // Warp-level reduction using shfl_down
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        val += warp.shfl_down(val, offset);
    }

    // First thread in each warp writes result
    if (warp.thread_rank() == 0) {
        int warp_id = threadIdx.x / 32;
        atomicAdd(&output[blockIdx.x], val);
    }
}

// Coalesced group example - threads with same condition
__global__ void coalescedGroupKernel(int *data, int n) {
    auto block = cg::this_thread_block();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (idx < n) ? data[idx] : 0;

    // Create coalesced group of threads where val > 0
    auto active = cg::coalesced_threads();

    if (val > 0) {
        // Only threads with positive values participate
        int rank = active.thread_rank();
        int size = active.size();

        // Parallel prefix sum within coalesced group
        int sum = val;
        for (int offset = 1; offset < size; offset *= 2) {
            int temp = active.shfl_up(sum, offset);
            if (rank >= offset) {
                sum += temp;
            }
        }

        if (idx < n) {
            data[idx] = sum;  // Store prefix sum
        }
    }
}

// Multi-block reduction using grid group
__global__ void multiBlockReduction(const float *input, float *output,
                                   int n, float *partial_sums) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    __shared__ float shared[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and reduce within block
    shared[tid] = (idx < n) ? input[idx] : 0.0f;
    block.sync();

    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        block.sync();
    }

    // First thread in each block writes partial sum
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared[0];
    }

    // Grid sync - all blocks must finish
    grid.sync();

    // Block 0 reduces partial sums
    if (blockIdx.x == 0 && tid < gridDim.x) {
        shared[tid] = partial_sums[tid];
    } else if (blockIdx.x == 0) {
        shared[tid] = 0.0f;
    }

    if (blockIdx.x == 0) {
        block.sync();

        // Final reduction
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && tid + s < gridDim.x) {
                shared[tid] += shared[tid + s];
            }
            block.sync();
        }

        if (tid == 0) {
            *output = shared[0];
        }
    }
}

// Demonstrate tiled partitions with multiple tile sizes
__global__ void tiledPartitionKernel(int *data, int n) {
    auto block = cg::this_thread_block();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (idx < n) ? data[idx] : 0;

    // Create different sized tiles to demonstrate flexibility
    // Use 16-thread tiles for this example
    auto tile16 = cg::tiled_partition<16>(block);

    if (idx < n) {
        // Sum within 16-thread tile using shuffle
        int sum = val;
        for (int offset = tile16.size() / 2; offset > 0; offset /= 2) {
            sum += tile16.shfl_down(sum, offset);
        }

        // First thread in each tile writes the tile sum
        if (tile16.thread_rank() == 0) {
            data[idx] = sum;
        }
    }
}

int main() {
    printf("=== Advanced Cooperative Groups Demo ===\n");

    // Check device capability
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("\nDevice: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Cooperative launch support: %s\n",
           prop.cooperativeLaunch ? "Yes" : "No");
    printf("Multi-device coop launch: %s\n",
           prop.cooperativeMultiDeviceLaunch ? "Yes" : "No");

    if (prop.major < 6) {
        printf("Error: Cooperative groups require compute capability 6.0+\n");
        return 1;
    }

    const int n = 1024 * 1024;
    const size_t bytes = n * sizeof(int);

    // Allocate memory
    int *h_data = (int*)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_data[i] = i % 100;
    }

    int *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Test 1: Warp-level reductions
    printf("\n=== Warp-Level Reduction ===\n");
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, blocksPerGrid * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_output, 0, blocksPerGrid * sizeof(float)));

    float *h_input = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    warpReductionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    float total = 0.0f;
    float *h_output = (float*)malloc(blocksPerGrid * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, blocksPerGrid * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < blocksPerGrid; i++) {
        total += h_output[i];
    }
    printf("Sum of %d elements: %.0f (expected: %d)\n", n, total, n);

    // Test 2: Coalesced groups
    printf("\n=== Coalesced Groups ===\n");
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    coalescedGroupKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Coalesced group prefix sum completed\n");

    // Test 3: Tiled partitions (16-thread tiles)
    printf("\n=== Tiled Partitions (16-thread tiles) ===\n");
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    tiledPartitionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Tiled partition reduction completed\n");

    // Test 4: Grid-wide synchronization (requires cooperative launch)
    if (prop.cooperativeLaunch) {
        printf("\n=== Grid-Wide Synchronization ===\n");

        int *d_counter;
        CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

        // Cooperative kernel launch
        // Create non-const copy of n for kernel args
        int n_param = n;
        void *args[] = {&d_data, &n_param, &d_counter};

        // Check max blocks for cooperative launch
        int numBlocksPerSm;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocksPerSm, gridSyncKernel, threadsPerBlock, 0));

        int maxBlocks = numBlocksPerSm * prop.multiProcessorCount;
        int coopBlocks = (blocksPerGrid < maxBlocks) ? blocksPerGrid : maxBlocks;

        printf("Launching %d cooperative blocks\n", coopBlocks);

        CUDA_CHECK(cudaLaunchCooperativeKernel(
            (void*)gridSyncKernel, coopBlocks, threadsPerBlock, args, 0, 0));

        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(d_counter);
    } else {
        printf("\n=== Grid-Wide Synchronization ===\n");
        printf("Not supported on this device\n");
    }

    // Cleanup
    free(h_data);
    free(h_input);
    free(h_output);
    cudaFree(d_data);
    cudaFree(d_input);
    cudaFree(d_output);

    printf("\n=== Key Concepts ===\n");
    printf("1. Grid groups enable synchronization across all thread blocks\n");
    printf("2. Tiled partitions create fixed-size subgroups (4, 8, 16, 32 threads)\n");
    printf("3. Coalesced groups adapt to runtime thread activity\n");
    printf("4. Warp-level operations avoid explicit synchronization\n");
    printf("5. Cooperative launch required for grid synchronization\n");
    printf("6. More flexible and efficient than traditional __syncthreads()\n");

    return 0;
}
