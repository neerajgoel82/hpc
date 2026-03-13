// Homework: 56_async_memory_ops.cu
// Complete based on classwork/56_async_memory_ops.cu
//
// Instructions:
// 1. Read the corresponding classwork file first
// 2. Implement the exercises below
// 3. Build: make homework (or make both)
// 4. Run your executable from the phase directory

#include <stdio.h>
#include <cuda_runtime.h>

/*
 * EXERCISE 1: Memory Pool Configuration
 * Implement a function that creates a custom memory pool with specific settings:
 * - Set the release threshold to 64MB
 * - Enable reuse of memory within the pool
 * - Benchmark allocation speed vs default pool
 */

/*
 * EXERCISE 2: Multi-Stream Async Allocation
 * Create a program that:
 * - Uses 4 different CUDA streams
 * - Each stream allocates, processes, and frees memory independently
 * - Measures total time and compares to sequential allocation
 */

/*
 * EXERCISE 3: Dynamic Workload Pattern
 * Implement a realistic pattern where:
 * - Array sizes vary per iteration (simulate dynamic workload)
 * - Use cudaMallocAsync for efficient handling
 * - Compare performance to pre-allocating max size
 */

int main() {
    printf("Homework: 56_async_memory_ops - Implement the exercises from classwork\n");
    printf("\nExercise 1: Memory Pool Configuration\n");
    printf("Exercise 2: Multi-Stream Async Allocation\n");
    printf("Exercise 3: Dynamic Workload Pattern\n");

    // Your code here

    return 0;
}
