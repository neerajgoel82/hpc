// Homework: 59_async_barriers.cu
// Complete based on classwork/59_async_barriers.cu
//
// Instructions:
// 1. Read the corresponding classwork file first
// 2. Implement the exercises below
// 3. Build: make homework (or make both)
// 4. Run your executable from the phase directory

#include <stdio.h>
#include <cuda_runtime.h>

/*
 * EXERCISE 1: Triple Buffering
 * Extend double-buffering to triple buffering:
 * - Use 3 buffers with 3 barriers
 * - Load buffer N, process buffer N-1, store buffer N-2
 * - Measure performance vs double-buffering
 */

/*
 * EXERCISE 2: Producer-Consumer with Multiple Producers
 * Implement a pattern with:
 * - Multiple producer threads per consumer
 * - Each producer loads different data portions
 * - Single consumer processes combined data
 * - Use barriers to coordinate
 */

/*
 * EXERCISE 3: Hierarchical Barriers
 * Create a kernel with:
 * - Warp-level barriers for intra-warp sync
 * - Block-level barriers for inter-warp sync
 * - Implement a hierarchical reduction using this pattern
 */

int main() {
    printf("Homework: 59_async_barriers - Implement the exercises from classwork\n");
    printf("\nExercise 1: Triple Buffering\n");
    printf("Exercise 2: Producer-Consumer with Multiple Producers\n");
    printf("Exercise 3: Hierarchical Barriers\n");

    // Your code here

    return 0;
}
