// Homework: 60_cooperative_groups_advanced.cu
// Complete based on classwork/60_cooperative_groups_advanced.cu
//
// Instructions:
// 1. Read the corresponding classwork file first
// 2. Implement the exercises below
// 3. Build: make homework (or make both)
// 4. Run your executable from the phase directory

#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

/*
 * EXERCISE 1: Multi-Level Reduction
 * Implement a reduction using:
 * - Warp-level reduction using tiled_partition<32>
 * - Block-level reduction combining warp results
 * - Grid-level reduction combining block results
 * Compare performance to single-level reduction
 */

/*
 * EXERCISE 2: Adaptive Group Size
 * Create a kernel that:
 * - Uses tiled_partition with sizes 4, 8, 16, 32
 * - Dynamically selects size based on workload
 * - Performs group-local operations efficiently
 */

/*
 * EXERCISE 3: Binary Partition Tree
 * Implement a binary tree reduction using:
 * - binary_partition() to split groups recursively
 * - Each level of tree uses smaller partitions
 * - Compare to traditional tree reduction
 */

/*
 * EXERCISE 4: Grid-Wide Histogram
 * Using cooperative groups, implement:
 * - Histogram computation across all blocks
 * - Grid synchronization to combine partial histograms
 * - Atomic-free within-block, atomic across blocks
 */

int main() {
    printf("Homework: 60_cooperative_groups_advanced - Implement the exercises from classwork\n");
    printf("\nExercise 1: Multi-Level Reduction\n");
    printf("Exercise 2: Adaptive Group Size\n");
    printf("Exercise 3: Binary Partition Tree\n");
    printf("Exercise 4: Grid-Wide Histogram\n");

    // Your code here

    return 0;
}
