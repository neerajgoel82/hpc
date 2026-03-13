// Homework: 57_memcpy_async.cu
// Complete based on classwork/57_memcpy_async.cu
//
// Instructions:
// 1. Read the corresponding classwork file first
// 2. Implement the exercises below
// 3. Build: make homework (or make both)
// 4. Run your executable from the phase directory

#include <stdio.h>
#include <cuda_runtime.h>

/*
 * EXERCISE 1: Three-Stage Pipeline
 * Extend the classwork's 2-stage pipeline to 3 stages:
 * - Stage 0: Load chunk N
 * - Stage 1: Process chunk N-1 while loading N
 * - Stage 2: Store chunk N-2 while processing N-1 and loading N
 */

/*
 * EXERCISE 2: Adaptive Pipeline Depth
 * Create a kernel that:
 * - Dynamically adjusts pipeline depth based on data size
 * - Uses 1 stage for small data, up to 4 stages for large data
 * - Measures performance improvement
 */

/*
 * EXERCISE 3: Image Processing Pipeline
 * Implement a 2D image convolution using async copy:
 * - Load image tiles asynchronously
 * - Apply 3x3 convolution filter
 * - Overlap loading next tile with processing current tile
 */

int main() {
    printf("Homework: 57_memcpy_async - Implement the exercises from classwork\n");
    printf("\nExercise 1: Three-Stage Pipeline\n");
    printf("Exercise 2: Adaptive Pipeline Depth\n");
    printf("Exercise 3: Image Processing Pipeline\n");

    // Your code here

    return 0;
}
