// Homework: 61_cutensor_basics.cu
// Complete based on classwork/61_cutensor_basics.cu
//
// Instructions:
// 1. Read the corresponding classwork file first
// 2. Implement the exercises below
// 3. Build: make homework (or make both)
// 4. Run your executable from the phase directory

#include <stdio.h>
#include <cuda_runtime.h>
#include <cutensor.h>

/*
 * EXERCISE 1: Batched Matrix Multiplication
 * Implement tensor contraction for batched matrix multiply:
 * - Input: A[batch, m, k] and B[batch, k, n]
 * - Output: C[batch, m, n]
 * - Einstein notation: C_bmn = A_bmk * B_bkn
 * - Compare performance to loop of cuBLAS calls
 */

/*
 * EXERCISE 2: 3D Tensor Contraction
 * Implement a more complex contraction:
 * - A[i,j,k] * B[k,l,m] -> C[i,j,l,m]
 * - Einstein notation: C_ijlm = A_ijk * B_klm
 * - Use tensors of size 64x64x64 and 64x64x64
 */

/*
 * EXERCISE 3: Tensor Reduction
 * Implement tensor reduction operations:
 * - Sum over one dimension: A[m,n,k] -> B[m,n] (sum over k)
 * - Use cutensorReduce operation
 * - Compare to custom kernel implementation
 */

/*
 * EXERCISE 4: Multi-Dimensional Transpose
 * Implement permutation for 4D tensor:
 * - A[i,j,k,l] -> B[k,i,l,j]
 * - Measure bandwidth achieved
 * - Compare to nested loop transpose
 */

/*
 * EXERCISE 5: Mixed Precision
 * Implement contraction with mixed precision:
 * - Inputs in FP16 (CUDA_R_16F)
 * - Accumulation in FP32 (CUTENSOR_COMPUTE_32F)
 * - Compare accuracy and performance to pure FP32
 */

int main() {
    printf("Homework: 61_cutensor_basics - Implement the exercises from classwork\n");
    printf("\nExercise 1: Batched Matrix Multiplication\n");
    printf("Exercise 2: 3D Tensor Contraction\n");
    printf("Exercise 3: Tensor Reduction\n");
    printf("Exercise 4: Multi-Dimensional Transpose\n");
    printf("Exercise 5: Mixed Precision Contraction\n");

    // Your code here

    return 0;
}
