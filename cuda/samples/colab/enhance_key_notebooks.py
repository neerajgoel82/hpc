#!/usr/bin/env python3
"""
Enhance key notebooks with comprehensive CUDA code examples
"""

import json
import os

base_dir = "/Users/negoel/code/mywork/github/neerajgoel82/cuda-samples/colab/notebooks"

# Enhanced notebook 10: Tiled Matrix Multiplication
notebook_10_examples = [
    {
        "title": "## Example 1: Naive Matrix Multiplication (Baseline)",
        "code": """%%cu
#include <stdio.h>
#include <stdlib.h>

__global__ void matMulNaive(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int N = 512;
    size_t size = N * N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Naive Matrix Multiplication (%dx%d)\\\\n", N, N);
    printf("Time: %.3f ms\\\\n", milliseconds);
    printf("GFLOPS: %.2f\\\\n", (2.0 * N * N * N) / (milliseconds * 1e6));

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify sample
    printf("Sample result: C[0][0] = %.4f\\\\n", h_C[0]);

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}"""
    },
    {
        "title": "## Example 2: Tiled Matrix Multiplication with Shared Memory",
        "code": """%%cu
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 16

__global__ void matMulTiled(float *A, float *B, float *C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile into shared memory
        if (row < N && (t * TILE_SIZE + tx) < N)
            As[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if (col < N && (t * TILE_SIZE + ty) < N)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    int N = 1024;
    size_t size = N * N * sizeof(float);

    printf("Matrix Multiplication: %dx%d\\\\n", N, N);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (N + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("\\\\nTiled Matrix Multiplication (Tile Size: %d)\\\\n", TILE_SIZE);
    printf("Time: %.3f ms\\\\n", milliseconds);
    printf("GFLOPS: %.2f\\\\n", (2.0 * N * N * N) / (milliseconds * 1e6));

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}"""
    },
    {
        "title": "## Example 3: Comparing Naive vs Tiled Performance",
        "code": """%%cu
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 16

__global__ void matMulNaive(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matMulTiled(float *A, float *B, float *C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < N && (t * TILE_SIZE + threadIdx.x) < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (t * TILE_SIZE + threadIdx.y) < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    int sizes[] = {128, 256, 512, 1024};

    printf("Performance Comparison: Naive vs Tiled Matrix Multiplication\\\\n");
    printf("================================================================\\\\n\\\\n");

    for (int s = 0; s < 4; s++) {
        int N = sizes[s];
        size_t size = N * N * sizeof(float);

        float *h_A = (float*)malloc(size);
        float *h_B = (float*)malloc(size);
        float *h_C = (float*)malloc(size);

        for (int i = 0; i < N * N; i++) {
            h_A[i] = 1.0f;
            h_B[i] = 1.0f;
        }

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C, size);

        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        dim3 blockDim(16, 16);
        dim3 gridDim((N + 15) / 16, (N + 15) / 16);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Naive version
        cudaEventRecord(start);
        matMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float naiveTime = 0;
        cudaEventElapsedTime(&naiveTime, start, stop);

        // Tiled version
        cudaEventRecord(start);
        matMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float tiledTime = 0;
        cudaEventElapsedTime(&tiledTime, start, stop);

        float speedup = naiveTime / tiledTime;

        printf("Size: %dx%d\\\\n", N, N);
        printf("  Naive: %.3f ms (%.2f GFLOPS)\\\\n",
               naiveTime, (2.0 * N * N * N) / (naiveTime * 1e6));
        printf("  Tiled: %.3f ms (%.2f GFLOPS)\\\\n",
               tiledTime, (2.0 * N * N * N) / (tiledTime * 1e6));
        printf("  Speedup: %.2fx\\\\n\\\\n", speedup);

        free(h_A); free(h_B); free(h_C);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_c);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}"""
    }
]

def update_notebook_with_examples(filepath, examples):
    """Update a notebook with comprehensive examples"""
    with open(filepath, 'r') as f:
        notebook = json.load(f)

    # Keep first 2 cells (title and concepts)
    new_cells = notebook['cells'][:2]

    # Add example cells
    for example in examples:
        # Add markdown title
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": example["title"].split('\\n')
        })

        # Add code cell
        new_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": example["code"].split('\\n')
        })

    # Add exercise, key takeaways, next steps, and notes
    new_cells.extend([
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Practical Exercise\n",
                "\n",
                "**Exercise 1:** Modify the tile size and measure performance impact\n",
                "\n",
                "**Exercise 2:** Implement verification to compare naive vs tiled results\n",
                "\n",
                "**Exercise 3:** Add support for non-square matrices\n",
                "\n",
                "**Exercise 4:** Profile memory access patterns"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%cu\n",
                "// Your solution here\n",
                "#include <stdio.h>\n",
                "\n",
                "int main() {\n",
                "    // TODO: Implement your solution\n",
                "    return 0;\n",
                "}"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Key Takeaways\n",
                "\n",
                "1. **Tiling** reduces global memory accesses dramatically\n",
                "2. **Shared memory** is ~100x faster than global memory\n",
                "3. **__syncthreads()** ensures all threads have loaded data\n",
                "4. Tile size affects occupancy and performance\n",
                "5. Typical speedup: 5-10x over naive implementation\n",
                "6. Memory reuse is key to GPU performance\n",
                "7. Blocking is a fundamental optimization technique"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Next Steps\n",
                "\n",
                "In the next notebook, we'll learn:\n",
                "- Memory coalescing patterns\n",
                "- Aligned vs unaligned access\n",
                "- Measuring memory bandwidth\n",
                "- Optimizing access patterns\n",
                "\n",
                "Continue to: **11_coalescing_demo.ipynb**"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Notes\n",
                "\n",
                "*Use this space to write your own notes and observations:*\n",
                "\n",
                "---\n",
                "\n",
                "\n",
                "\n",
                "---"
            ]
        }
    ])

    notebook['cells'] = new_cells

    with open(filepath, 'w') as f:
        json.dump(notebook, f, indent=1)

# Update notebook 10
filepath_10 = os.path.join(base_dir, "phase2/10_tiled_matrix_multiplication.ipynb")
update_notebook_with_examples(filepath_10, notebook_10_examples)
print(f"Enhanced {filepath_10}")

print("\\nKey notebooks enhanced successfully!")
