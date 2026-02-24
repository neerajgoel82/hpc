#!/usr/bin/env python3
"""
Generate Local CUDA .cu Files

Creates standalone .cu files for local compilation with nvcc.
Organized by phase with Makefiles for easy building.
"""

import os
from pathlib import Path

# Define output directory
LOCAL_DIR = Path("local")

# Common header for all CUDA files
COMMON_HEADER = """#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error at %s:%d: %s\\n", \\
                    __FILE__, __LINE__, cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)
"""

# Phase 1 Programs
PHASE1_PROGRAMS = {
    "01_hello_world.cu": """// Phase 1: Hello World from GPU
{header}

__global__ void helloKernel() {{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello from GPU thread %d!\\n", tid);
}}

int main() {{
    printf("=== CUDA Hello World ===\\n\\n");

    // Query device
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Running on: %s\\n\\n", prop.name);

    // Launch kernel
    printf("Launching kernel with 2 blocks, 4 threads each...\\n");
    helloKernel<<<2, 4>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\\nKernel completed successfully!\\n");
    return 0;
}}
""",

    "02_vector_add.cu": """// Phase 1: Vector Addition
{header}

__global__ void vectorAdd(float *a, float *b, float *c, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {{
        c[idx] = a[idx] + b[idx];
    }}
}}

void vectorAddCPU(float *a, float *b, float *c, int n) {{
    for (int i = 0; i < n; i++) {{
        c[i] = a[i] + b[i];
    }}
}}

int main() {{
    printf("=== Vector Addition: CPU vs GPU ===\\n\\n");

    int n = 10000000;  // 10M elements
    size_t size = n * sizeof(float);
    printf("Vector size: %d elements (%.2f MB)\\n\\n", n, size / 1024.0 / 1024.0);

    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c_cpu = (float*)malloc(size);
    float *h_c_gpu = (float*)malloc(size);

    // Initialize
    for (int i = 0; i < n; i++) {{
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }}

    // CPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float cpu_time, gpu_time;

    cudaEventRecord(start);
    vectorAddCPU(h_a, h_b, h_c_cpu, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_time, start, stop);

    // GPU computation
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost));
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    // Verify
    bool correct = true;
    for (int i = 0; i < n; i++) {{
        if (h_c_cpu[i] != h_c_gpu[i]) {{
            correct = false;
            break;
        }}
    }}

    printf("Results:\\n");
    printf("  CPU Time: %.2f ms\\n", cpu_time);
    printf("  GPU Time: %.2f ms\\n", gpu_time);
    printf("  Speedup: %.2fx\\n", cpu_time / gpu_time);
    printf("  Verification: %s\\n", correct ? "PASSED" : "FAILED");

    // Cleanup
    free(h_a); free(h_b); free(h_c_cpu); free(h_c_gpu);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}}
""",

    "03_matrix_add.cu": """// Phase 1: 2D Matrix Addition
{header}

__global__ void matrixAdd(float *a, float *b, float *c, int width, int height) {{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {{
        int idx = row * width + col;
        c[idx] = a[idx] + b[idx];
    }}
}}

int main() {{
    printf("=== 2D Matrix Addition ===\\n\\n");

    int width = 4096;
    int height = 4096;
    size_t size = width * height * sizeof(float);

    printf("Matrix size: %dx%d (%.2f MB)\\n\\n", width, height, size / 1024.0 / 1024.0);

    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize matrices
    for (int i = 0; i < width * height; i++) {{
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }}

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // 2D grid and block configuration
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    printf("Grid: %dx%d blocks\\n", gridDim.x, gridDim.y);
    printf("Block: %dx%d threads\\n\\n", blockDim.x, blockDim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixAdd<<<gridDim, blockDim>>>(d_a, d_b, d_c, width, height);
    cudaEventRecord(stop);

    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Verify
    bool correct = true;
    for (int i = 0; i < width * height; i++) {{
        if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5) {{
            correct = false;
            break;
        }}
    }}

    printf("Time: %.2f ms\\n", milliseconds);
    printf("Bandwidth: %.2f GB/s\\n", (3 * size / 1e9) / (milliseconds / 1000.0));
    printf("Verification: %s\\n", correct ? "PASSED" : "FAILED");

    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}}
""",
}

# Phase 2 Programs
PHASE2_PROGRAMS = {
    "01_memory_bandwidth.cu": """// Phase 2: Memory Bandwidth Benchmark
{header}

float measureBandwidth(size_t size, bool pinned, bool h2d) {{
    float *h_data, *d_data;

    if (pinned) {{
        CUDA_CHECK(cudaMallocHost(&h_data, size));
    }} else {{
        h_data = (float*)malloc(size);
    }}
    CUDA_CHECK(cudaMalloc(&d_data, size));

    // Initialize
    for (size_t i = 0; i < size/sizeof(float); i++) {{
        h_data[i] = i * 1.0f;
    }}

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (h2d) {{
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    }} else {{
        CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    }}
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float bandwidth = (size / 1e9) / (milliseconds / 1000.0);

    if (pinned) {{
        cudaFreeHost(h_data);
    }} else {{
        free(h_data);
    }}
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return bandwidth;
}}

int main() {{
    printf("=== Memory Bandwidth Benchmark ===\\n\\n");

    size_t sizes[] = {{1<<20, 1<<22, 1<<24, 1<<26}};  // 1MB to 64MB

    printf("%-15s %-20s %-20s %-20s %-20s\\n",
           "Size (MB)", "Pageable H→D", "Pinned H→D", "Pageable D→H", "Pinned D→H");
    printf("---------------------------------------------------------------------------------\\n");

    for (int i = 0; i < 4; i++) {{
        size_t size = sizes[i];
        float bw_page_h2d = measureBandwidth(size, false, true);
        float bw_pin_h2d = measureBandwidth(size, true, true);
        float bw_page_d2h = measureBandwidth(size, false, false);
        float bw_pin_d2h = measureBandwidth(size, true, false);

        printf("%-15zu %-20.2f %-20.2f %-20.2f %-20.2f\\n",
               size / (1024*1024), bw_page_h2d, bw_pin_h2d, bw_page_d2h, bw_pin_d2h);
    }}

    printf("\\nPinned memory provides significantly better bandwidth!\\n");
    return 0;
}}
""",

    "02_shared_memory.cu": """// Phase 2: Shared Memory Demonstration
{header}

#define TILE_SIZE 256

__global__ void reverseArrayShared(float *input, float *output, int n) {{
    __shared__ float tile[TILE_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory
    if (idx < n) {{
        tile[threadIdx.x] = input[idx];
    }}
    __syncthreads();

    // Write in reverse order within block
    if (idx < n) {{
        output[idx] = tile[blockDim.x - 1 - threadIdx.x];
    }}
}}

__global__ void reverseArrayGlobal(float *input, float *output, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {{
        int reverseIdx = (blockIdx.x + 1) * blockDim.x - 1 - threadIdx.x;
        if (reverseIdx < n) {{
            output[idx] = input[reverseIdx];
        }}
    }}
}}

int main() {{
    printf("=== Shared Memory Performance ===\\n\\n");

    int n = 10000000;
    size_t size = n * sizeof(float);

    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    for (int i = 0; i < n; i++) h_input[i] = i;

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = TILE_SIZE;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Test shared memory
    cudaEventRecord(start);
    reverseArrayShared<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_shared;
    cudaEventElapsedTime(&time_shared, start, stop);

    // Test global memory
    cudaEventRecord(start);
    reverseArrayGlobal<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_global;
    cudaEventElapsedTime(&time_global, start, stop);

    printf("Array size: %d elements\\n", n);
    printf("Shared memory: %.3f ms\\n", time_shared);
    printf("Global memory: %.3f ms\\n", time_global);
    printf("Speedup: %.2fx\\n\\n", time_global / time_shared);
    printf("Shared memory is faster due to on-chip access!\\n");

    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}}
""",

    "03_coalescing.cu": """// Phase 2: Memory Coalescing
{header}

__global__ void coalescedAccess(float *data, float *result, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {{
        result[idx] = data[idx] * 2.0f;
    }}
}}

__global__ void stridedAccess(float *data, float *result, int n, int stride) {{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (idx < n) {{
        result[idx] = data[idx] * 2.0f;
    }}
}}

int main() {{
    printf("=== Memory Coalescing Impact ===\\n\\n");

    int n = 10000000;
    size_t size = n * sizeof(float);

    float *d_data, *d_result;
    CUDA_CHECK(cudaMalloc(&d_data, size));
    CUDA_CHECK(cudaMalloc(&d_result, size));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Coalesced access
    cudaEventRecord(start);
    coalescedAccess<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_result, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_coalesced;
    cudaEventElapsedTime(&time_coalesced, start, stop);

    // Strided access (stride = 32)
    blocksPerGrid = (n/32 + threadsPerBlock - 1) / threadsPerBlock;
    cudaEventRecord(start);
    stridedAccess<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_result, n, 32);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_strided;
    cudaEventElapsedTime(&time_strided, start, stop);

    float bandwidth_coalesced = (size * 2 / 1e9) / (time_coalesced / 1000.0);
    float bandwidth_strided = (size * 2 / 1e9) / (time_strided / 1000.0);

    printf("Coalesced access: %.3f ms (%.2f GB/s)\\n", time_coalesced, bandwidth_coalesced);
    printf("Strided access:   %.3f ms (%.2f GB/s)\\n", time_strided, bandwidth_strided);
    printf("Performance degradation: %.2fx slower\\n\\n", time_strided / time_coalesced);
    printf("KEY: Adjacent threads should access adjacent memory!\\n");

    cudaFree(d_data); cudaFree(d_result);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}}
""",
}

# Phase 3 Programs
PHASE3_PROGRAMS = {
    "01_reduction.cu": """// Phase 3: Parallel Reduction
{header}

__global__ void reductionOptimized(float *input, float *output, int n) {{
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Load and add during load
    sdata[tid] = 0;
    if (idx < n) sdata[tid] = input[idx];
    if (idx + blockDim.x < n) sdata[tid] += input[idx + blockDim.x];
    __syncthreads();

    // Sequential addressing (no divergence)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            sdata[tid] += sdata[tid + s];
        }}
        __syncthreads();
    }}

    if (tid == 0) {{
        output[blockIdx.x] = sdata[0];
    }}
}}

float reductionCPU(float *data, int n) {{
    float sum = 0;
    for (int i = 0; i < n; i++) {{
        sum += data[i];
    }}
    return sum;
}}

int main() {{
    printf("=== Optimized Parallel Reduction ===\\n\\n");

    int n = 10000000;
    size_t size = n * sizeof(float);

    float *h_input = (float*)malloc(size);
    for (int i = 0; i < n; i++) h_input[i] = 1.0f;

    // CPU reduction
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    float cpu_sum = reductionCPU(h_input, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float cpu_time;
    cudaEventElapsedTime(&cpu_time, start, stop);

    // GPU reduction
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
    CUDA_CHECK(cudaMalloc(&d_output, blocksPerGrid * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    cudaEventRecord(start);
    reductionOptimized<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    cudaEventRecord(stop);

    float *h_output = (float*)malloc(blocksPerGrid * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventSynchronize(stop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // Final reduction on CPU
    float gpu_sum = 0;
    for (int i = 0; i < blocksPerGrid; i++) {{
        gpu_sum += h_output[i];
    }}

    printf("Array size: %d elements\\n", n);
    printf("CPU sum: %.0f (%.2f ms)\\n", cpu_sum, cpu_time);
    printf("GPU sum: %.0f (%.2f ms)\\n", gpu_sum, gpu_time);
    printf("Speedup: %.2fx\\n", cpu_time / gpu_time);
    printf("Verification: %s\\n", (cpu_sum == gpu_sum) ? "PASSED" : "FAILED");

    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}}
""",
}

def create_makefile(phase_dir, programs):
    """Generate Makefile for a phase"""
    makefile = """# Makefile for Phase CUDA programs
NVCC = nvcc
NVCC_FLAGS = -arch=sm_70 -O2
TARGETS = """ + " ".join([p.replace(".cu", "") for p in programs.keys()]) + """

all: $(TARGETS)

%: %.cu
\t$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
\trm -f $(TARGETS)

test: all
\t@for target in $(TARGETS); do \\
\t\techo "Running $$target..."; \\
\t\t./$$target || exit 1; \\
\t\techo ""; \\
\tdone

.PHONY: all clean test
"""
    return makefile

def generate_phase_programs(phase_num, programs):
    """Generate all programs for a phase"""
    phase_dir = LOCAL_DIR / f"phase{phase_num}"
    phase_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating Phase {phase_num}...")

    for filename, template in programs.items():
        filepath = phase_dir / filename
        content = template.format(header=COMMON_HEADER)

        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✓ Created {filename}")

    # Create Makefile
    makefile_path = phase_dir / "Makefile"
    with open(makefile_path, 'w') as f:
        f.write(create_makefile(phase_dir, programs))
    print(f"  ✓ Created Makefile")

    # Create README
    readme_path = phase_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(f"""# Phase {phase_num} - Local CUDA Programs

## Building

```bash
make          # Build all programs
make clean    # Clean build artifacts
make test     # Build and run all programs
```

## Running Individual Programs

```bash
./program_name
```

## Programs

""")
        for filename in programs.keys():
            f.write(f"- `{filename}`: {filename.replace('.cu', '').replace('_', ' ').title()}\\n")

    print(f"  ✓ Created README.md\\n")

def main():
    print("="*70)
    print("CUDA Local Programs Generator")
    print("="*70)
    print()

    # Generate Phase 1
    generate_phase_programs(1, PHASE1_PROGRAMS)

    # Generate Phase 2
    generate_phase_programs(2, PHASE2_PROGRAMS)

    # Generate Phase 3
    generate_phase_programs(3, PHASE3_PROGRAMS)

    # Create root Makefile
    root_makefile = LOCAL_DIR / "Makefile"
    with open(root_makefile, 'w') as f:
        f.write("""# Root Makefile for all phases

PHASES = phase1 phase2 phase3

all:
\t@for phase in $(PHASES); do \\
\t\techo "Building $$phase..."; \\
\t\t$(MAKE) -C $$phase || exit 1; \\
\tdone

clean:
\t@for phase in $(PHASES); do \\
\t\t$(MAKE) -C $$phase clean; \\
\tdone

test:
\t@for phase in $(PHASES); do \\
\t\techo "Testing $$phase..."; \\
\t\t$(MAKE) -C $$phase test || exit 1; \\
\tdone

.PHONY: all clean test
""")
    print("✓ Created root Makefile")

    print()
    print("="*70)
    print("Generation Complete!")
    print("="*70)
    print()
    print("To build all programs: cd local && make")
    print("To test all programs:  cd local && make test")
    print("To clean:             cd local && make clean")

if __name__ == "__main__":
    main()
