#!/usr/bin/env python3
"""
Generate ALL Local CUDA .cu Files for Phases 1-9

Creates comprehensive standalone .cu files for local compilation.
"""

import os
from pathlib import Path

LOCAL_DIR = Path("local")

COMMON_HEADER = """#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

# All phase programs - abbreviated for space
ALL_PROGRAMS = {
    4: {  # Phase 4: Advanced Memory
        "01_atomics.cu": """// Atomic Operations
{header}

__global__ void atomicHistogram(int *input, int *histogram, int n, int bins) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {{
        int bin = input[idx] % bins;
        atomicAdd(&histogram[bin], 1);
    }}
}}

int main() {{
    printf("=== Atomic Histogram ===\\n\\n");
    int n = 1000000;
    int bins = 256;

    int *h_input = (int*)malloc(n * sizeof(int));
    int *h_histogram = (int*)calloc(bins, sizeof(int));

    for (int i = 0; i < n; i++) h_input[i] = rand();

    int *d_input, *d_histogram;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_histogram, bins * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_histogram, 0, bins * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    atomicHistogram<<<(n+255)/256, 256>>>(d_input, d_histogram, n, bins);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_histogram, d_histogram, bins * sizeof(int), cudaMemcpyDeviceToHost));

    int total = 0;
    for (int i = 0; i < bins; i++) total += h_histogram[i];

    printf("Processed %d elements\\n", n);
    printf("Time: %.2f ms\\n", ms);
    printf("Verification: %s\\n", (total == n) ? "PASSED" : "FAILED");

    free(h_input); free(h_histogram);
    cudaFree(d_input); cudaFree(d_histogram);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}}
""",
    },

    5: {  # Phase 5: Advanced Algorithms
        "01_matmul_tiled.cu": """// Tiled Matrix Multiplication
{header}

#define TILE_WIDTH 16

__global__ void matmulTiled(float *A, float *B, float *C, int N) {{
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_WIDTH - 1) / TILE_WIDTH; t++) {{
        if (row < N && t * TILE_WIDTH + threadIdx.x < N)
            sA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_WIDTH + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_WIDTH + threadIdx.y < N)
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }}

    if (row < N && col < N)
        C[row * N + col] = sum;
}}

int main() {{
    printf("=== Tiled Matrix Multiplication ===\\n\\n");

    int N = 1024;
    size_t size = N * N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < N*N; i++) {{
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }}

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    float gflops = (2.0f * N * N * N) / (ms / 1000.0) / 1e9;

    printf("Matrix size: %dx%d\\n", N, N);
    printf("Time: %.2f ms\\n", ms);
    printf("Performance: %.2f GFLOPS\\n", gflops);

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}}
""",
    },

    6: {  # Phase 6: Streams
        "01_streams.cu": """// CUDA Streams
{header}

__global__ void kernel(float *data, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {{
        data[idx] = sqrt(data[idx]) * 2.0f;
    }}
}}

int main() {{
    printf("=== CUDA Streams ===\\n\\n");

    int nStreams = 4;
    int n = 10000000;
    int streamSize = n / nStreams;
    size_t streamBytes = streamSize * sizeof(float);

    float *h_data;
    CUDA_CHECK(cudaMallocHost(&h_data, n * sizeof(float)));

    for (int i = 0; i < n; i++) h_data[i] = i;

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(float)));

    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; i++)
        cudaStreamCreate(&streams[i]);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 256;
    int blocksPerStream = (streamSize + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start);

    for (int i = 0; i < nStreams; i++) {{
        int offset = i * streamSize;
        CUDA_CHECK(cudaMemcpyAsync(d_data + offset, h_data + offset,
                                   streamBytes, cudaMemcpyHostToDevice, streams[i]));
        kernel<<<blocksPerStream, threadsPerBlock, 0, streams[i]>>>(d_data + offset, streamSize);
        CUDA_CHECK(cudaMemcpyAsync(h_data + offset, d_data + offset,
                                   streamBytes, cudaMemcpyDeviceToHost, streams[i]));
    }}

    cudaEventRecord(stop);

    for (int i = 0; i < nStreams; i++)
        cudaStreamSynchronize(streams[i]);

    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Streams: %d\\n", nStreams);
    printf("Elements per stream: %d\\n", streamSize);
    printf("Time: %.2f ms\\n", ms);
    printf("Throughput: %.2f GB/s\\n", (n * sizeof(float) * 3 / 1e9) / (ms / 1000.0));

    for (int i = 0; i < nStreams; i++)
        cudaStreamDestroy(streams[i]);

    cudaFreeHost(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}}
""",
    },

    7: {  # Phase 7: Performance
        "01_occupancy.cu": """// Occupancy Tuning
{header}

__global__ void kernel(float *data, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {{
        float x = data[idx];
        for (int i = 0; i < 100; i++) {{
            x = sqrt(x + 1.0f);
        }}
        data[idx] = x;
    }}
}}

int main() {{
    printf("=== Occupancy Tuning ===\\n\\n");

    int n = 1000000;
    size_t size = n * sizeof(float);

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));

    int blockSizes[] = {{32, 64, 128, 256, 512, 1024}};

    printf("%-15s %-15s %-15s\\n", "Block Size", "Time (ms)", "Bandwidth (GB/s)");
    printf("-----------------------------------------------\\n");

    for (int i = 0; i < 6; i++) {{
        int blockSize = blockSizes[i];
        int gridSize = (n + blockSize - 1) / blockSize;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        kernel<<<gridSize, blockSize>>>(d_data, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        float bandwidth = (size * 2 / 1e9) / (ms / 1000.0);

        printf("%-15d %-15.2f %-15.2f\\n", blockSize, ms, bandwidth);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }}

    cudaFree(d_data);
    return 0;
}}
""",
    },

    8: {  # Phase 8: Applications
        "01_nbody.cu": """// N-Body Simulation
{header}

__global__ void nbodyKernel(float *pos, float *vel, int n, float dt) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float px = pos[i*3], py = pos[i*3+1], pz = pos[i*3+2];
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    for (int j = 0; j < n; j++) {{
        float dx = pos[j*3] - px;
        float dy = pos[j*3+1] - py;
        float dz = pos[j*3+2] - pz;
        float dist = sqrt(dx*dx + dy*dy + dz*dz + 1e-10f);
        float f = 1.0f / (dist * dist * dist);
        fx += dx * f;
        fy += dy * f;
        fz += dz * f;
    }}

    vel[i*3] += fx * dt;
    vel[i*3+1] += fy * dt;
    vel[i*3+2] += fz * dt;
    pos[i*3] += vel[i*3] * dt;
    pos[i*3+1] += vel[i*3+1] * dt;
    pos[i*3+2] += vel[i*3+2] * dt;
}}

int main() {{
    printf("=== N-Body Simulation ===\\n\\n");

    int n = 4096;
    float dt = 0.01f;
    int steps = 10;

    float *h_pos = (float*)malloc(n * 3 * sizeof(float));
    float *h_vel = (float*)malloc(n * 3 * sizeof(float));

    for (int i = 0; i < n * 3; i++) {{
        h_pos[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        h_vel[i] = 0.0f;
    }}

    float *d_pos, *d_vel;
    CUDA_CHECK(cudaMalloc(&d_pos, n * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vel, n * 3 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_pos, h_pos, n * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel, h_vel, n * 3 * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int step = 0; step < steps; step++) {{
        nbodyKernel<<<(n+255)/256, 256>>>(d_pos, d_vel, n, dt);
    }}

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Bodies: %d\\n", n);
    printf("Steps: %d\\n", steps);
    printf("Total time: %.2f ms\\n", ms);
    printf("Time per step: %.2f ms\\n", ms / steps);
    printf("Interactions/sec: %.2f million\\n", (n * n * steps / 1e6) / (ms / 1000.0));

    free(h_pos); free(h_vel);
    cudaFree(d_pos); cudaFree(d_vel);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}}
""",
    },

    9: {  # Phase 9: Modern CUDA
        "01_cuda_graphs.cu": """// CUDA Graphs
{header}

__global__ void kernel1(float *data, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= 2.0f;
}}

__global__ void kernel2(float *data, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += 1.0f;
}}

int main() {{
    printf("=== CUDA Graphs ===\\n\\n");

    int n = 10000000;
    size_t size = n * sizeof(float);

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));

    int blocks = (n + 255) / 256;
    int threads = 256;

    // Create graph
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaStream_t stream;

    cudaStreamCreate(&stream);

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    kernel1<<<blocks, threads, 0, stream>>>(d_data, n);
    kernel2<<<blocks, threads, 0, stream>>>(d_data, n);
    cudaStreamEndCapture(stream, &graph);

    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    // Time graph execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {{
        cudaGraphLaunch(graphExec, stream);
    }}
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Graph with 2 kernels\\n");
    printf("Launches: 1000\\n");
    printf("Total time: %.2f ms\\n", ms);
    printf("Time per launch: %.3f ms\\n", ms / 1000.0);

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}}
""",
    },
}

def create_makefile(phase, programs):
    targets = " ".join([p.replace(".cu", "") for p in programs.keys()])
    return f"""# Makefile for Phase {phase}
NVCC = nvcc
NVCC_FLAGS = -arch=sm_70 -O2
TARGETS = {targets}

all: $(TARGETS)

%: %.cu
\t$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
\trm -f $(TARGETS)

test: all
\t@for target in $(TARGETS); do \\
\t\techo "=== Running $$target ===" ; \\
\t\t./$$target || exit 1; \\
\t\techo ""; \\
\tdone

.PHONY: all clean test
"""

def generate_all_phases():
    print("="*70)
    print("Generating All Phase CUDA Programs")
    print("="*70)
    print()

    for phase_num, programs in ALL_PROGRAMS.items():
        phase_dir = LOCAL_DIR / f"phase{phase_num}"
        phase_dir.mkdir(parents=True, exist_ok=True)

        print(f"Phase {phase_num}: Generating {len(programs)} programs...")

        for filename, template in programs.items():
            filepath = phase_dir / filename
            content = template.format(header=COMMON_HEADER)
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"  ✓ {filename}")

        # Makefile
        with open(phase_dir / "Makefile", 'w') as f:
            f.write(create_makefile(phase_num, programs))

        # README
        with open(phase_dir / "README.md", 'w') as f:
            f.write(f"# Phase {phase_num} - Local CUDA Programs\\n\\n")
            f.write("## Build\\n```bash\\nmake\\n```\\n\\n")
            f.write("## Run\\n```bash\\nmake test\\n```\\n")

        print()

    # Root Makefile
    phases = " ".join([f"phase{p}" for p in ALL_PROGRAMS.keys()])
    with open(LOCAL_DIR / "Makefile", 'w') as f:
        f.write(f"""# Root Makefile
PHASES = phase1 phase2 phase3 {phases}

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
\t\techo "\\n=== Testing $$phase ==="; \\
\t\t$(MAKE) -C $$phase test || exit 1; \\
\tdone

.PHONY: all clean test
""")

    print("="*70)
    print(f"Generated programs for phases 4-9!")
    print("="*70)
    print()
    print("Total: ", sum(len(p) for p in ALL_PROGRAMS.values()), " new programs")

if __name__ == "__main__":
    generate_all_phases()
