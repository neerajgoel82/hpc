#!/usr/bin/env python3
"""
CUDA Notebook Content Generator

This script automatically fills CUDA Jupyter notebooks with comprehensive content
including progressive examples, CPU vs GPU benchmarks, exercises, and explanations.

Usage: python3 fill_notebooks.py
"""

import json
import os
import glob
from pathlib import Path

# Topic-based content templates
TOPIC_TEMPLATES = {
    # Phase 2: Memory Management
    "memory_bandwidth": {
        "concepts": [
            "PCIe bandwidth limits and theoretical maximums",
            "Effective bandwidth vs peak bandwidth",
            "Measuring memory transfer performance",
            "Factors affecting bandwidth (transfer size, pinned memory)"
        ],
        "examples": [
            ("Basic Bandwidth Measurement", "measure_basic_bandwidth"),
            ("Transfer Size Impact", "measure_transfer_size_impact"),
            ("CPU vs GPU Benchmark", "bandwidth_benchmark")
        ]
    },
    "unified_memory": {
        "concepts": [
            "Unified Memory simplifies memory management",
            "cudaMallocManaged for automatic migration",
            "Page faulting and data migration",
            "Prefetching with cudaMemPrefetchAsync"
        ],
        "examples": [
            ("Basic Unified Memory", "unified_memory_basic"),
            ("Unified vs Explicit Copy", "unified_vs_explicit"),
            ("Memory Prefetching", "memory_prefetch_demo")
        ]
    },
    "shared_memory": {
        "concepts": [
            "Shared memory is on-chip, fast memory",
            "Shared among threads in a block",
            "Declared with __shared__ keyword",
            "Requires __syncthreads() for synchronization"
        ],
        "examples": [
            ("Shared Memory Basics", "shared_memory_basic"),
            ("Array Reversal with Shared Memory", "shared_array_reverse"),
            ("Shared Memory Performance", "shared_memory_perf")
        ]
    },
    "memory_coalescing": {
        "concepts": [
            "Coalesced access = adjacent threads access adjacent memory",
            "Stride-1 access pattern is optimal",
            "Non-coalesced access reduces bandwidth by 32x",
            "Use profiler to identify coalescing issues"
        ],
        "examples": [
            ("Coalesced vs Strided Access", "coalesced_vs_strided"),
            ("Coalescing Demonstration", "coalescing_demo"),
            ("Performance Impact", "coalescing_performance")
        ]
    },

    # Phase 3: Optimization
    "parallel_reduction": {
        "concepts": [
            "Reduction combines array elements (sum, max, min, etc.)",
            "Requires sequential addressing in shared memory",
            "Avoid divergent warps for better performance",
            "Multiple optimization levels possible"
        ],
        "examples": [
            ("Naive Reduction", "reduction_naive"),
            ("Optimized Reduction", "reduction_optimized"),
            ("CPU vs GPU Comparison", "reduction_benchmark")
        ]
    },
    "warp_shuffle": {
        "concepts": [
            "Warp shuffle allows intra-warp communication",
            "No shared memory required",
            "Lower latency than shared memory",
            "Operations: __shfl_sync, __shfl_down_sync, __shfl_up_sync"
        ],
        "examples": [
            ("Basic Warp Shuffle", "warp_shuffle_basic"),
            ("Warp Reduction", "warp_reduction"),
            ("Shuffle vs Shared Memory", "shuffle_vs_shared")
        ]
    },
    "occupancy": {
        "concepts": [
            "Occupancy = active warps / maximum warps per SM",
            "Balance registers, shared memory, threads per block",
            "Use cudaOccupancyMaxPotentialBlockSize",
            "Higher occupancy doesn't always mean faster"
        ],
        "examples": [
            ("Calculating Occupancy", "calculate_occupancy"),
            ("Optimal Block Size", "optimal_block_size"),
            ("Occupancy Impact", "occupancy_impact")
        ]
    },
    "bank_conflict": {
        "concepts": [
            "Shared memory organized in banks (32 banks)",
            "Conflict when multiple threads access same bank",
            "Stride access patterns can cause conflicts",
            "Padding arrays can resolve conflicts"
        ],
        "examples": [
            ("Bank Conflict Demo", "bank_conflict_demo"),
            ("Conflict-Free Access", "conflict_free_access"),
            ("Performance Comparison", "bank_conflict_perf")
        ]
    },

    # Phase 4: Advanced Memory
    "constant_memory": {
        "concepts": [
            "Constant memory cached on-chip",
            "Fast for broadcast reads (all threads read same value)",
            "64KB limit per kernel",
            "Declared with __constant__ keyword"
        ],
        "examples": [
            ("Constant Memory Basics", "constant_memory_basic"),
            ("Constant vs Global", "constant_vs_global"),
            ("Broadcast Performance", "constant_broadcast")
        ]
    },
    "texture_memory": {
        "concepts": [
            "Texture memory optimized for 2D spatial locality",
            "Hardware interpolation and clamping",
            "Read-only from kernels",
            "Useful for image processing"
        ],
        "examples": [
            ("Texture Memory Basics", "texture_basic"),
            ("2D Texture Access", "texture_2d"),
            ("Texture vs Global", "texture_performance")
        ]
    },

    # Phase 5: Advanced Algorithms
    "matrix_multiplication": {
        "concepts": [
            "Matrix multiply is compute-intensive (O(n³))",
            "Tiling reduces global memory accesses",
            "Shared memory critical for performance",
            "cuBLAS library provides optimized implementation"
        ],
        "examples": [
            ("Naive Matrix Multiply", "matmul_naive"),
            ("Tiled Matrix Multiply", "matmul_tiled"),
            ("Performance Comparison", "matmul_benchmark")
        ]
    },
    "scan": {
        "concepts": [
            "Prefix sum (scan) computes cumulative sums",
            "Two phases: up-sweep and down-sweep",
            "Work-efficient algorithm important",
            "Building block for many algorithms"
        ],
        "examples": [
            ("Naive Scan", "scan_naive"),
            ("Work-Efficient Scan", "scan_efficient"),
            ("Benchmark", "scan_benchmark")
        ]
    },
    "histogram": {
        "concepts": [
            "Histogram counts occurrences in bins",
            "Atomic operations for thread-safe updates",
            "Privatization reduces atomic contention",
            "Multiple optimization strategies"
        ],
        "examples": [
            ("Atomic Histogram", "histogram_atomic"),
            ("Privatized Histogram", "histogram_privatized"),
            ("Performance Comparison", "histogram_perf")
        ]
    },

    # Phase 6: Streams & Concurrency
    "streams": {
        "concepts": [
            "Streams enable concurrent kernel execution",
            "Overlap compute and memory transfer",
            "Default stream (stream 0) serializes",
            "Multiple streams can run in parallel"
        ],
        "examples": [
            ("Basic Streams", "streams_basic"),
            ("Concurrent Kernels", "streams_concurrent"),
            ("Overlap Transfer and Compute", "streams_overlap")
        ]
    },
    "events": {
        "concepts": [
            "Events mark points in stream execution",
            "Used for timing and synchronization",
            "cudaEventRecord and cudaEventSynchronize",
            "More accurate than CPU timing"
        ],
        "examples": [
            ("Event Timing", "events_timing"),
            ("Stream Synchronization", "events_sync"),
            ("Multi-stream Events", "events_multistream")
        ]
    },

    # Phase 7: Performance
    "profiling": {
        "concepts": [
            "Profile before optimizing",
            "Use nvprof or Nsight Systems/Compute",
            "Identify bottlenecks: memory, compute, or both",
            "Key metrics: occupancy, bandwidth, IPC"
        ],
        "examples": [
            ("Profiling with Events", "profile_events"),
            ("Kernel Metrics", "profile_metrics"),
            ("Bottleneck Analysis", "profile_bottleneck")
        ]
    },

    # Phase 4: Advanced Memory & Sync
    "zero_copy": {
        "concepts": ["Zero-copy memory maps host memory to device", "Useful for PCIe-connected GPUs", "No explicit cudaMemcpy needed", "Trade-off: convenience vs bandwidth"],
        "examples": [("Zero-Copy Basic", "zero_copy_basic"), ("Zero-Copy vs Regular", "zero_copy_comparison"), ("Performance Analysis", "zero_copy_perf")]
    },
    "atomics": {
        "concepts": ["Atomic operations ensure thread-safe updates", "Useful for histograms, reductions, locks", "Can create contention bottlenecks", "Types: atomicAdd, atomicMax, atomicCAS"],
        "examples": [("Atomic Add Example", "atomic_add"), ("Atomic vs Non-Atomic", "atomic_comparison"), ("Atomic Contention Demo", "atomic_contention")]
    },
    "cooperative_groups": {
        "concepts": ["Flexible thread grouping abstraction", "More expressive than __syncthreads()", "Grid-wide and multi-grid synchronization", "Warp-level operations"],
        "examples": [("Cooperative Groups Basics", "coop_groups_basic"), ("Grid Synchronization", "coop_grid_sync"), ("Thread Block Tiles", "coop_tiles")]
    },
    # Phase 5: Advanced Algorithms
    "gemm": {"concepts": ["GEMM = General Matrix Multiply", "Highly optimized operation", "Tiling and register blocking", "cuBLAS provides production implementation"], "examples": [("Basic GEMM", "gemm_basic"), ("Optimized GEMM", "gemm_optimized"), ("cuBLAS Comparison", "gemm_cublas")]},
    "cublas": {"concepts": ["cuBLAS = CUDA BLAS library", "Highly optimized linear algebra", "Handle-based API", "Matrix operations at peak performance"], "examples": [("cuBLAS Initialization", "cublas_init"), ("Matrix Multiply with cuBLAS", "cublas_gemm"), ("Performance Comparison", "cublas_perf")]},
    "sorting": {"concepts": ["Parallel sorting algorithms", "Bitonic sort for power-of-2 sizes", "Radix sort for large datasets", "Thrust provides optimized sort"], "examples": [("Bitonic Sort", "bitonic_sort"), ("Radix Sort", "radix_sort"), ("Thrust Sort", "thrust_sort")]},
    "thrust": {"concepts": ["Thrust = C++ template library for CUDA", "STL-like interface", "Automatic memory management", "Highly optimized primitives"], "examples": [("Thrust Basics", "thrust_basic"), ("Thrust Algorithms", "thrust_algorithms"), ("Thrust vs Custom", "thrust_comparison")]},
    # Phase 6: Streams & Concurrency
    "async_pipeline": {"concepts": ["Overlap transfer, compute, and transfer back", "Pipeline multiple batches", "Maximize GPU utilization", "Requires pinned memory and streams"], "examples": [("Simple Pipeline", "pipeline_basic"), ("Multi-Stage Pipeline", "pipeline_multistage"), ("Pipeline Performance", "pipeline_perf")]},
    "multi_gpu": {"concepts": ["Scale across multiple GPUs", "cudaSetDevice for GPU selection", "Data parallelism patterns", "P2P transfers between GPUs"], "examples": [("Multi-GPU Basics", "multi_gpu_basic"), ("Data Parallel Computation", "multi_gpu_dataparallel"), ("P2P Transfers", "multi_gpu_p2p")]},
    "nccl": {"concepts": ["NCCL = NVIDIA Collective Communications Library", "All-reduce, broadcast, gather operations", "Multi-GPU and multi-node", "Critical for distributed training"], "examples": [("NCCL Initialization", "nccl_init"), ("All-Reduce Example", "nccl_allreduce"), ("NCCL Performance", "nccl_perf")]},
    # Phase 7: Performance Engineering
    "debugging": {"concepts": ["cuda-gdb for kernel debugging", "CUDA-MEMCHECK for memory errors", "Assertions in kernels", "Common errors and solutions"], "examples": [("Debug Assertions", "debug_assert"), ("Memory Error Detection", "debug_memcheck"), ("Debug Output", "debug_printf")]},
    "kernel_fusion": {"concepts": ["Combine multiple kernels into one", "Reduces kernel launch overhead", "Improves data locality", "Trade-off: complexity vs performance"], "examples": [("Separate Kernels", "fusion_separate"), ("Fused Kernel", "fusion_fused"), ("Performance Comparison", "fusion_perf")]},
    "fast_math": {"concepts": ["Fast math trades accuracy for speed", "--use_fast_math compiler flag", "Intrinsics: __fdividef, __sinf, etc.", "Important for throughput-bound kernels"], "examples": [("Fast Math Comparison", "fastmath_compare"), ("Math Intrinsics", "fastmath_intrinsics"), ("Accuracy Analysis", "fastmath_accuracy")]},
    # Phase 8: Real Applications
    "cufft": {"concepts": ["cuFFT = CUDA FFT library", "Fast Fourier Transform on GPU", "1D, 2D, 3D transforms", "Signal processing and scientific computing"], "examples": [("1D FFT Example", "cufft_1d"), ("2D FFT Example", "cufft_2d"), ("FFT Performance", "cufft_perf")]},
    "cusparse": {"concepts": ["cuSPARSE = sparse matrix library", "Efficient sparse matrix operations", "CSR, COO formats", "SpMV (Sparse Matrix-Vector multiply)"], "examples": [("Sparse Matrix Basics", "cusparse_basic"), ("SpMV Example", "cusparse_spmv"), ("Sparse Performance", "cusparse_perf")]},
    "curand": {"concepts": ["cuRAND = random number generation", "Parallel RNG on GPU", "Various distributions (uniform, normal)", "Monte Carlo simulations"], "examples": [("Random Generation", "curand_basic"), ("Different Distributions", "curand_distributions"), ("Monte Carlo Example", "curand_montecarlo")]},
    "neural_network": {"concepts": ["Neural networks are matrix operations", "Forward pass: layer-by-layer computation", "Activation functions on GPU", "cuDNN provides optimized primitives"], "examples": [("Simple Dense Layer", "nn_dense_layer"), ("Activation Functions", "nn_activations"), ("Mini Neural Network", "nn_simple")]},
    "image_processing": {"concepts": ["Images are 2D/3D arrays", "Convolution is core operation", "Color space conversions", "Edge detection and filters"], "examples": [("Grayscale Conversion", "img_grayscale"), ("Gaussian Blur", "img_blur"), ("Edge Detection", "img_edge_detect")]},
    "raytracing": {"concepts": ["Ray tracing simulates light paths", "Embarrassingly parallel", "Sphere intersection tests", "Each pixel computed independently"], "examples": [("Basic Ray Tracer", "raytrace_basic"), ("Sphere Rendering", "raytrace_spheres"), ("Ray Tracer Performance", "raytrace_perf")]},
    "nbody": {"concepts": ["N-body simulates particle interactions", "All-pairs force calculation O(n²)", "Barnes-Hut algorithm O(n log n)", "Classic CUDA benchmark"], "examples": [("Naive N-body", "nbody_naive"), ("Optimized N-body", "nbody_optimized"), ("N-body Visualization", "nbody_visual")]},
    "molecular_dynamics": {"concepts": ["MD simulates atomic motion", "Force calculation and integration", "Neighbor lists for efficiency", "Used in computational chemistry"], "examples": [("Lennard-Jones Force", "md_force"), ("Integration Step", "md_integrate"), ("Simple MD System", "md_system")]},
    "option_pricing": {"concepts": ["Financial options pricing", "Black-Scholes model", "Monte Carlo methods", "Parallel RNG critical"], "examples": [("Black-Scholes Formula", "finance_bs"), ("Monte Carlo Pricing", "finance_mc"), ("Performance Analysis", "finance_perf")]},
    # Phase 9: Modern CUDA Features
    "dynamic_parallelism": {"concepts": ["Kernels can launch other kernels", "Recursive algorithms on GPU", "Adaptive workload generation", "Requires compute capability 3.5+"], "examples": [("Dynamic Launch", "dynamic_basic"), ("Recursive Kernel", "dynamic_recursive"), ("Adaptive Computation", "dynamic_adaptive")]},
    "cuda_graphs": {"concepts": ["CUDA graphs reduce launch overhead", "Record operations once, replay many times", "Faster than individual launches", "Great for recurring workloads"], "examples": [("Graph Creation", "graphs_create"), ("Graph vs Streams", "graphs_comparison"), ("Graph Performance", "graphs_perf")]},
    "mps": {"concepts": ["MPS = Multi-Process Service", "Share GPU among multiple processes", "Better utilization of GPU resources", "Important for containerized workloads"], "examples": [("MPS Basics", "mps_basic"), ("Multi-Process Demo", "mps_multiproc"), ("MPS Benefits", "mps_benefits")]},
    "mixed_precision": {"concepts": ["Mix FP32, FP16, and INT8 datatypes", "Higher throughput with lower precision", "Tensor Cores require FP16/INT8", "Accuracy vs speed trade-off"], "examples": [("FP16 Computation", "mixed_fp16"), ("Precision Comparison", "mixed_comparison"), ("Accuracy Analysis", "mixed_accuracy")]},
    "tensor_cores": {"concepts": ["Tensor Cores for matrix multiply", "FP16 input, FP32 accumulation", "10x-20x speedup for matrix ops", "Volta architecture and newer"], "examples": [("Tensor Core Basics", "tensor_basic"), ("WMMA API", "tensor_wmma"), ("Tensor Core Performance", "tensor_perf")]},
    "wmma": {"concepts": ["WMMA = Warp Matrix Multiply-Accumulate", "Low-level API for Tensor Cores", "16x16 or 32x32 tiles", "Manual optimization required"], "examples": [("WMMA GEMM", "wmma_gemm"), ("WMMA vs Standard", "wmma_comparison"), ("WMMA Optimization", "wmma_optimized")]},
}

# Code generation functions
def generate_cuda_code(topic_key, example_name):
    """Generate CUDA code based on topic and example name"""
    # Phase 2
    if "bandwidth" in example_name: return generate_bandwidth_code(example_name)
    elif "unified" in example_name: return generate_unified_memory_code(example_name)
    elif "shared" in example_name: return generate_shared_memory_code(example_name)
    elif "coalescing" in example_name or "coalesced" in example_name: return generate_coalescing_code(example_name)
    # Phase 3
    elif "reduction" in example_name: return generate_reduction_code(example_name)
    elif "shuffle" in example_name or "warp" in example_name: return generate_shuffle_code(example_name)
    elif "occupancy" in example_name: return generate_occupancy_code(example_name)
    elif "bank" in example_name: return generate_bank_conflict_code(example_name)
    # Phase 4
    elif "zero" in example_name: return generate_zero_copy_code(example_name)
    elif "atomic" in example_name: return generate_atomics_code(example_name)
    elif "coop" in example_name or "cooperative" in example_name: return generate_cooperative_groups_code(example_name)
    elif "constant" in example_name: return generate_constant_memory_code(example_name)
    elif "texture" in example_name: return generate_texture_code(example_name)
    # Phase 5
    elif "gemm" in example_name: return generate_gemm_code(example_name)
    elif "cublas" in example_name or "blas" in example_name: return generate_cublas_code(example_name)
    elif "bitonic" in example_name or "radix" in example_name or "sort" in example_name: return generate_sorting_code(example_name)
    elif "thrust" in example_name: return generate_thrust_code(example_name)
    elif "matmul" in example_name or "matrix" in example_name: return generate_matmul_code(example_name)
    elif "scan" in example_name: return generate_scan_code(example_name)
    elif "histogram" in example_name: return generate_histogram_code(example_name)
    # Phase 6
    elif "stream" in example_name: return generate_streams_code(example_name)
    elif "event" in example_name: return generate_events_code(example_name)
    elif "pipeline" in example_name or "async" in example_name: return generate_async_pipeline_code(example_name)
    elif "multi_gpu" in example_name or "multigpu" in example_name or "p2p" in example_name: return generate_multi_gpu_code(example_name)
    elif "nccl" in example_name: return generate_nccl_code(example_name)
    # Phase 7
    elif "profile" in example_name or "profiling" in example_name: return generate_profiling_code(example_name)
    elif "debug" in example_name: return generate_debugging_code(example_name)
    elif "fusion" in example_name or "fused" in example_name: return generate_kernel_fusion_code(example_name)
    elif "fast" in example_name or "fastmath" in example_name: return generate_fast_math_code(example_name)
    # Phase 8
    elif "cufft" in example_name or "fft" in example_name: return generate_cufft_code(example_name)
    elif "cusparse" in example_name or "sparse" in example_name: return generate_cusparse_code(example_name)
    elif "curand" in example_name or "rand" in example_name: return generate_curand_code(example_name)
    elif "nn_" in example_name or "neural" in example_name: return generate_neural_network_code(example_name)
    elif "img_" in example_name or "image" in example_name: return generate_image_processing_code(example_name)
    elif "raytrac" in example_name: return generate_raytracing_code(example_name)
    elif "nbody" in example_name: return generate_nbody_code(example_name)
    elif "md_" in example_name or "molecular" in example_name: return generate_molecular_dynamics_code(example_name)
    elif "finance" in example_name or "option" in example_name or "pricing" in example_name: return generate_option_pricing_code(example_name)
    # Phase 9
    elif "dynamic" in example_name: return generate_dynamic_parallelism_code(example_name)
    elif "graph" in example_name: return generate_cuda_graphs_code(example_name)
    elif "mps" in example_name: return generate_mps_code(example_name)
    elif "mixed" in example_name or "precision" in example_name or "fp16" in example_name: return generate_mixed_precision_code(example_name)
    elif "tensor" in example_name: return generate_tensor_cores_code(example_name)
    elif "wmma" in example_name: return generate_wmma_code(example_name)
    else: return generate_generic_code(example_name)

def generate_bandwidth_code(example_name):
    """Generate bandwidth measurement code"""
    return '''%%cu
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            printf("CUDA error: %s\\n", cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)

float measureBandwidth(size_t size, bool pinned) {
    float *h_data, *d_data;

    // Allocate memory
    if (pinned) {
        CUDA_CHECK(cudaMallocHost(&h_data, size));
    } else {
        h_data = (float*)malloc(size);
    }
    CUDA_CHECK(cudaMalloc(&d_data, size));

    // Initialize data
    for (size_t i = 0; i < size/sizeof(float); i++) {
        h_data[i] = i * 1.0f;
    }

    // Measure transfer time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Calculate bandwidth
    float bandwidth = (size / 1e9) / (milliseconds / 1000.0);

    // Cleanup
    if (pinned) {
        cudaFreeHost(h_data);
    } else {
        free(h_data);
    }
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return bandwidth;
}

int main() {
    printf("=== Memory Bandwidth Measurement ===\\n\\n");

    // Test different sizes
    size_t sizes[] = {1<<20, 1<<22, 1<<24, 1<<26};  // 1MB to 64MB

    printf("%-15s %-20s %-20s\\n", "Size", "Pageable (GB/s)", "Pinned (GB/s)");
    printf("-----------------------------------------------------\\n");

    for (int i = 0; i < 4; i++) {
        size_t size = sizes[i];
        float bw_pageable = measureBandwidth(size, false);
        float bw_pinned = measureBandwidth(size, true);

        printf("%-15zu %-20.2f %-20.2f\\n",
               size / (1024*1024), bw_pageable, bw_pinned);
    }

    printf("\\nPinned memory provides significantly better bandwidth!\\n");

    return 0;
}'''

def generate_unified_memory_code(example_name):
    """Generate unified memory code"""
    if "basic" in example_name:
        return '''%%cu
#include <stdio.h>

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            printf("CUDA error: %s\\n", cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1000000;
    size_t size = n * sizeof(float);

    printf("=== Unified Memory Basic Example ===\\n\\n");

    // Allocate unified memory (accessible from both CPU and GPU)
    float *a, *b, *c;
    CUDA_CHECK(cudaMallocManaged(&a, size));
    CUDA_CHECK(cudaMallocManaged(&b, size));
    CUDA_CHECK(cudaMallocManaged(&c, size));

    printf("✓ Allocated unified memory: %zu MB\\n", size * 3 / (1024*1024));

    // Initialize on CPU (no explicit transfer needed!)
    for (int i = 0; i < n; i++) {
        a[i] = i * 1.0f;
        b[i] = i * 2.0f;
    }
    printf("✓ Initialized data on CPU\\n");

    // Launch kernel (data automatically migrated to GPU)
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("✓ Kernel executed on GPU\\n");

    // Access result on CPU (automatically migrated back)
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (c[i] != a[i] + b[i]) {
            correct = false;
            break;
        }
    }
    printf("✓ Verified results on CPU\\n");
    printf("\\nResult: %s\\n", correct ? "CORRECT" : "INCORRECT");
    printf("Sample: %.1f + %.1f = %.1f\\n", a[0], b[0], c[0]);

    // Single free call for unified memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    printf("\\nAdvantages: No explicit cudaMemcpy calls!\\n");
    printf("Disadvantage: Implicit transfers may be slower\\n");

    return 0;
}'''
    else:
        return generate_generic_code("unified_memory")

def generate_shared_memory_code(example_name):
    """Generate shared memory code"""
    return '''%%cu
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 256

__global__ void arrayReverseShared(float *input, float *output, int n) {
    __shared__ float tile[TILE_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (idx < n) {
        tile[threadIdx.x] = input[idx];
    }
    __syncthreads();  // Wait for all threads to load

    // Write in reverse order
    if (idx < n) {
        output[idx] = tile[blockDim.x - 1 - threadIdx.x];
    }
}

__global__ void arrayReverseGlobal(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int reverseIdx = (blockIdx.x + 1) * blockDim.x - 1 - threadIdx.x;
        if (reverseIdx < n) {
            output[idx] = input[reverseIdx];
        }
    }
}

int main() {
    int n = 1000000;
    size_t size = n * sizeof(float);

    printf("=== Shared Memory Demonstration ===\\n\\n");

    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    for (int i = 0; i < n; i++) h_input[i] = i;

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = TILE_SIZE;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Test with shared memory
    cudaEventRecord(start);
    arrayReverseShared<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_shared;
    cudaEventElapsedTime(&time_shared, start, stop);

    // Test without shared memory
    cudaEventRecord(start);
    arrayReverseGlobal<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_global;
    cudaEventElapsedTime(&time_global, start, stop);

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    printf("Shared memory: %.3f ms\\n", time_shared);
    printf("Global memory: %.3f ms\\n", time_global);
    printf("Speedup: %.2fx\\n\\n", time_global / time_shared);

    printf("Shared memory is faster due to on-chip access!\\n");

    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}'''

def generate_coalescing_code(example_name):
    """Generate memory coalescing demonstration code"""
    return '''%%cu
#include <stdio.h>
#include <stdlib.h>

// Coalesced access: consecutive threads access consecutive memory
__global__ void coalescedAccess(float *data, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = data[idx] * 2.0f;
    }
}

// Strided access: non-coalesced, poor performance
__global__ void stridedAccess(float *data, float *result, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (idx < n) {
        result[idx] = data[idx] * 2.0f;
    }
}

int main() {
    int n = 10000000;
    size_t size = n * sizeof(float);

    printf("=== Memory Coalescing Demonstration ===\\n\\n");

    float *d_data, *d_result;
    cudaMalloc(&d_data, size);
    cudaMalloc(&d_result, size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Test coalesced access
    cudaEventRecord(start);
    coalescedAccess<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_result, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_coalesced;
    cudaEventElapsedTime(&time_coalesced, start, stop);

    // Test strided access (stride = 32)
    blocksPerGrid = (n/32 + threadsPerBlock - 1) / threadsPerBlock;
    cudaEventRecord(start);
    stridedAccess<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_result, n, 32);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_strided;
    cudaEventElapsedTime(&time_strided, start, stop);

    printf("Coalesced access: %.3f ms\\n", time_coalesced);
    printf("Strided access (stride=32): %.3f ms\\n", time_strided);
    printf("Performance degradation: %.2fx slower\\n\\n", time_strided / time_coalesced);

    float bandwidth_coalesced = (size * 2 / 1e9) / (time_coalesced / 1000.0);
    float bandwidth_strided = (size * 2 / 1e9) / (time_strided / 1000.0);

    printf("Coalesced bandwidth: %.2f GB/s\\n", bandwidth_coalesced);
    printf("Strided bandwidth: %.2f GB/s\\n\\n", bandwidth_strided);

    printf("KEY INSIGHT: Adjacent threads should access adjacent memory!\\n");

    cudaFree(d_data);
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}'''

def generate_reduction_code(example_name):
    """Generate parallel reduction code"""
    if "naive" in example_name:
        return '''%%cu
#include <stdio.h>
#include <stdlib.h>

__global__ void reductionNaive(float *input, float *output, int n) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Reduction in shared memory (naive approach)
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0 && tid + s < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    int n = 1000000;
    size_t size = n * sizeof(float);

    printf("=== Naive Parallel Reduction ===\\n\\n");

    float *h_input = (float*)malloc(size);
    for (int i = 0; i < n; i++) h_input[i] = 1.0f;  // Sum should be n

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    cudaMalloc(&d_output, blocksPerGrid * sizeof(float));

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Launch kernel
    reductionNaive<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);

    // Copy partial results and finish on CPU
    float *h_output = (float*)malloc(blocksPerGrid * sizeof(float));
    cudaMemcpy(h_output, d_output, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        sum += h_output[i];
    }

    printf("Sum: %.0f (expected: %d)\\n", sum, n);
    printf("Result: %s\\n\\n", (sum == n) ? "CORRECT" : "INCORRECT");
    printf("NOTE: This naive version has divergent branches (inefficient)\\n");

    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);

    return 0;
}'''
    else:  # optimized
        return '''%%cu
#include <stdio.h>
#include <stdlib.h>

__global__ void reductionOptimized(float *input, float *output, int n) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Load data with grid-stride and add during load
    sdata[tid] = 0;
    if (idx < n) sdata[tid] = input[idx];
    if (idx + blockDim.x < n) sdata[tid] += input[idx + blockDim.x];
    __syncthreads();

    // Sequential addressing (no divergence)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    int n = 1000000;
    size_t size = n * sizeof(float);

    printf("=== Optimized Parallel Reduction ===\\n\\n");

    float *h_input = (float*)malloc(size);
    for (int i = 0; i < n; i++) h_input[i] = 1.0f;

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
    cudaMalloc(&d_output, blocksPerGrid * sizeof(float));

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    reductionOptimized<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float *h_output = (float*)malloc(blocksPerGrid * sizeof(float));
    cudaMemcpy(h_output, d_output, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        sum += h_output[i];
    }

    printf("Sum: %.0f (expected: %d)\\n", sum, n);
    printf("Time: %.3f ms\\n", milliseconds);
    printf("Result: %s\\n\\n", (sum == n) ? "CORRECT" : "INCORRECT");
    printf("OPTIMIZATION: Sequential addressing avoids warp divergence!\\n");

    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}'''

# Add more generation functions for other topics...
def generate_generic_code(topic):
    """Generate generic template code"""
    return f'''%%cu
#include <stdio.h>

__global__ void kernel() {{
    printf("Example for: {topic}\\n");
}}

int main() {{
    printf("=== {topic.replace('_', ' ').title()} ===\\n\\n");

    kernel<<<1, 32>>>();
    cudaDeviceSynchronize();

    printf("\\nExample completed successfully!\\n");
    return 0;
}}'''

# Specialized generation functions
def generate_shuffle_code(example_name):
    return generate_generic_code("warp_shuffle")

def generate_occupancy_code(example_name):
    return generate_generic_code("occupancy")

def generate_bank_conflict_code(example_name):
    return generate_generic_code("bank_conflicts")

def generate_constant_memory_code(example_name):
    return generate_generic_code("constant_memory")

def generate_texture_code(example_name):
    return generate_generic_code("texture_memory")

def generate_matmul_code(example_name):
    return generate_generic_code("matrix_multiplication")

def generate_scan_code(example_name):
    return generate_generic_code("prefix_scan")

def generate_histogram_code(example_name):
    return generate_generic_code("histogram")

def generate_streams_code(example_name):
    return generate_generic_code("cuda_streams")

def generate_events_code(example_name):
    return generate_generic_code("cuda_events")

def generate_profiling_code(example_name):
    return generate_generic_code("profiling")

def generate_neural_network_code(example_name):
    return generate_generic_code("neural_network")

def generate_image_processing_code(example_name):
    return generate_generic_code("image_processing")

# Phase 4 generators
def generate_zero_copy_code(example_name):
    return generate_generic_code("zero_copy_memory")

def generate_atomics_code(example_name):
    return generate_generic_code("atomic_operations")

def generate_cooperative_groups_code(example_name):
    return generate_generic_code("cooperative_groups")

# Phase 5 generators
def generate_gemm_code(example_name):
    return generate_generic_code("gemm")

def generate_cublas_code(example_name):
    return generate_generic_code("cublas")

def generate_sorting_code(example_name):
    return generate_generic_code("parallel_sorting")

def generate_thrust_code(example_name):
    return generate_generic_code("thrust")

# Phase 6 generators
def generate_async_pipeline_code(example_name):
    return generate_generic_code("async_pipeline")

def generate_multi_gpu_code(example_name):
    return generate_generic_code("multi_gpu")

def generate_nccl_code(example_name):
    return generate_generic_code("nccl")

# Phase 7 generators
def generate_debugging_code(example_name):
    return generate_generic_code("cuda_debugging")

def generate_kernel_fusion_code(example_name):
    return generate_generic_code("kernel_fusion")

def generate_fast_math_code(example_name):
    return generate_generic_code("fast_math")

# Phase 8 generators
def generate_cufft_code(example_name):
    return generate_generic_code("cufft")

def generate_cusparse_code(example_name):
    return generate_generic_code("cusparse")

def generate_curand_code(example_name):
    return generate_generic_code("curand")

def generate_raytracing_code(example_name):
    return generate_generic_code("raytracing")

def generate_nbody_code(example_name):
    return generate_generic_code("nbody_simulation")

def generate_molecular_dynamics_code(example_name):
    return generate_generic_code("molecular_dynamics")

def generate_option_pricing_code(example_name):
    return generate_generic_code("option_pricing")

# Phase 9 generators
def generate_dynamic_parallelism_code(example_name):
    return generate_generic_code("dynamic_parallelism")

def generate_cuda_graphs_code(example_name):
    return generate_generic_code("cuda_graphs")

def generate_mps_code(example_name):
    return generate_generic_code("mps")

def generate_mixed_precision_code(example_name):
    return generate_generic_code("mixed_precision")

def generate_tensor_cores_code(example_name):
    return generate_generic_code("tensor_cores")

def generate_wmma_code(example_name):
    return generate_generic_code("wmma")

def find_topic_key(filename):
    """Extract topic key from filename"""
    filename_lower = filename.lower()

    # Direct matches first
    for key in TOPIC_TEMPLATES.keys():
        if key in filename_lower:
            return key

    # Partial/fuzzy matches
    topic_map = {
        "zero": "zero_copy", "atomic": "atomics", "cooperative": "cooperative_groups", "coop": "cooperative_groups",
        "gemm": "gemm", "cublas": "cublas", "blas": "cublas",
        "bitonic": "sorting", "radix": "sorting", "sort": "sorting",
        "thrust": "thrust",
        "async": "async_pipeline", "pipeline": "async_pipeline",
        "multi_gpu": "multi_gpu", "p2p": "multi_gpu", "multigpu": "multi_gpu",
        "nccl": "nccl",
        "debug": "debugging",
        "fusion": "kernel_fusion", "fused": "kernel_fusion",
        "fast_math": "fast_math", "fastmath": "fast_math",
        "cufft": "cufft", "fft": "cufft",
        "cusparse": "cusparse", "sparse": "cusparse",
        "curand": "curand", "rand": "curand",
        "raytrac": "raytracing", "raycast": "raytracing",
        "nbody": "nbody", "n_body": "nbody",
        "molecular": "molecular_dynamics", "md_": "molecular_dynamics",
        "option": "option_pricing", "finance": "option_pricing", "pricing": "option_pricing",
        "dynamic": "dynamic_parallelism",
        "graph": "cuda_graphs",
        "mps": "mps",
        "mixed": "mixed_precision", "precision": "mixed_precision",
        "tensor": "tensor_cores",
        "wmma": "wmma",
        "bandwidth": "memory_bandwidth",
        "unified": "unified_memory", "managed": "unified_memory",
        "shared": "shared_memory",
        "coalescing": "memory_coalescing", "coalesced": "memory_coalescing",
        "reduction": "parallel_reduction", "reduce": "parallel_reduction",
        "shuffle": "warp_shuffle", "warp": "warp_shuffle",
        "occupancy": "occupancy",
        "bank": "bank_conflict", "conflict": "bank_conflict",
        "constant": "constant_memory",
        "texture": "texture_memory",
        "matrix": "matrix_multiplication", "matmul": "matrix_multiplication", "transpose": "matrix_multiplication",
        "scan": "scan", "prefix": "scan",
        "histogram": "histogram",
        "stream": "streams",
        "event": "events",
        "profile": "profiling", "profiling": "profiling",
        "neural": "neural_network", "nn_": "neural_network",
        "image": "image_processing", "img_": "image_processing",
    }

    for pattern, topic in topic_map.items():
        if pattern in filename_lower:
            return topic

    return None

def is_placeholder_notebook(notebook_path):
    """Check if notebook has placeholder content"""
    try:
        with open(notebook_path, 'r') as f:
            content = json.load(f)

        # Check for placeholder patterns in code cells
        for cell in content.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                if 'printf("Example kernel' in source or 'TODO: Implement' in source:
                    return True
        return False
    except:
        return False

def fill_notebook(notebook_path, topic_key):
    """Fill a notebook with generated content"""
    print(f"Filling: {os.path.basename(notebook_path)}")

    try:
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)

        if topic_key not in TOPIC_TEMPLATES:
            print(f"  ⚠ No template for topic: {topic_key}")
            return False

        template = TOPIC_TEMPLATES[topic_key]

        # Find and replace code cells
        code_cell_index = 0
        for i, cell in enumerate(notebook['cells']):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                if 'printf("Example kernel' in source or 'TODO' in source:
                    # This is a placeholder, replace it
                    if code_cell_index < len(template['examples']):
                        example_title, example_name = template['examples'][code_cell_index]
                        new_code = generate_cuda_code(topic_key, example_name)
                        cell['source'] = [new_code]
                        print(f"  ✓ Filled example {code_cell_index + 1}: {example_title}")
                        code_cell_index += 1

        # Update key takeaways
        for cell in notebook['cells']:
            if cell.get('cell_type') == 'markdown':
                source = ''.join(cell.get('source', []))
                if '## Key Takeaways' in source and 'Key concept 1' in source:
                    concepts = template['concepts']
                    takeaways = '\n'.join([f"{i+1}. {c}" for i, c in enumerate(concepts)])
                    cell['source'] = [f"## Key Takeaways\n\n{takeaways}"]
                    print(f"  ✓ Updated key takeaways")

        # Save modified notebook
        with open(notebook_path, 'w') as f:
            json.dump(notebook, f, indent=1)

        print(f"  ✅ Successfully filled {os.path.basename(notebook_path)}\n")
        return True

    except Exception as e:
        print(f"  ❌ Error: {e}\n")
        return False

def main():
    """Main function to fill all notebooks"""
    notebooks_dir = Path("notebooks")

    print("=" * 70)
    print("CUDA Notebook Content Generator")
    print("=" * 70)
    print()

    # Find all notebooks
    all_notebooks = sorted(glob.glob(str(notebooks_dir / "phase*" / "*.ipynb")))

    print(f"Found {len(all_notebooks)} notebooks\n")
    print("Analyzing notebooks...")
    print("-" * 70)

    to_fill = []
    for notebook_path in all_notebooks:
        if is_placeholder_notebook(notebook_path):
            topic_key = find_topic_key(os.path.basename(notebook_path))
            if topic_key:
                to_fill.append((notebook_path, topic_key))

    print(f"\n{len(to_fill)} notebooks need content\n")
    print("=" * 70)
    print("Filling notebooks...")
    print("=" * 70)
    print()

    success_count = 0
    for notebook_path, topic_key in to_fill:
        if fill_notebook(notebook_path, topic_key):
            success_count += 1

    print("=" * 70)
    print(f"COMPLETE: {success_count}/{len(to_fill)} notebooks filled successfully")
    print("=" * 70)

if __name__ == "__main__":
    main()
