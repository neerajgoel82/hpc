#!/usr/bin/env python3
"""
Script to generate all CUDA learning curriculum Jupyter notebooks
Based on CUDA_LEARNING_CURRICULUM.md
"""

import json
import os

# Notebook metadata template
METADATA = {
    "accelerator": "GPU",
    "colab": {
        "gpuType": "T4",
        "provenance": []
    },
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.8.10"
    }
}

# Define all notebooks with their content
NOTEBOOKS = {
    # Phase 2: Memory Management
    "phase2/06_memory_basics.ipynb": {
        "title": "Memory Basics and Data Transfer",
        "phase": "Phase 2: Memory Management",
        "objectives": [
            "Master CUDA memory allocation and deallocation",
            "Understand different memory transfer patterns",
            "Learn about pinned (page-locked) memory",
            "Measure memory transfer bandwidth",
            "Optimize host-device data movement"
        ],
        "concepts": """## Concept: CUDA Memory Model

**Memory Types:**
- **Global Memory**: Large, slow, accessible by all threads
- **Pinned Memory**: Non-pageable host memory for faster transfers
- **Device Memory**: GPU VRAM

**Memory Functions:**
```cuda
cudaMalloc(&d_ptr, size);           // Allocate device memory
cudaFree(d_ptr);                    // Free device memory
cudaMemcpy(dst, src, size, kind);   // Copy memory
cudaMallocHost(&h_ptr, size);       // Allocate pinned memory
cudaFreeHost(h_ptr);                // Free pinned memory
```

**Transfer Bandwidth:**
- Pinned memory: ~12 GB/s (PCIe 3.0 x16)
- Pageable memory: ~6 GB/s
- Async transfers possible with pinned memory"""
    },

    "phase2/07_bandwidth_test.ipynb": {
        "title": "Memory Bandwidth Benchmarking",
        "phase": "Phase 2: Memory Management",
        "objectives": [
            "Measure host-to-device transfer bandwidth",
            "Compare pinned vs pageable memory performance",
            "Understand PCIe bandwidth limitations",
            "Profile memory access patterns",
            "Optimize data transfer strategies"
        ],
        "concepts": """## Concept: Memory Bandwidth

**Theoretical Bandwidth:**
- PCIe 3.0 x16: 15.75 GB/s per direction
- PCIe 4.0 x16: 31.5 GB/s per direction

**Factors Affecting Bandwidth:**
- Transfer size (larger is better)
- Memory type (pinned vs pageable)
- Transfer direction (H2D, D2H, D2D)
- Concurrent operations

**Bandwidth Calculation:**
```
Bandwidth = DataSize / Time
```"""
    },

    "phase2/08_unified_memory.ipynb": {
        "title": "Unified Memory and Managed Memory",
        "phase": "Phase 2: Memory Management",
        "objectives": [
            "Understand Unified Memory concept",
            "Use cudaMallocManaged for simpler code",
            "Learn about page faults and migration",
            "Understand performance implications",
            "Use prefetching for optimization"
        ],
        "concepts": """## Concept: Unified Memory

**Unified Memory:**
- Single pointer accessible from CPU and GPU
- Automatic data migration
- Simpler code, but requires understanding

**Functions:**
```cuda
cudaMallocManaged(&ptr, size);      // Allocate managed memory
cudaMemPrefetchAsync(ptr, size, device);  // Prefetch data
cudaMemAdvise(ptr, size, advice, device); // Give hints
```

**Benefits:**
- Simplified memory management
- Automatic migration
- Oversubscription support

**Considerations:**
- Page fault overhead
- Migration costs
- Requires compute capability 6.0+"""
    },

    "phase2/09_shared_memory_basics.ipynb": {
        "title": "Shared Memory Basics",
        "phase": "Phase 2: Memory Management - Shared Memory",
        "objectives": [
            "Understand shared memory architecture",
            "Declare and use shared memory",
            "Implement basic tiling with shared memory",
            "Learn synchronization with __syncthreads()",
            "Understand shared memory scope"
        ],
        "concepts": """## Concept: Shared Memory

**What is Shared Memory?**
- Fast on-chip memory (similar to L1 cache)
- Shared among threads in a block
- ~100x faster than global memory
- Limited size (48-96 KB per SM)

**Declaration:**
```cuda
__shared__ float sharedData[256];  // Static allocation
extern __shared__ float sharedData[];  // Dynamic allocation
```

**Key Points:**
- Requires __syncthreads() for synchronization
- Per-block scope
- Bank conflicts can reduce performance
- Ideal for data reuse within block"""
    },

    "phase2/10_matrix_multiply_tiled.ipynb": {
        "title": "Tiled Matrix Multiplication",
        "phase": "Phase 2: Memory Management - Shared Memory",
        "objectives": [
            "Implement tiled matrix multiplication",
            "Use shared memory for optimization",
            "Understand blocking techniques",
            "Reduce global memory accesses",
            "Measure performance improvements"
        ],
        "concepts": """## Concept: Tiled Matrix Multiplication

**Tiling Strategy:**
- Load tiles from global to shared memory
- Perform computations on tiles
- Reduce global memory bandwidth

**Algorithm:**
1. Each block loads TILE_SIZE x TILE_SIZE elements
2. Compute partial products using shared memory
3. Accumulate results
4. Repeat for all tiles

**Performance:**
- Reduces global memory traffic
- Increases arithmetic intensity
- Typical speedup: 5-10x over naive implementation"""
    },

    "phase2/11_coalescing_demo.ipynb": {
        "title": "Memory Coalescing Demonstration",
        "phase": "Phase 2: Memory Management - Shared Memory",
        "objectives": [
            "Understand memory coalescing concept",
            "Identify coalesced vs uncoalesced access",
            "Measure performance impact",
            "Learn access pattern optimization",
            "Fix common coalescing issues"
        ],
        "concepts": """## Concept: Memory Coalescing

**Coalesced Access:**
- Consecutive threads access consecutive memory
- Single memory transaction for warp
- Maximum bandwidth utilization

**Uncoalesced Access:**
- Random or strided access patterns
- Multiple memory transactions
- Reduced bandwidth

**Rules for Coalescing:**
- Threads in warp access consecutive addresses
- Aligned to segment size (32, 64, 128 bytes)
- Within same cache line

**Performance Impact:**
- Coalesced: ~300 GB/s
- Uncoalesced: ~30 GB/s (10x slower)"""
    }
}

# Continue with more notebooks...
NOTEBOOKS_PHASE3_TO_9 = {
    # Phase 3: Optimization Fundamentals
    "phase3/12_warp_divergence.ipynb": {
        "title": "Warp Divergence and Branch Efficiency",
        "phase": "Phase 3: Optimization Fundamentals",
        "topics": ["warp execution", "branch divergence", "performance impact"]
    },
    "phase3/13_warp_shuffle.ipynb": {
        "title": "Warp-Level Primitives and Shuffle Operations",
        "phase": "Phase 3: Optimization Fundamentals",
        "topics": ["__shfl operations", "warp-level communication", "reduction"]
    },
    "phase3/14_occupancy_tuning.ipynb": {
        "title": "Occupancy Optimization",
        "phase": "Phase 3: Optimization Fundamentals",
        "topics": ["occupancy calculation", "resource usage", "performance tuning"]
    },
    "phase3/15_parallel_reduction.ipynb": {
        "title": "Parallel Reduction Algorithms",
        "phase": "Phase 3: Optimization Fundamentals",
        "topics": ["reduction tree", "shared memory reduction", "warp reduction"]
    },
    "phase3/16_prefix_sum.ipynb": {
        "title": "Prefix Sum (Scan) Algorithms",
        "phase": "Phase 3: Optimization Fundamentals",
        "topics": ["inclusive scan", "exclusive scan", "work-efficient scan"]
    },
    "phase3/17_histogram.ipynb": {
        "title": "Histogram with Atomic Operations",
        "phase": "Phase 3: Optimization Fundamentals",
        "topics": ["atomic operations", "histogram computation", "privatization"]
    },

    # Phase 4: Advanced Memory & Synchronization
    "phase4/18_texture_memory.ipynb": {
        "title": "Texture Memory and Caching",
        "phase": "Phase 4: Advanced Memory & Synchronization",
        "topics": ["texture memory", "texture objects", "image filtering"]
    },
    "phase4/19_constant_memory.ipynb": {
        "title": "Constant Memory Usage",
        "phase": "Phase 4: Advanced Memory & Synchronization",
        "topics": ["constant memory", "read-only cache", "coefficients"]
    },
    "phase4/20_zero_copy.ipynb": {
        "title": "Zero-Copy Memory",
        "phase": "Phase 4: Advanced Memory & Synchronization",
        "topics": ["zero-copy", "mapped memory", "direct access"]
    },
    "phase4/21_atomics.ipynb": {
        "title": "Atomic Operations Patterns",
        "phase": "Phase 4: Advanced Memory & Synchronization",
        "topics": ["atomic operations", "atomicAdd", "atomicCAS", "thread-safety"]
    },
    "phase4/22_cooperative_groups.ipynb": {
        "title": "Cooperative Groups API",
        "phase": "Phase 4: Advanced Memory & Synchronization",
        "topics": ["cooperative groups", "flexible synchronization", "grid sync"]
    },
    "phase4/23_multi_kernel_sync.ipynb": {
        "title": "Multi-Kernel Synchronization",
        "phase": "Phase 4: Advanced Memory & Synchronization",
        "topics": ["kernel dependencies", "streams", "events"]
    },

    # Phase 5: Advanced Algorithms
    "phase5/24_gemm_optimized.ipynb": {
        "title": "Highly Optimized GEMM",
        "phase": "Phase 5: Advanced Algorithms",
        "topics": ["matrix multiply", "register tiling", "optimization"]
    },
    "phase5/25_cublas_integration.ipynb": {
        "title": "cuBLAS Library Integration",
        "phase": "Phase 5: Advanced Algorithms",
        "topics": ["cuBLAS", "BLAS operations", "library usage"]
    },
    "phase5/26_matrix_transpose.ipynb": {
        "title": "Efficient Matrix Transpose",
        "phase": "Phase 5: Advanced Algorithms",
        "topics": ["transpose", "shared memory", "bank conflicts"]
    },
    "phase5/27_bitonic_sort.ipynb": {
        "title": "Bitonic Sort Algorithm",
        "phase": "Phase 5: Advanced Algorithms",
        "topics": ["parallel sorting", "bitonic sort", "comparison networks"]
    },
    "phase5/28_radix_sort.ipynb": {
        "title": "Radix Sort Implementation",
        "phase": "Phase 5: Advanced Algorithms",
        "topics": ["radix sort", "digit extraction", "parallel sort"]
    },
    "phase5/29_thrust_examples.ipynb": {
        "title": "Thrust Library Examples",
        "phase": "Phase 5: Advanced Algorithms",
        "topics": ["Thrust", "STL-like", "high-level algorithms"]
    },

    # Phase 6: Streams & Concurrency
    "phase6/30_streams_basic.ipynb": {
        "title": "CUDA Streams Basics",
        "phase": "Phase 6: Streams & Concurrency",
        "topics": ["streams", "async execution", "concurrency"]
    },
    "phase6/31_async_pipeline.ipynb": {
        "title": "Asynchronous Pipeline",
        "phase": "Phase 6: Streams & Concurrency",
        "topics": ["overlapping", "pipeline", "compute and transfer"]
    },
    "phase6/32_events_timing.ipynb": {
        "title": "Events and Performance Timing",
        "phase": "Phase 6: Streams & Concurrency",
        "topics": ["cuda events", "timing", "synchronization"]
    },
    "phase6/33_multi_gpu_basic.ipynb": {
        "title": "Multi-GPU Programming Basics",
        "phase": "Phase 6: Streams & Concurrency",
        "topics": ["multi-GPU", "device management", "load balancing"]
    },
    "phase6/34_p2p_transfer.ipynb": {
        "title": "Peer-to-Peer Memory Transfer",
        "phase": "Phase 6: Streams & Concurrency",
        "topics": ["P2P", "GPU Direct", "inter-GPU transfer"]
    },
    "phase6/35_nccl_collectives.ipynb": {
        "title": "NCCL Multi-GPU Communication",
        "phase": "Phase 6: Streams & Concurrency",
        "topics": ["NCCL", "collectives", "multi-GPU comm"]
    },

    # Phase 7: Performance Engineering
    "phase7/36_profiling_demo.ipynb": {
        "title": "Profiling with Nsight Tools",
        "phase": "Phase 7: Performance Engineering",
        "topics": ["profiling", "Nsight Compute", "bottlenecks"]
    },
    "phase7/37_debugging_cuda.ipynb": {
        "title": "CUDA Debugging Techniques",
        "phase": "Phase 7: Performance Engineering",
        "topics": ["debugging", "cuda-memcheck", "common errors"]
    },
    "phase7/38_kernel_fusion.ipynb": {
        "title": "Kernel Fusion Optimization",
        "phase": "Phase 7: Performance Engineering",
        "topics": ["kernel fusion", "combining operations", "overhead reduction"]
    },
    "phase7/39_fast_math.ipynb": {
        "title": "Fast Math Operations",
        "phase": "Phase 7: Performance Engineering",
        "topics": ["fast math", "precision tradeoffs", "intrinsics"]
    },
    "phase7/40_advanced_optimization.ipynb": {
        "title": "Advanced Optimization Techniques",
        "phase": "Phase 7: Performance Engineering",
        "topics": ["ILP", "loop unrolling", "PTX"]
    },

    # Phase 8: Real-World Applications
    "phase8/41_cufft_demo.ipynb": {
        "title": "cuFFT for Fourier Transforms",
        "phase": "Phase 8: Real-World Applications",
        "topics": ["cuFFT", "FFT", "frequency domain"]
    },
    "phase8/42_cusparse_demo.ipynb": {
        "title": "cuSPARSE for Sparse Matrices",
        "phase": "Phase 8: Real-World Applications",
        "topics": ["cuSPARSE", "sparse matrices", "CSR format"]
    },
    "phase8/43_curand_demo.ipynb": {
        "title": "cuRAND for Random Numbers",
        "phase": "Phase 8: Real-World Applications",
        "topics": ["cuRAND", "random generation", "Monte Carlo"]
    },
    "phase8/44_image_processing.ipynb": {
        "title": "Image Processing Pipeline",
        "phase": "Phase 8: Real-World Applications",
        "topics": ["image processing", "filters", "convolution"]
    },
    "phase8/45_raytracer.ipynb": {
        "title": "GPU Ray Tracing",
        "phase": "Phase 8: Real-World Applications",
        "topics": ["ray tracing", "rendering", "graphics"]
    },
    "phase8/46_nbody_simulation.ipynb": {
        "title": "N-Body Physics Simulation",
        "phase": "Phase 8: Real-World Applications",
        "topics": ["n-body", "physics", "simulation"]
    },
    "phase8/47_neural_network.ipynb": {
        "title": "Neural Network from Scratch",
        "phase": "Phase 8: Real-World Applications",
        "topics": ["neural network", "backpropagation", "deep learning"]
    },
    "phase8/48_molecular_dynamics.ipynb": {
        "title": "Molecular Dynamics Simulation",
        "phase": "Phase 8: Real-World Applications",
        "topics": ["molecular dynamics", "force calculation", "MD"]
    },
    "phase8/49_option_pricing.ipynb": {
        "title": "Financial Option Pricing",
        "phase": "Phase 8: Real-World Applications",
        "topics": ["Monte Carlo", "option pricing", "finance"]
    },

    # Phase 9: Advanced Topics
    "phase9/50_dynamic_parallelism.ipynb": {
        "title": "Dynamic Parallelism",
        "phase": "Phase 9: Advanced Topics",
        "topics": ["dynamic parallelism", "nested kernels", "recursion"]
    },
    "phase9/51_cuda_graphs.ipynb": {
        "title": "CUDA Graphs",
        "phase": "Phase 9: Advanced Topics",
        "topics": ["CUDA graphs", "graph capture", "optimization"]
    },
    "phase9/52_mps_demo.ipynb": {
        "title": "Multi-Process Service (MPS)",
        "phase": "Phase 9: Advanced Topics",
        "topics": ["MPS", "multi-process", "GPU sharing"]
    },
    "phase9/53_mixed_precision.ipynb": {
        "title": "Mixed Precision Computing",
        "phase": "Phase 9: Advanced Topics",
        "topics": ["mixed precision", "FP16", "FP32", "accuracy"]
    },
    "phase9/54_tensor_cores.ipynb": {
        "title": "Tensor Core Programming",
        "phase": "Phase 9: Advanced Topics",
        "topics": ["tensor cores", "matrix cores", "deep learning"]
    },
    "phase9/55_wmma_gemm.ipynb": {
        "title": "WMMA Matrix Multiply",
        "phase": "Phase 9: Advanced Topics",
        "topics": ["WMMA", "warp matrix", "tensor cores"]
    }
}

def create_markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.split('\n')
    }

def create_code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.split('\n')
    }

def generate_notebook(notebook_num, title, phase, objectives, concepts, examples=None):
    """Generate a complete notebook structure"""
    cells = []

    # Title cell
    cells.append(create_markdown_cell(f"""# Notebook {notebook_num:02d}: {title}
## {phase}

**Learning Objectives:**
""" + "\n".join(f"- {obj}" for obj in objectives)))

    # Concepts cell
    cells.append(create_markdown_cell(concepts))

    # Add example code cells
    if not examples:
        examples = [
            {
                "title": f"Example 1: Basic {title}",
                "code": """%%cu
#include <stdio.h>

__global__ void kernel() {
    printf("Example kernel\\n");
}

int main() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}"""
            }
        ]

    for i, example in enumerate(examples, 1):
        cells.append(create_markdown_cell(f"## {example['title']}"))
        cells.append(create_code_cell(example['code']))

    # Exercise cell
    cells.append(create_markdown_cell("""## Practical Exercise

Complete the following exercises to practice the concepts learned."""))

    cells.append(create_code_cell("""%%cu
// Your solution here
#include <stdio.h>

int main() {
    // TODO: Implement your solution
    return 0;
}"""))

    # Key takeaways
    cells.append(create_markdown_cell("""## Key Takeaways

1. Key concept 1
2. Key concept 2
3. Key concept 3"""))

    # Next steps
    next_num = notebook_num + 1
    cells.append(create_markdown_cell(f"""## Next Steps

Continue to: **{next_num:02d}_next_topic.ipynb**"""))

    # Notes section
    cells.append(create_markdown_cell("""## Notes

*Use this space to write your own notes and observations:*

---



---"""))

    return {
        "cells": cells,
        "metadata": METADATA,
        "nbformat": 4,
        "nbformat_minor": 0
    }

def main():
    base_dir = "/Users/negoel/code/mywork/github/neerajgoel82/cuda-samples/colab/notebooks"

    # Generate detailed Phase 2 notebooks
    detailed_notebooks = [
        (6, "phase2", NOTEBOOKS["phase2/06_memory_basics.ipynb"]),
        (7, "phase2", NOTEBOOKS["phase2/07_bandwidth_test.ipynb"]),
        (8, "phase2", NOTEBOOKS["phase2/08_unified_memory.ipynb"]),
        (9, "phase2", NOTEBOOKS["phase2/09_shared_memory_basics.ipynb"]),
        (10, "phase2", NOTEBOOKS["phase2/10_matrix_multiply_tiled.ipynb"]),
        (11, "phase2", NOTEBOOKS["phase2/11_coalescing_demo.ipynb"]),
    ]

    for num, phase, info in detailed_notebooks:
        notebook = generate_notebook(
            num,
            info["title"],
            info["phase"],
            info["objectives"],
            info["concepts"]
        )

        filepath = os.path.join(base_dir, phase, f"{num:02d}_{info['title'].lower().replace(' ', '_').replace('-', '_')}.ipynb")
        with open(filepath, 'w') as f:
            json.dump(notebook, f, indent=1)
        print(f"Created {filepath}")

    # Generate remaining notebooks (phases 3-9)
    for path, info in NOTEBOOKS_PHASE3_TO_9.items():
        num = int(path.split('/')[1].split('_')[0])
        phase = path.split('/')[0]

        objectives = [
            f"Understand {info['topics'][0]}",
            f"Learn {info['topics'][1]}",
            f"Master {info['topics'][2]}",
            "Apply concepts in practical scenarios",
            "Measure and analyze performance"
        ]

        concepts = f"""## Concept: {info['title']}

**Topics Covered:**
{chr(10).join(f'- {topic}' for topic in info['topics'])}

**Key Concepts:**
This notebook covers {info['topics'][0]} in the context of {info['phase']}."""

        notebook = generate_notebook(
            num,
            info["title"],
            info["phase"],
            objectives,
            concepts
        )

        filepath = os.path.join(base_dir, path)
        with open(filepath, 'w') as f:
            json.dump(notebook, f, indent=1)
        print(f"Created {filepath}")

if __name__ == "__main__":
    main()
    print("\nAll notebooks generated successfully!")
