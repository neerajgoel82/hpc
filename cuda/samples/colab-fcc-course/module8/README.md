# Module 8: Triton - High-Level GPU Programming

**Part of**: FreeCodeCamp CUDA Programming Course
**Duration**: 4-6 hours
**Difficulty**: Intermediate

---

## Overview

Module 8 introduces Triton, OpenAI's Python-based GPU programming language. Triton lets you write high-performance GPU kernels with Python-like syntax while achieving performance comparable to hand-written CUDA.

---

## What is Triton?

**Triton** is a language and compiler for parallel programming that:
- Uses Python-like syntax
- Automatically handles memory coalescing
- Automatically manages shared memory
- Compiles to highly optimized PTX/SASS
- Achieves 80-95% of hand-tuned CUDA performance
- Significantly faster to develop than CUDA

**Key Benefits**:
- 🚀 10-100x less code than CUDA
- 🎯 Automatic optimization (coalescing, tiling)
- 🐍 Python integration (PyTorch, JAX)
- 📊 Built-in benchmarking
- 🧠 Easier to learn than CUDA

---

## Notebooks in This Module

### 01. Triton Vector Addition (`01_triton_vector_add.ipynb`)
**Duration**: 1.5-2 hours

**Learning Objectives**:
- Install and setup Triton
- Write your first Triton kernel
- Understand Triton programming model
- Benchmark against CUDA and PyTorch
- Use Triton's autotuning

**Key Concepts**:
- `@triton.jit` decorator for kernels
- Block-based programming model
- `tl.load()` and `tl.store()` operations
- `tl.program_id()` for indexing
- Grid and block configuration
- Autotuning with `@triton.autotune`

**What You'll Build**: Vector addition in Triton with benchmarking

**Performance**: Matches CUDA performance with 10x less code

---

### 02. CUDA Softmax Implementation (`02_softmax.ipynb`)
**Duration**: 1.5 hours

**Learning Objectives**:
- Implement softmax in CUDA
- Handle numerical stability (subtract max)
- Use shared memory reduction
- Understand online algorithms
- Benchmark performance

**Key Concepts**:
- Softmax formula: `softmax(x)_i = exp(x_i) / sum(exp(x_j))`
- Numerical stability: subtract max before exp
- Two-pass algorithm: max, then exp/sum
- Online (fused) algorithm: single pass
- Shared memory for reduction

**What You'll Build**: Numerically stable softmax in CUDA

**Challenge**: Softmax is memory-bandwidth bound and tricky to optimize

---

### 03. Triton Softmax (`03_triton_softmax.ipynb`)
**Duration**: 2 hours

**Learning Objectives**:
- Implement softmax in Triton
- Compare Triton vs CUDA complexity
- Achieve competitive performance
- Use Triton's reduction primitives
- Apply autotuning for optimization

**Key Concepts**:
- `tl.max()` and `tl.sum()` reductions
- `tl.exp()` element-wise operations
- Automatic memory management
- Block-level processing
- Kernel fusion

**What You'll Build**: Softmax in Triton (10x less code than CUDA)

**Performance**: 80-95% of hand-tuned CUDA, 10x easier to write

**Comparison**:
- **CUDA softmax**: ~100-150 lines with shared memory, reductions, sync
- **Triton softmax**: ~15-20 lines with automatic optimizations

---

## Learning Path

```
01_triton_vector_add
        ↓
        Learn Triton basics
        ↓
02_softmax (CUDA reference)
        ↓
        Understand algorithm
        ↓
03_triton_softmax
        ↓
        Compare implementations
```

## Prerequisites

### Knowledge
- Completed Modules 5-7
- Understanding of CUDA kernels
- Python programming
- Basic PyTorch familiarity (helpful)

### Technical
- Google Colab with GPU
- PyTorch installed
- Triton library (pip install)

## Success Criteria

By completing Module 8, you should be able to:
- [ ] Write Triton kernels from scratch
- [ ] Convert CUDA kernels to Triton
- [ ] Use Triton's automatic optimizations
- [ ] Benchmark Triton vs CUDA
- [ ] Apply autotuning effectively
- [ ] Integrate Triton with PyTorch
- [ ] Understand when to use Triton vs CUDA
- [ ] Achieve 80-95% of CUDA performance with less code

## Triton vs CUDA Comparison

### Vector Addition

**CUDA** (~50 lines):
```cuda
__global__ void vector_add(float* c, float* a, float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Host code
float *d_a, *d_b, *d_c;
cudaMalloc(&d_a, size);
cudaMalloc(&d_b, size);
cudaMalloc(&d_c, size);
cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

int threads = 256;
int blocks = (n + threads - 1) / threads;
vector_add<<<blocks, threads>>>(d_c, d_a, d_b, n);

cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
```

**Triton** (~15 lines):
```python
@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)

# Launch
grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
vector_add_kernel[grid](a, b, c, n, BLOCK_SIZE=1024)
```

**Key Differences**:
- No explicit memory management in Triton
- Automatic boundary handling with masks
- Block-based thinking instead of thread-based
- Python integration

---

## Triton Programming Model

### Block-Based Execution
- Each program instance processes a **block** of data
- Similar to CUDA thread blocks, but higher level
- Automatic tiling and memory coalescing

### Key Functions

#### Indexing
```python
pid = tl.program_id(axis=0)     # Block ID
offsets = tl.arange(0, BLOCK_SIZE)  # Indices within block
```

#### Memory Operations
```python
# Load with masking
data = tl.load(ptr + offsets, mask=mask, other=0.0)

# Store with masking
tl.store(ptr + offsets, data, mask=mask)
```

#### Element-wise Operations
```python
result = tl.exp(x)      # Exponential
result = tl.sqrt(x)     # Square root
result = tl.log(x)      # Natural log
result = x + y          # Addition (and all arithmetic)
```

#### Reductions
```python
max_val = tl.max(data, axis=0)    # Max reduction
sum_val = tl.sum(data, axis=0)    # Sum reduction
```

### Autotuning
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n'],
)
@triton.jit
def my_kernel(...):
    ...
```

---

## Performance Expectations

| Kernel | CUDA (lines) | Triton (lines) | Performance |
|--------|--------------|----------------|-------------|
| **Vector Add** | ~50 | ~15 | 100% |
| **Softmax** | ~150 | ~20 | 85-95% |
| **Matrix Multiply** | ~200 | ~30 | 80-90% |
| **Flash Attention** | ~500 | ~50 | 90-95% |

**General Rule**: Triton achieves 80-95% of hand-tuned CUDA with 5-10x less code

---

## When to Use Triton vs CUDA

### Use Triton When:
✅ Rapid prototyping needed
✅ Integrating with PyTorch/JAX
✅ Don't need absolute maximum performance
✅ Want automatic optimizations
✅ Team prefers Python
✅ Research and experimentation

### Use CUDA When:
✅ Need absolute peak performance (100%)
✅ Low-level control required
✅ Using CUDA libraries (cuBLAS, cuDNN)
✅ Deploying to diverse hardware
✅ Complex multi-kernel pipelines
✅ Production-critical performance

### Use Both:
✅ Prototype in Triton, optimize critical parts in CUDA
✅ Use Triton for custom ops, CUDA libraries for standard ops
✅ Team has both Python and C++ expertise

---

## Common Pitfalls

1. **Block Size Too Small**
   ```python
   # BAD - too small, underutilizes GPU
   BLOCK_SIZE = 32
   
   # GOOD - use 256, 512, or 1024
   BLOCK_SIZE = 512
   ```

2. **Forgetting Masks**
   ```python
   # Must use mask for partial blocks
   mask = offsets < n
   data = tl.load(ptr + offsets, mask=mask)
   ```

3. **Wrong Grid Size**
   ```python
   # Correct grid calculation
   grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
   # triton.cdiv is ceiling division
   ```

4. **Not Using Autotuning**
   ```python
   # Triton can autotune block sizes
   # Use @triton.autotune for best performance
   ```

## Benchmarking

### Triton Built-in Benchmark
```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'cuda', 'torch'],
        line_names=['Triton', 'CUDA', 'PyTorch'],
        ylabel='GB/s',
        plot_name='performance-comparison',
    )
)
def benchmark(size, provider):
    # Benchmark code
    ...
```

## Time Estimate
- **Fast pace**: 4-6 hours
- **Moderate pace**: 6-8 hours
- **Deep dive**: 10-12 hours (with experimentation)

## Additional Resources

### Official Resources
- [Triton Documentation](https://triton-lang.org/)
- [Triton GitHub](https://github.com/openai/triton)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)

### Learning
- [OpenAI Triton Blog Post](https://openai.com/research/triton)
- [Triton Conference Talks](https://www.youtube.com/results?search_query=triton+gpu+programming)

### Course Resources
- [FreeCodeCamp CUDA Course](https://www.youtube.com/watch?v=86FAWCzIe_4)

## Practice Exercises

1. **Element-wise ops**: Implement ReLU, GELU, sigmoid in Triton
2. **Reductions**: Min, max, mean, variance
3. **Matrix ops**: Transpose, matrix-vector multiply
4. **Custom layers**: Layer normalization, batch normalization
5. **Attention**: Simplified attention mechanism

## Next Module

After completing Module 8, proceed to:
**Module 9: PyTorch CUDA Extensions** - Learn to integrate custom CUDA/Triton kernels into PyTorch for production use.

**Path**: `../module9/README.md`

---

**Pro Tip**: Triton is the future of GPU kernel development. It offers 80-95% of CUDA performance with 10x faster development. Perfect for research, prototyping, and most production use cases!

## Questions to Test Understanding

1. What are the main advantages of Triton over CUDA?
2. How does Triton's programming model differ from CUDA?
3. What is autotuning and how do you use it?
4. When should you use CUDA instead of Triton?
5. How do you handle boundary conditions in Triton?
6. What is a typical performance ratio: Triton vs hand-tuned CUDA?
7. How do you integrate Triton kernels with PyTorch?
8. What is the purpose of masks in Triton?

If you can write efficient Triton kernels and understand the trade-offs vs CUDA, you're ready for Module 9! 🚀
