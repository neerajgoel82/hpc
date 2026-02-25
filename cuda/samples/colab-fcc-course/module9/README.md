# Module 9: PyTorch CUDA Extensions

**Part of**: FreeCodeCamp CUDA Programming Course
**Duration**: 3-5 hours
**Difficulty**: Intermediate to Advanced

---

## Overview

Module 9 teaches you how to integrate custom CUDA/Triton kernels into PyTorch. You'll learn to create PyTorch extensions, handle autograd, optimize for training, and deploy custom operations in production models.

---

## What are PyTorch CUDA Extensions?

PyTorch CUDA Extensions allow you to:
- Write custom GPU operations in CUDA/C++
- Integrate seamlessly with PyTorch
- Support automatic differentiation (autograd)
- Achieve maximum performance for custom ops
- Deploy in production PyTorch models

**Use Cases**:
- Custom layers not in PyTorch
- Fused operations for efficiency
- Novel algorithms for research
- Performance-critical operations
- Domain-specific computations

---

## Notebooks in This Module

### 01. Polynomial CUDA Extension (`polynomial_cuda.ipynb`)
**Duration**: 1.5-2 hours

**Learning Objectives**:
- Create your first CUDA extension
- Compile CUDA C++ code with PyTorch
- Implement forward and backward passes
- Integrate with PyTorch autograd
- Handle gradients correctly

**Key Concepts**:
- PyTorch C++ Extension API
- `torch.utils.cpp_extension.load_inline()`
- Forward pass implementation
- Backward pass and gradient computation
- `torch.autograd.Function` wrapper

**What You'll Build**: Custom polynomial activation function

**Example**: `y = a*x^3 + b*x^2 + c*x + d`

**Performance**: 2-5x faster than PyTorch native operations

---

### 02. PyTorch Extension Demo (`pytorch_extension_demo.ipynb`)
**Duration**: 2-3 hours

**Learning Objectives**:
- Build production-ready extensions
- Use setup.py for packaging
- Handle multiple input/output tensors
- Implement complex autograd functions
- Optimize memory usage
- Test and benchmark extensions

**Key Concepts**:
- Extension build systems (setuptools, CMake)
- `setup.py` configuration
- `CppExtension` and `CUDAExtension`
- In-place operations
- Tensor shape handling
- Memory management

**What You'll Build**: Complete PyTorch extension with:
- Custom CUDA kernel
- Python binding
- Autograd support
- Unit tests
- Benchmarks

---

## Extension Types

### 1. Inline Extensions (Rapid Prototyping)
```python
from torch.utils.cpp_extension import load_inline

cuda_source = """
__global__ void my_kernel(...) {
    // CUDA code
}
"""

cpp_source = """
torch::Tensor forward(...) {
    // Launch kernel
}
"""

module = load_inline(
    name='my_extension',
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=['forward'],
    verbose=True
)
```

**Pros**: Fast iteration, no build system
**Cons**: Recompiles every run, not for production

### 2. Setup.py Extensions (Production)
```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_extension',
    ext_modules=[
        CUDAExtension('my_extension', [
            'my_extension.cpp',
            'my_kernel.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

**Pros**: Compiled once, production-ready
**Cons**: Requires build step

---

## Creating a Custom Autograd Function

### Step 1: Write CUDA Kernel
```cpp
// forward_kernel.cu
__global__ void forward_kernel(
    const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = /* your computation */;
    }
}
```

### Step 2: Write C++ Binding
```cpp
// extension.cpp
#include <torch/extension.h>

torch::Tensor forward_cuda(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    
    forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel()
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Forward (CUDA)");
    m.def("backward", &backward_cuda, "Backward (CUDA)");
}
```

### Step 3: Python Autograd Wrapper
```python
import torch
import my_extension

class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = my_extension.forward(input)
        ctx.save_for_backward(input)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = my_extension.backward(grad_output, input)
        return grad_input

# Use as PyTorch function
def my_op(x):
    return MyFunction.apply(x)
```

---

## Learning Path

```
01_polynomial_cuda
        ↓
        Learn basics
        ↓
02_pytorch_extension_demo
        ↓
        Production patterns
        ↓
        Apply to your models
```

## Prerequisites

### Knowledge
- Completed Modules 5-8
- CUDA kernel programming
- PyTorch basics (tensors, autograd)
- Python and C++ fundamentals

### Technical
- Google Colab with GPU
- PyTorch installed
- CUDA toolkit (included in Colab)
- C++ compiler

## Success Criteria

By completing Module 9, you should be able to:
- [ ] Create inline CUDA extensions
- [ ] Build production extensions with setup.py
- [ ] Implement forward and backward passes
- [ ] Integrate with PyTorch autograd
- [ ] Handle multiple inputs/outputs
- [ ] Manage tensor memory correctly
- [ ] Test and benchmark extensions
- [ ] Deploy custom ops in PyTorch models

## Key Concepts

### PyTorch Tensor Interop
```cpp
// Access tensor data
float* data = tensor.data_ptr<float>();
int size = tensor.numel();
auto sizes = tensor.sizes();  // Shape

// Create tensors
auto output = torch::zeros_like(input);
auto output = torch::empty({N, C, H, W}, input.options());

// Check properties
tensor.is_cuda();  // Is on GPU?
tensor.dtype();    // Data type
tensor.device();   // Device
```

### Error Checking in Extensions
```cpp
#include <torch/extension.h>

// Check CUDA errors
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor forward(torch::Tensor input) {
    CHECK_INPUT(input);
    // ...
}
```

### Gradient Computation
```python
class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Save tensors needed for backward
        ctx.save_for_backward(input)
        output = my_extension.forward(input)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        input, = ctx.saved_tensors
        
        # Compute gradient
        grad_input = my_extension.backward(grad_output, input)
        
        # Return gradient for each input
        return grad_input
```

---

## Performance Expectations

| Operation | PyTorch Native | Custom CUDA | Speedup |
|-----------|----------------|-------------|---------|
| **Custom Activation** | Baseline | 2-5x | ✓ |
| **Fused Op** | Multiple kernels | Single kernel | 3-10x |
| **Custom Layer** | Suboptimal | Optimized | 2-20x |
| **Novel Algorithm** | N/A or slow | Fast | ∞ or 10-100x |

---

## Common Patterns

### 1. Simple Element-wise Operation
```cpp
// CUDA kernel
__global__ void elementwise_kernel(float* out, const float* in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = /* operation on in[idx] */;
    }
}

// C++ wrapper
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    elementwise_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        input.numel()
    );
    return output;
}
```

### 2. Reduction Operation
```cpp
// Implement reduction (sum, max, etc.)
// Use shared memory or atomic operations
// Return scalar or reduced tensor
```

### 3. Fused Operation
```cpp
// Combine multiple ops in one kernel
// Example: Batch norm + ReLU
// Reduces memory bandwidth, faster
```

---

## Common Pitfalls

1. **Forgetting Gradient Checks**
   ```python
   # Always test gradients
   from torch.autograd import gradcheck
   
   input = torch.randn(10, requires_grad=True, dtype=torch.double).cuda()
   test = gradcheck(MyFunction.apply, input, eps=1e-6, atol=1e-4)
   print("Gradient check:", test)
   ```

2. **Memory Leaks**
   ```cpp
   // Don't manually manage memory
   // PyTorch handles it
   // Use torch::empty(), torch::zeros(), etc.
   ```

3. **Shape Mismatches**
   ```cpp
   // Always check tensor shapes
   TORCH_CHECK(input.size(0) == weight.size(1),
               "Size mismatch: ", input.size(0), " vs ", weight.size(1));
   ```

4. **Not Handling Multiple Devices**
   ```python
   # Ensure operation works on current device
   output = my_op(input.cuda())  # Not: my_op(input).cuda()
   ```

## Debugging Tips

### Test Extension Compilation
```python
# Verbose output shows compilation
load_inline(..., verbose=True)
```

### Check CUDA Errors
```cpp
// After kernel launch
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
}
```

### Validate Gradients
```python
torch.autograd.gradcheck(MyFunction.apply, inputs)
```

### Profile Performance
```python
import torch.utils.benchmark as benchmark

t = benchmark.Timer(
    stmt='my_op(x)',
    setup='from __main__ import my_op, x',
    globals={'x': x}
)
print(t.timeit(100))
```

## Time Estimate
- **Fast pace**: 3-4 hours
- **Moderate pace**: 5-6 hours
- **Deep dive**: 8-10 hours (with advanced features)

## Additional Resources

### Official Documentation
- [PyTorch C++ Extension Guide](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [Custom Autograd Functions](https://pytorch.org/docs/stable/notes/extending.html)
- [PyTorch Extension Examples](https://github.com/pytorch/extension-cpp)

### Advanced Topics
- [TorchScript Integration](https://pytorch.org/docs/stable/jit.html)
- [Dispatcher and Registration](https://pytorch.org/tutorials/advanced/dispatcher.html)

### Course Resources
- [FreeCodeCamp CUDA Course](https://www.youtube.com/watch?v=86FAWCzIe_4)

## Practice Exercises

1. **Custom activations**: Implement Swish, Mish, custom activation
2. **Fused layers**: BatchNorm + ReLU, Conv + BatchNorm + ReLU
3. **Attention**: Custom attention mechanism
4. **Custom loss**: Implement custom loss function
5. **Sparse ops**: Custom sparse matrix operations

## Production Checklist

Before deploying extensions:
- [ ] Gradient checks pass
- [ ] Unit tests for all functions
- [ ] Benchmarks vs PyTorch native
- [ ] Handle edge cases (empty tensors, etc.)
- [ ] Error messages are clear
- [ ] Documentation written
- [ ] Works on multiple GPU types
- [ ] Memory usage optimized

## Next Steps

**Congratulations!** You've completed the FreeCodeCamp CUDA Course! 🎉

### Continue Learning:
1. **Apply to your projects**: Integrate custom ops in your models
2. **Explore advanced topics**: Multi-GPU, quantization, TorchScript
3. **Contribute**: Share your extensions with the community
4. **Stay updated**: Follow CUDA and PyTorch releases

### Advanced Topics to Explore:
- Multi-GPU training with custom ops
- Quantized operations (INT8, FP16)
- TorchScript compilation
- Distributed training integration
- ONNX export support

---

**Pro Tip**: PyTorch extensions are the bridge between research and production. Master them to deploy cutting-edge algorithms at scale!

## Questions to Test Understanding

1. What are the two ways to create PyTorch extensions?
2. How do you implement backward pass for autograd?
3. What is `ctx.save_for_backward()` used for?
4. How do you test gradient correctness?
5. What's the difference between inline and setup.py extensions?
6. How do you handle multiple input tensors?
7. When should you create a custom extension vs use PyTorch native?
8. How do you ensure your extension works on different devices?

If you can create production-ready PyTorch extensions with custom CUDA kernels, you've mastered GPU computing! 🚀🎓
